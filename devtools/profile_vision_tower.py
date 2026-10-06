# -*- coding: utf-8 -*-
"""Qwen3-VL 视觉塔 29 秒到底烧在哪：分段计时 + kernel 级 profiler。

已知（bench_prefill_breakdown.py）：
    视觉塔 eager / sdpa / flash_attention_2 = 29.16 / 28.91 / 29.30 s
    LLM 1479 token prefill                  = 1.02 s
    => 慢的是视觉塔，但**与注意力后端无关**。

按计算量估：27 层、hidden 1152、3360 个 patch token
    FLOPs ≈ 2 × 15.2M × 3360 × 27 ≈ 2.8 TFLOP
    4090D bf16 有效算力按 100 TFLOPS 算 ≈ 28 ms
实测 29 s => 慢了约 1000 倍。这个量级不可能是算力，只能是"根本没走 GEMM"
或者"每个 op 都在做 GPU↔CPU 往返"。

所以本脚本做两件事：
  1) 把 forward 拆成 patch_embed / blocks / merger 三段，先定位在哪一段；
  2) 对该段开 torch.profiler，按 CUDA 时间和 CPU 自耗时间各取 top，
     并统计 kernel 启动次数 —— 1000 倍慢通常伴随异常多的 kernel 数。
"""
import importlib.util
import os
import sys
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp
cu = types.ModuleType("comfy.utils")
cu.ProgressBar = type("PB", (), {"__init__": lambda self, *a, **k: None,
                                 "update_absolute": lambda self, *a, **k: None})
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu

spec = importlib.util.spec_from_file_location("q35n", os.path.join(NODE_DIR, "nodes.py"))
QM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QM)

import torch
from PIL import Image

node = QM.Qwen35PromptEnhancer()
path = node._resolve_path(
    "Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]", "")
print("加载模型 ...", flush=True)
model, processor, _ = node._load(path, "4bit", "auto")
model.eval()
vis = model.model.visual

sizes = [(543, 768), (581, 768)]
pil = [Image.new("RGB", wh) for wh in sizes]
for im in pil:
    px = im.load()
    w, h = im.size
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            px[x, y] = ((x * 255) // max(1, w - 1), 128, (x + y) % 256)

messages = [
    {"role": "system", "content": QM.DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": [{"type": "image", "image": im} for im in pil]
                              + [{"type": "text", "text": "描述这个场景。"}]},
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True,
    return_tensors="pt").to(model.device)
pixel_values = inputs["pixel_values"]
grid = inputs["image_grid_thw"]
print(f"pixel_values {tuple(pixel_values.shape)}   grid_thw {tuple(grid.shape)}  "
      f"总 patch token = {int(pixel_values.shape[0])}")

print()
print("=" * 74)
print("1) 视觉塔内部三段计时")
print("=" * 74)


def timed(fn, warmup=1, rep=2):
    with torch.inference_mode():
        for _ in range(warmup):
            r = fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(rep):
            r = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / rep, r


t_total, _ = timed(lambda: vis(pixel_values, grid_thw=grid))
print(f"  整塔 forward          : {t_total:7.2f}s")

# 逐段复刻 Qwen3VLVisionModel.forward 的前半部分
with torch.inference_mode():
    hidden = vis.patch_embed(pixel_values)
    print(f"  patch_embed 输出       : {tuple(hidden.shape)}")

t_patch, _ = timed(lambda: vis.patch_embed(pixel_values))
print(f"  ├ patch_embed         : {t_patch:7.2f}s  ({t_patch / t_total * 100:.1f}%)")

# 单层耗时
blk = vis.blocks[0]
with torch.inference_mode():
    h0 = vis.patch_embed(pixel_values)
    pos_ids = vis.fast_pos_embed_interpolate(grid) if hasattr(vis, "fast_pos_embed_interpolate") else None
    pe = vis.pos_embed.weight
    cu_seqlens = torch.repeat_interleave(
        grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device),
                            cu_seqlens])
    pos_emb = vis.rotary_pos_emb(int(cu_seqlens[-1].item()))
    pos_emb = torch.cat((pos_emb, pos_emb), dim=-1)
    kwargs = {"cu_seqlens": cu_seqlens, "position_embeddings": (pos_emb, pos_emb)}

t_one, _ = timed(lambda: blk(h0, **kwargs))
print(f"  ├ 单层 block ×1       : {t_one:7.2f}s   ×27 = {t_one * 27:7.2f}s")
print(f"  └ merger（仅末次）     : {(t_total - t_patch - t_one * 27):7.2f}s（含位置编码等）")

print()
print("=" * 74)
print("2) kernel 级 profiler（整塔一次 forward）")
print("=" * 74)
with torch.inference_mode():
    vis(pixel_values, grid_thw=grid)          # 预热
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        vis(pixel_values, grid_thw=grid)
        torch.cuda.synchronize()

evts = prof.key_averages()
print(f"  事件总数 {len(evts)}")
print()
print("  --- 按 CUDA 总时间 top 12 ---")
for e in sorted(evts, key=lambda x: -(x.self_device_time_total or 0))[:12]:
    ms = (e.self_device_time_total or 0) / 1000
    if ms < 1:
        break
    print(f"    {ms:9.1f} ms  ×{e.count:<6d}  {e.key[:64]}")
print()
print("  --- 按 CPU 自耗时间 top 12 ---")
for e in sorted(evts, key=lambda x: -(x.self_cpu_time_total or 0))[:12]:
    ms = (e.self_cpu_time_total or 0) / 1000
    if ms < 1:
        break
    print(f"    {ms:9.1f} ms  ×{e.count:<6d}  {e.key[:64]}")

prof_tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
print()
print("  --- profiler 汇总表（sort_by=cuda_time_total）---")
for line in prof_tbl.splitlines():
    print("   ", line)

node._release(force=True)
