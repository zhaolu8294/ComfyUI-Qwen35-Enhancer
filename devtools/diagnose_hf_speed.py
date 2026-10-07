# -*- coding: utf-8 -*-
"""HF 9B 解码只有 7.7 tok/s —— 分层定位慢在哪。

现场（2026-10-07 22:15，comfyui.log）：
    HF 9B（quant=none、flash_attention_2）7 张图，235 tok/次、30.40s/次
    解码 8 tok/s —— 而同一台机器上 GGUF 27B 实测 38~40 tok/s。
    **9B 比 27B 慢 5 倍**，说明瓶颈不在参数量，而在某条实现路径上。

加载日志里那行自曝了嫌疑：
    线性注意力：24 层里 24 层走 torch 回退 —— 逐 token 跑 fp32 小算子，
    实测 1.0ms/层，24 层即占解码约一半。
但按 1.0ms×24=24ms 反推应得 ~20 tok/s，实测只有 8 —— 差 2.5 倍，
所以要么「1.0ms/层」这个估计不是在解码路径下测的，要么另有热点。

本脚本不猜，直接测：
    1. 打印真实层结构与每层的注意力实现（线性 vs 全注意力）
    2. 带图生成 32 token，torch.profiler 抓 CUDA 时间，按算子聚合
    3. 统计 **kernel 调用次数 / 输出 token** —— 小算子淹没的典型特征
    4. 纯文本生成（不带图）做对照，排除视觉塔

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_hf_speed.py
"""
from __future__ import annotations

import os
import sys
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
MODEL = os.path.join(CV, "models", "text_encoders",
                     "huihui-ai_Huihui-Qwen3.5-9B-abliterated")
IMG = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test\CellWorks39.jpg"

# ---------------------------------------------------------------- 桩
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
fp.get_input_directory = lambda: os.environ.get("TEMP", ".")
fp.get_output_directory = lambda: os.environ.get("TEMP", ".")
sys.modules["folder_paths"] = fp


class _PB:
    def __init__(self, *a, **k):
        pass

    def update_absolute(self, *a, **k):
        pass


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = _PB
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu


class _IPE(BaseException):
    pass


mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda *a, **k: None
mm.free_memory = lambda *a, **k: None
mm.soft_empty_cache = lambda *a, **k: None
mm.throw_exception_if_processing_interrupted = lambda: None
mm.InterruptProcessingException = _IPE
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import torch                                              # noqa: E402
from torch.profiler import ProfilerActivity, profile      # noqa: E402
import nodes as QM                                        # noqa: E402


def hr(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def layers_of(model):
    """兼容不同 transformers 版本，找出 decoder layer 列表。"""
    for path in ("model.language_model.layers", "model.model.layers",
                 "model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return path, list(obj)
        except AttributeError:
            continue
    return None, []


def attn_kind(layer):
    """看一眼这层的注意力模块是什么实现。"""
    for name in ("self_attn", "attention", "attn", "linear_attn"):
        m = getattr(layer, name, None)
        if m is not None:
            return name, type(m).__name__
    return "-", "-"


def main():
    node = QM.Qwen35BatchImageTagger()

    hr("STEP 0  加载 HF 9B（quant=none / attn=auto）")
    t0 = time.perf_counter()
    model, processor, fresh = node._load(MODEL, "none", "auto")
    print(f"  加载完成 {time.perf_counter() - t0:.1f}s  (新加载={fresh})")

    cfg = model.config
    hr("STEP 1  真实层结构")
    print(f"  架构        : {getattr(cfg, 'architectures', None)}")
    print(f"  模型类      : {type(model).__name__}  ({type(model).__module__})")
    print(f"  隐藏层数    : {getattr(cfg, 'num_hidden_layers', '?')}")
    print(f"  权重精度    : "
          f"{next(model.parameters()).dtype}  设备 {next(model.parameters()).device}")
    lt = getattr(cfg, "layer_types", None)
    if lt:
        from collections import Counter
        print(f"  layer_types : {dict(Counter(lt))}")
    path, layers = layers_of(model)
    print(f"  层列表      : {path}  共 {len(layers)} 层")
    from collections import Counter
    kinds = Counter()
    for i, ly in enumerate(layers):
        a, c = attn_kind(ly)
        kinds[c] += 1
        if i < 6 or i == len(layers) - 1:
            print(f"    [{i:2d}] {a:<12} = {c}")
    if len(layers) > 6:
        print(f"    ... 共 {len(layers)} 层")
    print(f"  注意力实现统计: {dict(kinds)}")

    try:
        import fla  # noqa: F401
        print("  fla(flash-linear-attention): 已装")
    except Exception:
        print("  fla(flash-linear-attention): **未装**（线性注意力只能走 torch 回退）")

    # ---------------- 真实提示词 ----------------
    try:
        rec, name = QM._resolve_tag_preset("character")
        sysp = str(rec.get("prompt") or "").strip()
        fmt = rec.get("format") or "raw"
        print(f"  预设        : character（format={fmt}，system {len(sysp)} 字符）")
    except Exception as e:
        print(f"  预设解析失败({e})，改用最简提示词")
        sysp, fmt = "You are a helpful assistant.", "raw"
    userp = "Describe this image in detail."

    # ---------------- 预热 ----------------
    hr("STEP 2  预热（1 次，不计入）")
    t0 = time.perf_counter()
    txt, ntok, pre, dec, cut = node._tag_one_image(
        model, processor, IMG, sysp, userp, 1536, 24, 0.2, False, 1, fmt, 0)
    warm = time.perf_counter() - t0
    print(f"  {ntok} tok / {warm:.2f}s  (prefill {pre:.2f}s、解码 {dec:.2f}s)")

    # ---------------- profiler 抓解码 ----------------
    hr("STEP 3  profile：带图生成 32 token")
    N_OUT = 32
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False) as prof:
        txt, ntok, pre, dec, cut = node._tag_one_image(
            model, processor, IMG, sysp, userp, 1536, N_OUT, 0.2, False, 1, fmt, 0)
    total = pre + dec
    print(f"  本次 {ntok} tok：prefill {pre:.2f}s、解码 {dec:.2f}s")
    if dec > 0 and ntok:
        print(f"  解码速度 {ntok / max(dec, 1e-6):.1f} tok/s   "
              f"（每 token {dec / max(ntok, 1) * 1000:.1f} ms）")

    hr("STEP 4  CUDA 时间 TOP 25（按算子聚合）")
    evs = prof.key_averages()
    rows = []
    for e in evs:
        if e.device_type.name != "CUDA" and e.self_device_time_total <= 0:
            continue
        rows.append((e.self_device_time_total, e.count, e.key))
    rows.sort(reverse=True)
    tot_us = sum(r[0] for r in rows) or 1.0
    print(f"  {'算子':<52}{'次数':>8}{'总耗时ms':>11}{'占比':>8}{'每次us':>10}")
    for us, cnt, key in rows[:25]:
        per = us / cnt if cnt else 0
        print(f"  {key[:52]:<52}{cnt:>8}{us / 1000:>11.1f}{us / tot_us * 100:>7.1f}%{per:>10.1f}")
    print(f"  {'':<52}{'':>8}{tot_us / 1000:>11.1f}")

    hr("STEP 5  kernel 调用次数（小算子淹没的判据）")
    cuda_evs = [e for e in evs if e.device_type.name == "CUDA"]
    total_kernel = sum(e.count for e in cuda_evs)
    print(f"  CUDA 事件总数（≈kernel 调用数）: {total_kernel}")
    if ntok:
        print(f"  平均每个输出 token           : {total_kernel / max(ntok, 1):.0f} 个 kernel")
    print(f"  平均每个 kernel 耗时          : {tot_us / max(total_kernel, 1):.1f} us")
    big = sum(1 for e in cuda_evs if e.count and (e.self_device_time_total / e.count) >= 100)
    print(f"  单次 ≥100us 的算子种类        : {big} / {len(cuda_evs)}")

    hr("STEP 6  纯文本生成对照（不带图，排除视觉塔）")
    try:
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a short paragraph about a cat."}]
        text_in = processor.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        inputs = processor(text=[text_in], return_tensors="pt").to(model.device)
        torch.manual_seed(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=N_OUT, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"  纯文本 {n_new} tok / {dt:.2f}s = {n_new / dt:.1f} tok/s")
        print("  （若这里也慢，瓶颈与视觉塔/图像无关，在解码实现本身）")
    except Exception as e:
        print(f"  纯文本对照失败（不致命）: {type(e).__name__}: {e}")

    hr("STEP 7  收尾")
    model = processor = None
    node._release(force=True)
    print("  已释放")


if __name__ == "__main__":
    main()
