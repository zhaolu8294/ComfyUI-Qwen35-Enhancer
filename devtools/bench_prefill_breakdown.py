# -*- coding: utf-8 -*-
"""prefill 30 秒到底花在哪：分段计时 + 同权重 A/B 换后端。

背景：装上 flash-attn、视觉塔后端确认切成 flash_attention_2 之后，
prefill 仍是 30.4s（36 ms / 视觉 token），几乎没变。所以「视觉塔非 flash 路径」
这个假设至少不是全部原因，必须把 30 秒拆开看。

三个问题，一个脚本回答：

  Q1  processor.apply_chat_template（纯 CPU 预处理）占多少？
  Q2  视觉塔 forward 单独多久？换 sdpa / eager / flash_attention_2 各多久？
      —— 同一份权重、同一份输入，只有 forward 里解析出的注意力接口不同，
         forward() 每次调用才读 self.config._attn_implementation，
         所以可以在不重新加载的情况下 A/B/C，这是最干净的对照。
  Q3  LLM 那 1562 个 token 的 prefill 占多少？

只读不写，跑完释放模型。
"""
import importlib.util
import logging
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

torch.manual_seed(0)

node = QM.Qwen35PromptEnhancer()
path = node._resolve_path(
    "Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]", "")

print("=" * 74)
print("加载模型（quant=4bit, attention=auto）")
print("=" * 74)
t0 = time.perf_counter()
model, processor, _ = node._load(path, "4bit", "auto")
print(f"  加载耗时 {time.perf_counter() - t0:.2f}s")
model.eval()
vis = model.model.visual
print(f"  视觉塔后端 = {vis.config._attn_implementation}")

# ---- 与真机日志同尺寸的两张图 ----
sizes = [(543, 768), (581, 768)]
pil = []
for w, h in sizes:
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            v = (x * 255) // max(1, w - 1)
            px[x, y] = (v, (255 - v) // 2, (x + y) % 256)
    pil.append(im)

content = [{"type": "image", "image": im} for im in pil]
content.append({"type": "text", "text": "让画面里的人物缓缓抬起头，向镜头走来。"})
messages = [
    {"role": "system", "content": QM.DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": content},
]

print()
print("=" * 74)
print("Q1  processor 预处理（CPU，含图片张量构造）")
print("=" * 74)
kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True,
          return_tensors="pt")
try:
    inputs = processor.apply_chat_template(messages, **kw)
except TypeError:
    inputs = processor.apply_chat_template(messages, **kw)
t = time.perf_counter()
inputs = processor.apply_chat_template(messages, **kw)
t_proc = time.perf_counter() - t
in_len = inputs["input_ids"].shape[1]
n_vtok = QM._count_visual_tokens(inputs)
print(f"  apply_chat_template : {t_proc:.2f}s   ->  {in_len} tok（其中视觉 {n_vtok}）")
inputs = inputs.to(model.device)

# ---- Q2 视觉塔单独计时，三种后端 ----
px = inputs["pixel_values"]
grid = inputs.get("image_grid_thw")
print()
print("=" * 74)
print("Q2  视觉塔 forward（同权重、同输入，只换注意力后端）")
print("=" * 74)
print(f"  pixel_values {tuple(px.shape)}  image_grid_thw {None if grid is None
                                                 else tuple(grid.shape)}")
q2 = {}
for impl in ("eager", "sdpa", "flash_attention_2"):
    vis.config._attn_implementation = impl      # 子配置无下辖子配置，setter 安全
    try:
        with torch.inference_mode():
            model.get_image_features(px, grid)             # 预热
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = model.get_image_features(px, grid)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
        q2[impl] = dt
        try:
            feat = out.last_hidden_state
        except AttributeError:
            feat = out[0] if isinstance(out, (tuple, list)) else out
        shape = tuple(getattr(feat, "shape", ()))
        print(f"  {impl:20s} : {dt:7.2f}s   输出 {shape}")
    except Exception as e:
        print(f"  {impl:20s} : 失败 {type(e).__name__}: {str(e)[:90]}")

# ---- Q3 整段 prefill（视觉塔 + LLM） ----
print()
print("=" * 74)
print("Q3  完整前向（视觉塔 + LLM 的 1562 token prefill）")
print("=" * 74)
vis.config._attn_implementation = "flash_attention_2"
with torch.inference_mode():
    model(**inputs)                                   # 预热
    torch.cuda.synchronize()
    t = time.perf_counter()
    model(**inputs)
    torch.cuda.synchronize()
    t_full = time.perf_counter() - t
print(f"  model(**inputs)  : {t_full:.2f}s")
if "flash_attention_2" in q2:
    print(f"  扣掉视觉塔      : {t_full - q2['flash_attention_2']:.2f}s  "
          f"<- 这就是 LLM 部分")
print(f"  对照：打补丁前（同权重同输入）整段 30.33s、视觉塔 29.30s")

print()
print("=" * 74)
print("结论")
print("=" * 74)
if q2:
    base = q2.get("sdpa") or q2.get("eager")
    fa = q2.get("flash_attention_2")
    if base and fa:
        print(f"  视觉塔 sdpa -> flash_attention_2：{base:.2f}s -> {fa:.2f}s "
              f"（{base / fa:.2f}x）")
    for k, v in q2.items():
        print(f"  {k:20s} {v:7.2f}s   占整段 {v / t_full * 100:5.1f}%"
              if t_full else "")
    print(f"  LLM 部分            {t_full - (fa or 0):7.2f}s   "
          f"占整段 {(t_full - (fa or 0)) / t_full * 100:5.1f}%" if t_full else "")

node._release(force=True)
