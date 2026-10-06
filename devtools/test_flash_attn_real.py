# -*- coding: utf-8 -*-
"""flash-attn 真机端到端验证：装好后视觉塔 prefill 到底快了多少。

对照基准（用户 2026-10-07 03:22 实机日志，quant=none/bf16 + attn=sdpa）：
    2 张图 543x768 / 581x768，max_image_side=768
    输入 1647 tok  ->  prefill 33.70s  ->  约 40 ms / 视觉 token

本次用**完全相同的两张图尺寸**、同样 768 上限跑一遍，唯一变量是
attention=auto（装了 flash-attn 后会解析成 flash_attention_2）。
所以两次的 prefill 差值就是 flash-attn 的净收益。

只加载模型、只跑一次生成，不写任何文件。
"""
import importlib.util
import logging
import os
import sys
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

# ---------------------------------------------------------------- 最小 stub
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp

PUSH = []


class FakeProgressBar:
    def __init__(self, total, node_id=None):
        self.total, self.node_id = total, node_id

    def update_absolute(self, value, total=None):
        PUSH.append(value)


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = FakeProgressBar
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu

# 捕获节点日志
LOGS = []


class _Cap(logging.Handler):
    def emit(self, rec):
        LOGS.append(rec.getMessage())


# ---------------------------------------------------------------- 载入节点
spec = importlib.util.spec_from_file_location("q35n", os.path.join(NODE_DIR, "nodes.py"))
QM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(QM)
QM.logger.addHandler(_Cap())
QM.logger.setLevel(logging.INFO)

print("=" * 74)
print("0) 后端解析")
print("=" * 74)
impl, note = QM._resolve_attention("auto")
print(f"  attention='auto'  ->  {impl}   {note}")
print(f"  flash-attn 版本 = {QM._flash_attn_version()}")
assert impl == "flash_attention_2", "auto 没解析到 flash_attention_2"

# ---------------------------------------------------------------- 造图
import torch
from PIL import Image

# 与基准日志里的两张图同尺寸；内容随便，只影响 token 数不影响耗时量级
sizes = [(543, 768), (581, 768)]        # (w, h)，对应日志里的 "543x768, 581x768"
imgs = []
for i, (w, h) in enumerate(sizes):
    arr = torch.zeros((h, w, 3), dtype=torch.float32)
    arr[..., 0] = torch.linspace(0, 1, w).unsqueeze(0)      # 加点渐变，避免纯色
    arr[..., 2] = torch.linspace(1, 0, h).unsqueeze(1)
    imgs.append(arr.unsqueeze(0))                            # (1, H, W, 3)

print()
print("=" * 74)
print("1) 真机加载 + 生成（quant=4bit, attention=auto, max_image_side=768）")
print("=" * 74)
node = QM.Qwen35PromptEnhancer()
t0 = time.perf_counter()
out = node.enhance(
    model_name="Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]",
    system_prompt=QM.DEFAULT_SYSTEM_PROMPT,
    user_prompt="让画面里的人物缓缓抬起头，向镜头走来，背景霓虹灯忽明忽暗。",
    quantization="4bit",
    attention="auto",
    mode="image2video",
    image=imgs[0],
    image_2=imgs[1],
    max_images=4,
    keep_model_loaded=False,
    unload_other_models=False,
    temperature=0.4,
    max_new_tokens=96,
    max_image_side=768,
    show_progress=True,
    progress_interval=5.0,
    bilingual="off",
    unique_id="verify-flash-attn",
)
wall = time.perf_counter() - t0

print()
print("=" * 74)
print("2) 节点日志")
print("=" * 74)
for m in LOGS:
    if any(k in m for k in ("注意力后端", "视觉塔", "权重分布", "耗时分解", "其中",
                            "生成英文", "加载扩写", "合计", "prefill", "警告", "⚠")):
        print("  " + m)

print()
print("=" * 74)
print("3) 结论")
print("=" * 74)
vis_backend = None
for m in LOGS:
    if "注意力后端" in m:
        vis_backend = m
        break
print(f"  视觉塔后端 : {vis_backend}")
en = out[0] if isinstance(out, tuple) else out
print(f"  输出字符数 : {len(en)}")
print(f"  墙钟总耗时 : {wall:.2f}s（含 4bit 量化加载）")
ok = vis_backend and "视觉塔 flash_attention_2" in vis_backend
print()
print("  " + ("✅ 视觉塔已切到 flash_attention_2 —— flash-attn 生效"
               if ok else "❌ 视觉塔没切到 flash 路径，需要排查"))
sys.exit(0 if ok else 1)
