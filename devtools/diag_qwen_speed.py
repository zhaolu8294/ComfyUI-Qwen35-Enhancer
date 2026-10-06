# -*- coding: utf-8 -*-
"""扩写耗时诊断 —— Part A：输入构成分析（纯 CPU，不加载模型、不占显存）

目的：量出 prefill 阶段的 token 构成，定位耗时大头。
"""
import os
import sys
import time

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\prompt_generator\Qwen3-VL-8B-Instruct"
NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"

sys.path.insert(0, NODE_DIR)

# 从节点里取真实的 system prompt（不导入 torch，只取常量）
import re
src = open(os.path.join(NODE_DIR, "nodes.py"), encoding="utf-8").read()
m = re.search(r'DEFAULT_SYSTEM_PROMPT = """(.*?)"""', src, re.S)
SYS_PROMPT = m.group(1) if m else ""
print(f"system prompt 字符数: {len(SYS_PROMPT)}")

from transformers import AutoProcessor
from PIL import Image

print("\n[1] 加载 processor ...")
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL_DIR)
print(f"    processor 加载耗时: {time.time()-t0:.2f}s")

USER_TEXT = "一只橘猫在清晨的咖啡馆窗边打哈欠，阳光斜射进来，背景是巴黎街景。"

def count_tokens(images, text=USER_TEXT, system=SYS_PROMPT):
    content = []
    for im in images:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": text})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return inputs["input_ids"].shape[1]

print("\n[2] 不同输入的 prefill token 数（input_ids 长度）")
print(f"    {'配置':<28} {'tokens':>8}   {'相对纯文本':>10}")
print("    " + "-" * 52)

base_only = count_tokens([], system=None, text=USER_TEXT)
print(f"    {'仅用户文本(无 system)':<28} {base_only:>8}   {'1.0x':>10}")

sys_only = count_tokens([], system=SYS_PROMPT, text=USER_TEXT)
print(f"    {'用户文本 + system prompt':<28} {sys_only:>8}   {sys_only/base_only:>9.1f}x")

print()
for size in (256, 512, 768, 1024, 1280, 1536):
    img = Image.new("RGB", (size, size), (128, 128, 128))
    n = count_tokens([img], system=SYS_PROMPT, text=USER_TEXT)
    print(f"    {'+ 1图 ' + str(size) + 'x' + str(size):<28} {n:>8}   {n/base_only:>9.1f}x")

print()
img = Image.new("RGB", (1024, 1024), (128, 128, 128))
n4 = count_tokens([img, img, img, img], system=SYS_PROMPT, text=USER_TEXT)
print(f"    {'+ 4图 1024x1024':<28} {n4:>8}   {n4/base_only:>9.1f}x")

print("\n[3] 结论参考")
print("    system prompt 占比 = (用户文本+system - 仅用户文本)")
print(f"    = {sys_only - base_only} tokens")
