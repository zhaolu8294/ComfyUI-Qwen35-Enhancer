# -*- coding: utf-8 -*-
"""真机验证 ComfyUI-Qwen35-Enhancer 节点：加载 Qwen3.5-9B 并跑一次真实改写。"""
import os
import sys
import time

COMFY = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(COMFY, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
sys.path.insert(0, COMFY)
sys.path.insert(0, NODE_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("q35", os.path.join(NODE_DIR, "nodes.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("=" * 70)
print("MODELS_DIR =", m.MODELS_DIR)
print("发现的模型:")
choices = m.Qwen35PromptEnhancer.INPUT_TYPES()["required"]["model_name"][0]
for c in choices:
    print("   -", c)
print("=" * 70)
sys.stdout.flush()

TARGET = None
for c in choices:
    if "Qwen3.5-9B" in c:
        TARGET = c
        break
if TARGET is None:
    TARGET = choices[0]
print("测试目标:", TARGET)
sys.stdout.flush()

node = m.Qwen35PromptEnhancer()

USER = "雨夜霓虹街道上的赛博朋克猫，缓缓走向镜头。"

t0 = time.time()
print(f"\n--- 开始加载 + 推理 (8bit) ---")
sys.stdout.flush()
try:
    out = node.enhance(
        model_name=TARGET,
        system_prompt=m.DEFAULT_SYSTEM_PROMPT,
        user_prompt=USER,
        quantization="8bit",
        attention="sdpa",
        enable_thinking=False,
        image=None,
        keep_model_loaded=False,
        unload_other_models=False,
        temperature=0.4,
        max_new_tokens=1024,
        seed=42,
    )
    dt = time.time() - t0
    text = out[0]
    print(f"\n--- 成功 (耗时 {dt:.1f}s) ---")
    print("输出长度:", len(text), "字符")
    print()
    print("========== 模型输出 ==========")
    print(text)
    print("========== 输出结束 ==========")
    print()
    print("检查项:")
    print("  含 integrated_multimodal_description:", "integrated_multimodal_description" in text)
    print("  含 overall_soundscape:", "overall_soundscape" in text)
    print("  含 non_diegetic_music:", "non_diegetic_music" in text)
    print("  残留 think 标记:", ("<think" in text.lower() or "</think" in text.lower()))
except Exception as e:
    dt = time.time() - t0
    import traceback
    print(f"\n--- 失败 (耗时 {dt:.1f}s) ---")
    traceback.print_exc()
