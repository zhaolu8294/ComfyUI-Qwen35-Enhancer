# -*- coding: utf-8 -*-
"""Qwen35PromptEnhancer 改造后校验：schema 顺序 / mode 注入 / 多图收集"""
import sys, os, types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
sys.path.insert(0, os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer"))

fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp

import nodes as M

print("=== 1. INPUT_TYPES ===")
it = M.Qwen35PromptEnhancer.INPUT_TYPES()
req = list(it["required"].keys())
opt = list(it["optional"].keys())
print("required:", req)
print("optional:", opt)

order = list(req)
for k, v in it["optional"].items():
    if v[0] == "IMAGE":
        continue
    order.append(k)
print("widgets_values 顺序 (%d 项):" % len(order))
for i, k in enumerate(order):
    print("   [%d] %s" % (i, k))

print()
print("=== 2. _inject_mode_hint ===")
sp = M.DEFAULT_SYSTEM_PROMPT
for mode in ("text2video", "image2video", "reference"):
    out = M._inject_mode_hint(sp, mode)
    tag = "IMAGE MODE" if "[IMAGE MODE" in out else ("REFERENCE MODE" if "[REFERENCE MODE" in out else "无")
    before = out.find("Rewrite the user input now:")
    print("   %-12s 长度 %d -> %-4d  注入=%s  收尾句仍在末尾=%s" % (
        mode, len(sp), len(out), tag, before > 0 and out.rstrip().endswith("Rewrite the user input now:")))

print()
print("=== 3. _collect_images ===")
import torch
a = torch.rand(2, 8, 8, 3)
b = torch.rand(1, 8, 8, 3)
c = torch.rand(1, 8, 8, 3)
print("   batch2 + 1 + 1 (max4) ->", len(M.Qwen35PromptEnhancer._collect_images(a, b, c, None, 4)))
print("   batch2 + 1 + 1 (max3) ->", len(M.Qwen35PromptEnhancer._collect_images(a, b, c, None, 3)))
print("   全 None             ->", len(M.Qwen35PromptEnhancer._collect_images(None, None, None, None, 4)))
d = torch.rand(9, 8, 8, 3)
print("   单路 batch9 (max4)  ->", len(M.Qwen35PromptEnhancer._collect_images(d, None, None, None, 4)))

print()
print("=== 4. 模型发现 ===")
found = M.discover_local_models()
for k in found:
    print("   ", k)
