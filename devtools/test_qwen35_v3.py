# -*- coding: utf-8 -*-
"""改后校验：语法、widgets 顺序、enhance 签名、图片缩放逻辑"""
import os
import sys
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
sys.path.insert(0, NODE)

# mock folder_paths
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp

import torch
import nodes as QM

print("=" * 66)
print("校验 1: INPUT_TYPES 与 widgets 顺序")
print("=" * 66)
it = QM.Qwen35PromptEnhancer.INPUT_TYPES()
req = list(it["required"].keys())
opt_all = list(it.get("optional", {}).keys())
link_inputs = {"image", "image_2", "image_3", "image_4"}
opt_widgets = [k for k in opt_all if k not in link_inputs]
order = req + opt_widgets
print(f"required ({len(req)}): {req}")
print(f"optional 全部 ({len(opt_all)}): {opt_all}")
print(f"其中 link 输入 ({len(link_inputs)}): {sorted(link_inputs)}")
print(f"其中 widget 参数 ({len(opt_widgets)}): {opt_widgets}")
print(f"\n>>> widgets_values 顺序（{len(order)} 项）:")
for i, k in enumerate(order):
    print(f"    [{i:>2}] {k}")
assert len(order) == 15, f"应为 15 项，实际 {len(order)}"
assert order[-1] == "max_image_side", "新参数必须在最后一位（保证旧工作流不错位）"
print("\n  [OK] 15 项，新增 max_image_side 位于末位 → 旧的 14 项工作流会走默认值，不会错位")

print("\n" + "=" * 66)
print("校验 2: enhance() 签名与 INPUT_TYPES 是否一致")
print("=" * 66)
import inspect
sig = inspect.signature(QM.Qwen35PromptEnhancer.enhance)
params = [p for p in sig.parameters if p != "self"]
print(f"enhance 参数 ({len(params)}): {params}")
missing = [k for k in order if k not in params]
extra = [p for p in params if p not in order and p not in link_inputs]
print(f"INPUT_TYPES 有但 enhance 缺失: {missing or '无'}")
print(f"enhance 有但 INPUT_TYPES 未声明（不含 link 输入）: {extra or '无'}")
assert not missing and not extra, "签名不一致"
print("  [OK] 完全一致")

print("\n" + "=" * 66)
print("校验 3: 图片缩放逻辑")
print("=" * 66)
cls = QM.Qwen35PromptEnhancer
for (h, w) in [(1536, 1536), (2000, 1000), (800, 600)]:
    t = torch.zeros((1, h, w, 3), dtype=torch.float32)
    out = cls._collect_images(t, None, None, None, 4, 1280)
    im = out[0]
    print(f"  输入 {w}x{h}  ->  输出 {im.width}x{im.height}  (长边={max(im.size)})")
    assert max(im.size) <= 1280, "缩放后仍超限"
# 不限制时保持原样
t = torch.zeros((1, 1536, 1536, 3), dtype=torch.float32)
im = cls._collect_images(t, None, None, None, 4, 0)[0]
print(f"  max_side=0（不限）: 1536x1536 -> {im.width}x{im.height}")
assert im.size == (1536, 1536)
print("  [OK] 缩放正确，max_side=0 时保持原尺寸")

print("\n" + "=" * 66)
print("校验 4: max_images 截断仍生效")
print("=" * 66)
t = torch.zeros((3, 512, 512, 3), dtype=torch.float32)
out = cls._collect_images(t, t, None, None, 4, 1280)
print(f"  两路各 3 帧 batch，max_images=4 -> 收集 {len(out)} 张")
assert len(out) == 4
print("  [OK]")

print("\n>>> 全部校验通过")
