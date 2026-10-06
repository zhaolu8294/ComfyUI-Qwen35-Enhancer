# -*- coding: utf-8 -*-
"""把示例工作流的 Qwen35PromptEnhancer 节点补到 15 项 widgets_values"""
import json
import os
import shutil

SRC = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow"
DST = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer\examples"
FILES = [
    "h3_i2v_qwen35.json",
    "h3_ref2v_multi_qwen35.json",
    "prompt_enhancer_qwen35_text.json",
    "prompt_enhancer_qwen35_image.json",
]

for f in FILES:
    p = os.path.join(SRC, f)
    d = json.load(open(p, encoding="utf-8"))
    fixed = 0
    for n in d["nodes"]:
        if n.get("type") == "Qwen35PromptEnhancer":
            w = n.get("widgets_values")
            if isinstance(w, list) and len(w) == 14:
                w.append(1280)          # max_image_side 默认值
                fixed += 1
            elif isinstance(w, list) and len(w) == 15:
                fixed += 0
    if fixed:
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 同步到发布用的 examples 副本
    shutil.copy2(p, os.path.join(DST, f))
    print(f"{f:<36} 补全节点数={fixed}  -> 已同步 examples/")
print("\n完成")

# 复核
print("\n复核 examples 内 widgets 长度:")
for f in FILES:
    d = json.load(open(os.path.join(DST, f), encoding="utf-8"))
    lens = [len(n["widgets_values"]) for n in d["nodes"] if n.get("type") == "Qwen35PromptEnhancer"]
    print(f"  {f:<36} {lens}")
