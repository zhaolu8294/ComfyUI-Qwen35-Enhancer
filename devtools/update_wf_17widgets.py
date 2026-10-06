# -*- coding: utf-8 -*-
"""把示例工作流的 Qwen35PromptEnhancer 节点从 15 项补到 17 项 widgets_values

新增末两位：show_progress=True, progress_interval=2.0
（加在参数末尾，旧值索引不受影响；这里显式补齐是为了让工作流自描述、可读）
"""
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
TAIL = [True, 2.0]          # show_progress, progress_interval

for f in FILES:
    p = os.path.join(SRC, f)
    d = json.load(open(p, encoding="utf-8"))
    fixed = 0
    for n in d["nodes"]:
        if n.get("type") == "Qwen35PromptEnhancer":
            w = n.get("widgets_values")
            if not isinstance(w, list):
                continue
            if len(w) == 15:
                w.extend(TAIL)
                fixed += 1
            elif len(w) == 17:
                pass
            else:
                print(f"  !! {f} 节点 id={n['id']} widgets 长度异常: {len(w)}")
    if fixed:
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    shutil.copy2(p, os.path.join(DST, f))
    print(f"{f:<36} 补全节点数={fixed}  -> 已同步 examples/")

print("\n复核：")
ok = True
for f in FILES:
    for base, tag in ((SRC, "工作流"), (DST, "examples")):
        d = json.load(open(os.path.join(base, f), encoding="utf-8"))
        for n in d["nodes"]:
            if n.get("type") == "Qwen35PromptEnhancer":
                w = n["widgets_values"]
                flag = "OK " if len(w) == 17 else "FAIL"
                if len(w) != 17:
                    ok = False
                print(f"  [{flag}] {tag}/{f:<34} 长度={len(w):<3} "
                      f"末两位={w[-2:]!r}")
print("\n全部 17 项" if ok else "\n存在问题")
