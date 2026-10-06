# -*- coding: utf-8 -*-
"""把双语翻译回退为默认关闭：4 个示例工作流 + 节点内 examples/ 副本。

只改数值，不动结构：Qwen35PromptEnhancer 的 widgets_values[17]（bilingual）
由 "en_then_zh" 改成 "off"。输出端口、PreviewAny、连线全部保留 ——
想让某个工作流用中文预览时，只在界面上把下拉切回 en_then_zh 即可。

幂等：已经是 off 就跳过。
"""
import json
import os

ROOT = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow"
EXAMPLES = (r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes"
            r"\ComfyUI-Qwen35-Enhancer\examples")

NAMES = [
    "h3_i2v_qwen35.json",
    "h3_ref2v_multi_qwen35.json",
    "prompt_enhancer_qwen35_text.json",
    "prompt_enhancer_qwen35_image.json",
]

OLD, NEW = "en_then_zh", "off"
changed, skipped = [], []

for base in (ROOT, EXAMPLES):
    for fn in NAMES:
        p = os.path.join(base, fn)
        if not os.path.isfile(p):
            print(f"  跳过（不存在）: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)

        qn = [n for n in d["nodes"] if n.get("type") == "Qwen35PromptEnhancer"]
        if len(qn) != 1:
            raise SystemExit(f"{fn}: Qwen 节点数量异常 {len(qn)}")
        q = qn[0]
        w = q.get("widgets_values") or []
        if len(w) < 18:
            raise SystemExit(f"{fn}: widgets 数量异常 {len(w)}")

        if w[17] == NEW:
            skipped.append(p)
            continue
        if w[17] != OLD:
            raise SystemExit(f"{fn}: bilingual 值意外 = {w[17]!r}")

        w[17] = NEW
        q["widgets_values"] = w
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
            f.write("\n")
        changed.append(p)
        print(f"  [改] {os.path.relpath(p, base):<36} bilingual: {OLD} -> {NEW}")

print(f"\n改动 {len(changed)} 个，已是 off 跳过 {len(skipped)} 个")
# 复核
for base in (ROOT, EXAMPLES):
    for fn in NAMES:
        p = os.path.join(base, fn)
        if not os.path.isfile(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        q = [n for n in d["nodes"] if n.get("type") == "Qwen35PromptEnhancer"][0]
        assert (q.get("widgets_values") or [])[17] == NEW, f"复核失败: {p}"
        assert len(q.get("outputs") or []) == 2, f"输出端口被误改: {p}"
print("复核通过：8 个文件的 bilingual 均为 off，输出端口仍为 2")
