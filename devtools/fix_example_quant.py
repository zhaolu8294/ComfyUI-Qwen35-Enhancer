# -*- coding: utf-8 -*-
"""把扩写示例工作流里的 quantization 改为 none。

真机 A/B 结论（详见 README 与「诊断报告_扩写变慢.md」第十三章）：
8bit 相对 bf16/4bit 是**纯负收益** —— 加载慢 73%、每 token 慢 40%、
显存还多占 3.2GiB。示例工作流不该用一个已被证伪的档位当默认。

为什么用脚本改而不是手改 JSON：widgets_values 是按 INPUT_TYPES 的
required+optional 键顺序排的**数组**，手改要数位置，错一位就整体串值，
而且 ComfyUI 不报错、只是值悄悄错位。这里直接向节点要键顺序。
"""
import importlib.util
import json
import os
import sys

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
PLUG = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
EX = os.path.join(PLUG, "examples")
TARGET_TYPE = "Qwen35PromptEnhancer"
NEW_VALUE = "none"

sys.path.insert(0, CV)
import folder_paths  # noqa: E402,F401  真实模块，只为满足 nodes.py 的导入

spec = importlib.util.spec_from_file_location("qw35_nodes", os.path.join(PLUG, "nodes.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

it = m.Qwen35PromptEnhancer.INPUT_TYPES()
keys = list(it["required"]) + list(it["optional"])
if "quantization" not in keys:
    sys.exit("INPUT_TYPES 里找不到 quantization —— 键名变了，脚本需要同步更新")
qidx = keys.index("quantization")
qspec = it["required"].get("quantization") or it["optional"].get("quantization")
print(f"quantization 位于 widgets 第 {qidx} 位（0 起计）")
print(f"该控件可选值: {list(qspec[0])}")
if NEW_VALUE not in qspec[0]:
    sys.exit(f"{NEW_VALUE!r} 不在可选值里，放弃修改")
print()

changed_files, changed_nodes = 0, 0
for fname in sorted(os.listdir(EX)):
    if not fname.endswith(".json") or ".bak_" in fname:
        continue
    p = os.path.join(EX, fname)
    with open(p, "r", encoding="utf-8") as f:
        wf = json.load(f)

    report, dirty = [], False
    for n in wf.get("nodes", []):
        if n.get("type") != TARGET_TYPE:
            continue
        wv = n.get("widgets_values", [])
        if qidx >= len(wv):
            report.append(f"  [node {n.get('id')}] !! widgets 只有 {len(wv)} 项，"
                          f"不足 {qidx + 1}，跳过（可能控件被删过）")
            continue
        old = wv[qidx]
        if old == NEW_VALUE:
            report.append(f"  [node {n.get('id')}] 已是 {old!r}，不动")
            continue
        wv[qidx] = NEW_VALUE
        dirty = True
        changed_nodes += 1
        report.append(f"  [node {n.get('id')}] {old!r} -> {NEW_VALUE!r}")

    if not report:
        continue
    print(f"{fname}:")
    print("\n".join(report))
    if dirty:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)
        changed_files += 1
        print(f"  -> 已写回（{os.path.getsize(p)} 字节）")

print()
print(f"共改写 {changed_files} 个文件 / {changed_nodes} 个节点")
