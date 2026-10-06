# -*- coding: utf-8 -*-
"""从 INPUT_TYPES 的默认值生成批量打标节点的示例工作流。

为什么不手写 JSON：widgets_values 必须与 INPUT_TYPES 的键顺序**逐位对齐**，
手写时插一个控件就会整体错位，而且错位后 ComfyUI 不报错、只是值悄悄串位。
这里直接读默认值来生成，顺序天然正确。

用法：
    python make_batch_tagger_example.py
输出：
    ../ComfyUI-Qwen35-Enhancer/examples/batch_tagger_qwen35.json
"""
import importlib.util
import json
import os
import sys

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
PLUG = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

sys.path.insert(0, CV)
import folder_paths  # noqa: E402,F401  真实模块，只为满足 nodes.py 的导入

spec = importlib.util.spec_from_file_location("qw35_nodes", os.path.join(PLUG, "nodes.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

it = m.Qwen35BatchImageTagger.INPUT_TYPES()
keys = list(it["required"]) + list(it["optional"])
values = []
for key in keys:
    item = it["required"].get(key) or it["optional"].get(key)
    opts = item[1] if len(item) > 1 else {}
    if "default" in opts:
        values.append(opts["default"])
    else:
        # 形如 (_model_choices(),) 的下拉：默认取第一个选项
        cand = item[0]
        values.append(cand[0] if isinstance(cand, (list, tuple)) and cand else "")

# 模型名用本机真实发现到的第一个，别在示例里留一个本机不存在的名字
choices = list(m._model_choices())
if choices:
    values[0] = choices[0]

wf = {
    "last_node_id": 2,
    "last_link_id": 1,
    "nodes": [
        {
            "id": 1,
            "type": "Qwen35BatchImageTagger",
            "pos": [0, 0],
            "size": [620, 920],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {"name": "report", "type": "STRING", "links": [1]},
                {"name": "tagged", "type": "INT", "links": None},
            ],
            "widgets_values": values,
            "properties": {"Node name for S&R": "Qwen35BatchImageTagger"},
            "title": "Qwen3.5 Batch Image Tagger  ← 先填 folder_path，可先 dry_run 看清单",
            "color": "#232",
            "bgcolor": "#353",
        },
        {
            "id": 2,
            "type": "PreviewAny",
            "pos": [660, 0],
            "size": [460, 620],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [{"name": "source", "type": "*", "link": 1}],
            "outputs": [{"name": "STRING", "type": "STRING", "links": None}],
            "properties": {"Node name for S&R": "PreviewAny"},
            "widgets_values": [None, None, None],
        },
    ],
    "links": [[1, 1, 0, 2, 0, "STRING"]],
    "groups": [],
    "config": {},
    "extra": {
        "ds": {"scale": 0.75, "offset": [80, 60]},
        "info": {"name": "qwen35_batch_image_tagger", "author": "WorkBuddy"},
    },
    "version": 0.4,
    "id": "a7c1e9d4-2b6f-4a53-9c8e-0f1d2a3b4c5d",
}

out = os.path.join(PLUG, "examples", "batch_tagger_qwen35.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(wf, fh, ensure_ascii=False, indent=2)

print(f"widgets 个数 = {len(values)}（INPUT_TYPES 键数 {len(keys)}）")
for i, (k, v) in enumerate(zip(keys, values)):
    disp = (v[:56] + "...") if isinstance(v, str) and len(v) > 56 else v
    print(f"  [{i:>2}] {k:20s} = {disp!r}")
print("已写入:", out)
print("大小:", os.path.getsize(out), "字节")
