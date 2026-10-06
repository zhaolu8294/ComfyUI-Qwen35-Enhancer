# -*- coding: utf-8 -*-
"""
把 4 个示例工作流同步到新节点：
  1) Qwen 节点 widgets 17 -> 18（末尾追加 bilingual="en_then_zh"）
  2) Qwen 节点新增第二个输出端口 prompt_zh
  3) 新增一个 PreviewAny 节点接住 prompt_zh —— 否则中文输出悬空，界面上看不到
幂等：重复执行不会重复插入节点。
"""
import json
import os

W = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow"
FILES = [
    "h3_i2v_qwen35.json",
    "h3_ref2v_multi_qwen35.json",
    "prompt_enhancer_qwen35_text.json",
    "prompt_enhancer_qwen35_image.json",
]
NEW_WIDGET = "en_then_zh"

for fn in FILES:
    p = os.path.join(W, fn)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)

    qn = [n for n in d["nodes"] if n.get("type") == "Qwen35PromptEnhancer"]
    assert len(qn) == 1, f"{fn}: Qwen 节点数量异常 {len(qn)}"
    q = qn[0]

    # ---- 1) widgets 17 -> 18 ----
    w = q.get("widgets_values", [])
    if len(w) == 17:
        w.append(NEW_WIDGET)
    elif len(w) != 18:
        raise SystemExit(f"{fn}: 意外 widgets 数量 {len(w)}")
    q["widgets_values"] = w

    # ---- 2) 追加 prompt_zh 输出端口 ----
    outs = q.get("outputs", [])
    assert outs and outs[0].get("name") == "prompt", f"{fn}: 首个输出不是 prompt"
    if len(outs) == 1:
        outs.append({"name": "prompt_zh", "type": "STRING", "links": []})
    q["outputs"] = outs

    # ---- 3) 新增 PreviewAny（幂等）----
    if any(n.get("type") == "PreviewAny" for n in d["nodes"]):
        print(f"{fn}: 已存在 PreviewAny，仅同步 widgets")
    else:
        new_id = int(d.get("last_node_id", 0)) + 1
        new_link = int(d.get("last_link_id", 0)) + 1
        qpos = q.get("pos", [0, 0])
        qsize = q.get("size") or [560, 660]
        orders = [n.get("order", 0) for n in d["nodes"]]

        prev = {
            "id": new_id,
            "type": "PreviewAny",
            "pos": [qpos[0] + qsize[0] + 60, qpos[1]],
            "size": [390, 420],
            "flags": {},
            "order": (max(orders) + 1) if orders else 0,
            "mode": 0,
            "inputs": [{"name": "source", "type": "*", "link": new_link}],
            "outputs": [{"name": "STRING", "type": "STRING", "links": None}],
            "properties": {"Node name for S&R": "PreviewAny"},
            "widgets_values": [None, None, None],
        }
        d["nodes"].append(prev)
        outs[1]["links"] = [new_link]
        d.setdefault("links", []).append(
            [new_link, q["id"], 1, new_id, 0, "STRING"]
        )
        d["last_node_id"] = new_id
        d["last_link_id"] = new_link
        print(f"{fn}: widgets={len(w)}  PreviewAny id={new_id} link={new_link} "
              f"pos={prev['pos']}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write("\n")

print("\n同步完成")
