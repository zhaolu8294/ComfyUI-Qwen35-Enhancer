# -*- coding: utf-8 -*-
"""节点新增 mode / max_images 后，同步更新已有两个扩写工作流的 widgets_values"""
import json
import os
import uuid

# 原为工作区绝对路径；备份到仓库后改为「脚本自身所在目录」，
# 因为工作流 JSON 与脚本同级存放。
WF = os.path.dirname(os.path.abspath(__file__))
IMG = "1 (15).png"

# 新顺序：[0..5] 不变 -> mode, max_images -> 旧的 [6..11] 后移
def migrate(old, mode, max_images=4):
    assert len(old) == 12, f"预期 12 项，实际 {len(old)}"
    return old[:6] + [mode, max_images] + old[6:]


def update(fname, mode, has_image):
    p = os.path.join(WF, fname)
    d = json.load(open(p, encoding="utf-8"))
    d["id"] = str(uuid.uuid4())

    for n in d["nodes"]:
        if n["type"] == "LoadImage":
            n["widgets_values"] = [IMG, "image"]
            n["title"] = "参考图（最多 4 张，可留空）"
            n["size"] = [340, 330]

        if n["type"] == "Qwen35PromptEnhancer":
            n["widgets_values"] = migrate(n["widgets_values"], mode)
            n["title"] = "Qwen3.5 / Qwen3-VL Prompt Enhancer"
            n["size"] = [560, 660]
            if has_image:
                n["inputs"] = [
                    {"name": "image", "shape": 7, "type": "IMAGE", "link": 1},
                    {"name": "image_2", "shape": 7, "type": "IMAGE", "link": None},
                    {"name": "image_3", "shape": 7, "type": "IMAGE", "link": None},
                    {"name": "image_4", "shape": 7, "type": "IMAGE", "link": None},
                ]
                n["outputs"] = [{"name": "prompt", "type": "STRING", "links": []}]
            else:
                n["inputs"] = []
                n["outputs"] = [{"name": "prompt", "type": "STRING", "links": []}]

    if has_image:
        for n in d["nodes"]:
            if n["type"] == "LoadImage":
                n["outputs"] = [
                    {"name": "IMAGE", "type": "IMAGE", "links": [1]},
                    {"name": "MASK", "type": "MASK", "links": None},
                ]
        d["links"] = [[1, 1, 0, 2, 0, "IMAGE"]]
    else:
        d["links"] = []

    d["last_node_id"] = 2
    d["last_link_id"] = 1 if has_image else 0

    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print("updated:", fname, " nodes:", len(d["nodes"]), " links:", len(d["links"]))


update("prompt_enhancer_qwen35_text.json", "text2video", False)
update("prompt_enhancer_qwen35_image.json", "image2video", True)
print("done")
