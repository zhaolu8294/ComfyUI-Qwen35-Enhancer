# -*- coding: utf-8 -*-
"""生成 Qwen3.5 Prompt Enhancer 的两个 ComfyUI 工作流。"""
import json
import os
import sys
import importlib.util

# 直接复用节点里的 DEFAULT_SYSTEM_PROMPT，保证两边永远一致
COMFY = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(COMFY, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
sys.path.insert(0, COMFY)
sys.path.insert(0, NODE_DIR)
_spec = importlib.util.spec_from_file_location("q35nodes", os.path.join(NODE_DIR, "nodes.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
H3_SKILL = _mod.DEFAULT_SYSTEM_PROMPT

MODEL_35 = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"
MODEL_VL8 = "Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]"


def build(with_image=False):
    nodes = []
    links = []
    next_link = [1]
    next_node = [1]

    def make_node(class_type, widget_specs, link_specs=None, output_count=1,
                  size=(480, 400), title=None):
        if link_specs is None:
            link_specs = []
        nid = next_node[0]
        next_node[0] += 1
        node_inputs = []
        for (input_name, input_type, src_node, src_slot) in link_specs:
            link_id = next_link[0]
            next_link[0] += 1
            node_inputs.append({
                "name": input_name, "type": input_type,
                "link": link_id, "slot_index": len(node_inputs),
            })
            links.append([link_id, src_node, src_slot, nid, len(node_inputs) - 1, input_type])
        node_outputs = [{"name": "", "type": "*", "links": []} for _ in range(output_count)]
        node = {
            "id": nid, "type": class_type,
            "pos": [0, 0], "size": list(size), "flags": {}, "order": 0, "mode": 0,
            "inputs": node_inputs, "outputs": node_outputs,
            "widgets_values": list(widget_specs),
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.33"},
        }
        if title:
            node["title"] = title
        nodes.append(node)
        return nid

    user_default = "\u96e8\u591c\u9713\u8679\u8857\u9053\u4e0a\u7684\u8d5b\u535a\u670b\u514b\u732b\uff0c\u7f13\u7f13\u8d70\u5411\u955c\u5934\u3002"

    if with_image:
        # LoadImage
        n_img = make_node("LoadImage", ["example.png"], size=(340, 330),
                          title="参考图（可选）")
        # 主节点：required 6 + optional 6 = 12 个 widget
        n_llm = make_node(
            "Qwen35PromptEnhancer",
            [
                MODEL_35,      # 0 model_name
                H3_SKILL,      # 1 system_prompt
                "\u628a\u8fd9\u5f20\u56fe\u53d8\u6210\u96e8\u591c\u9713\u8679\u8857\u9053\u573a\u666f\uff0c\u6a21\u578b\u7f13\u7f13\u8d70\u5411\u955c\u5934\u3002",  # 2 user_prompt
                "8bit",        # 3 quantization
                "sdpa",        # 4 attention
                False,         # 5 enable_thinking
                False,         # 6 keep_model_loaded
                True,          # 7 unload_other_models
                0.4,           # 8 temperature
                1024,          # 9 max_new_tokens
                42,            # 10 seed
                "",            # 11 custom_model_path
            ],
            link_specs=[("image", "IMAGE", n_img, 0)],
            output_count=1,
            size=(560, 620),
            title="Qwen3.5 / Qwen3-VL Prompt Enhancer",
        )
        name = "prompt_enhancer_qwen35_image.json"
        wf_name = "h3_prompt_enhancer_qwen35_image"
    else:
        n_llm = make_node(
            "Qwen35PromptEnhancer",
            [
                MODEL_35,
                H3_SKILL,
                user_default,
                "8bit",
                "sdpa",
                False,
                False,
                True,
                0.4,
                1024,
                42,
                "",
            ],
            link_specs=[],
            output_count=1,
            size=(560, 620),
            title="Qwen3.5 / Qwen3-VL Prompt Enhancer",
        )
        name = "prompt_enhancer_qwen35_text.json"
        wf_name = "h3_prompt_enhancer_qwen35_text"

    # 排版
    if with_image:
        nodes[0]["pos"] = [0, 0]
        nodes[1]["pos"] = [420, 0]
    else:
        nodes[0]["pos"] = [0, 0]

    # 回填 outputs.links
    for ln in links:
        link_id, src, src_slot, dst, dst_slot, ltype = ln
        for n in nodes:
            if n["id"] == src:
                while len(n["outputs"]) <= src_slot:
                    n["outputs"].append({"name": "", "type": "*", "links": []})
                n["outputs"][src_slot]["links"].append(link_id)
                break

    return name, {
        "last_node_id": next_node[0] - 1,
        "last_link_id": next_link[0] - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.8, "offset": [100, 100]},
            "info": {"name": wf_name, "author": "WorkBuddy"},
        },
        "version": 0.4,
    }


for with_image in (False, True):
    name, wf = build(with_image)
    with open(name, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)

    nids = {n["id"] for n in wf["nodes"]}
    broken = []
    for ln in wf["links"]:
        if ln[1] not in nids:
            broken.append(f"src {ln[1]}")
        if ln[3] not in nids:
            broken.append(f"dst {ln[3]}")
    print(f"{name}: {os.path.getsize(name)} bytes, {len(wf['nodes'])} nodes, "
          f"{len(wf['links'])} links  {'OK' if not broken else broken}")
    for n in wf["nodes"]:
        print(f"   #{n['id']} {n['type']} widgets={len(n['widgets_values'])} inputs={len(n['inputs'])}")
