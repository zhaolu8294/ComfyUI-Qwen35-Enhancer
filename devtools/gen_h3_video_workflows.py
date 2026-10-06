# -*- coding: utf-8 -*-
"""
基于官方 video_minimax_h3_r2v.json 骨架，生成两个端到端工作流：
  1. h3_i2v_qwen35.json       图生视频（首帧 + 可选尾帧）
  2. h3_ref2v_multi_qwen35.json  多图参考生视频（3 张参考图）

关键点：
  * 模型文件名必须换成用户本地实际存在的文件（官方用 pruned / nvfp4_awq，本地没有）
  * Qwen35PromptEnhancer 输出 STRING 直接接 H3 节点的 prompt 输入
  * MiniMaxH3ReferenceToVideo 的 ref_images 是 Autogrow 输入，
    name 必须写成点号路径 ref_images.ref_image_N，索引从 0 开始
"""
import json
import os
import sys
import types
import uuid
import copy

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
TPL = (r"E:\AI\ComfyUI-aki-v3\python\Lib\site-packages"
       r"\comfyui_workflow_templates_json\templates\video_minimax_h3_r2v.json")
OUT = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow"

sys.path.insert(0, os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer"))
_fp = types.ModuleType("folder_paths")
_fp.models_dir = os.path.join(CV, "models")
_fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = _fp
import nodes as QM  # noqa: E402

SYS_PROMPT = QM.DEFAULT_SYSTEM_PROMPT
QWEN_MODEL = "Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]"

# 用户本地实际存在的模型文件
MDL = {
    "unet_ref2v": "minimax_h3_ref2va_int8_convrot.safetensors",
    "unet_fl2va": "minimax_h3_fl2va_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "vae_video": "minimax_h3_video_vae_fp16.safetensors",
    "vae_audio": "minimax_h3_audio_vae_fp32.safetensors",
    "lora_ref2v": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "lora_fl2v": "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
}

# input 目录里实际存在的图
IMG_A = "1 (15).png"
IMG_B = "1 (12) (2).png"
IMG_C = "1 (29) (2).png"


def qwen_widgets(mode, user_prompt, max_images=4):
    """widgets_values 必须严格按 INPUT_TYPES 声明顺序（含 optional 里的 widget 项）。"""
    return [
        QWEN_MODEL,        # 0 model_name
        SYS_PROMPT,        # 1 system_prompt
        user_prompt,       # 2 user_prompt
        "8bit",            # 3 quantization
        "sdpa",            # 4 attention
        False,             # 5 enable_thinking
        mode,              # 6 mode
        max_images,        # 7 max_images
        False,             # 8 keep_model_loaded
        True,              # 9 unload_other_models
        0.4,               # 10 temperature
        1024,              # 11 max_new_tokens
        42,                # 12 seed
        "",                # 13 custom_model_path
    ]


def make_qwen_node(nid, pos, mode, user_prompt, title, input_links):
    """input_links: [(输入名, link_id or None), ...]"""
    inputs = []
    for name, lk in input_links:
        inputs.append({"name": name, "shape": 7, "type": "IMAGE", "link": lk})
    return {
        "id": nid,
        "type": "Qwen35PromptEnhancer",
        "pos": pos,
        "size": [560, 620],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": [{"name": "prompt", "type": "STRING", "links": [279]}],
        "properties": {"cnr_id": "ComfyUI-Qwen35-Enhancer", "ver": "1.0"},
        "title": title,
        "widgets_values": qwen_widgets(mode, user_prompt),
    }


def make_loadimage(nid, pos, filename, title):
    return {
        "id": nid,
        "type": "LoadImage",
        "pos": pos,
        "size": [290, 330],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": None},
        ],
        "properties": {"cnr_id": "comfy-core", "ver": "0.3.33"},
        "title": title,
        "widgets_values": [filename, "image"],
    }


def common_model_swap(nodes, unet, lora, turbo_steps):
    nodes[127]["widgets_values"] = [unet, "default"]
    nodes[128]["widgets_values"] = [MDL["clip"], "minimax", "default"]
    nodes[119]["widgets_values"] = [MDL["vae_video"]]
    nodes[120]["widgets_values"] = [MDL["vae_audio"]]
    nodes[145]["widgets_values"] = [lora, 1]
    nodes[144]["widgets_values"] = [turbo_steps, "fixed"]   # turbo 步数
    nodes[143]["widgets_values"] = [20, "fixed"]            # 非 turbo 步数


# ---------------------------------------------------------------- I2V
def build_i2v():
    d = json.load(open(TPL, encoding="utf-8"))
    d["id"] = str(uuid.uuid4())
    nodes = {n["id"]: n for n in d["nodes"]}

    common_model_swap(nodes, MDL["unet_fl2va"], MDL["lora_fl2v"], 4)

    # 136: ReferenceToVideo -> ImageToVideo
    n136 = nodes[136]
    n136["type"] = "MiniMaxH3ImageToVideo"
    n136["title"] = "MiniMax H3 图生视频"
    n136["pos"] = [-620, 5420]
    n136["size"] = [400, 340]
    n136["inputs"] = [
        {"name": "clip", "type": "CLIP", "link": 272},
        {"name": "vae", "type": "VAE", "link": 273},
        {"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": 279},
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 276},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": 277},
        {"name": "length", "type": "INT", "widget": {"name": "length"}, "link": 275},
        {"name": "first_frame", "shape": 7, "type": "IMAGE", "link": 278},
        {"name": "last_frame", "shape": 7, "type": "IMAGE", "link": 282},
    ]
    n136["widgets_values"] = ["", 1344, 768, 124]

    # 换掉模板配图（用户本地没有官方那两张）
    nodes[137]["widgets_values"] = [IMG_A, "image"]
    nodes[137]["title"] = "首帧图"
    nodes[139]["widgets_values"] = [IMG_B, "image"]
    nodes[139]["title"] = "尾帧图（可选）"

    # 删掉 PrimitiveStringMultiline，换成扩写节点
    d["nodes"] = [n for n in d["nodes"] if n["id"] != 138]
    qwen = make_qwen_node(
        200, [-1520, 6680], "image2video",
        "让首帧里的人物缓缓抬起头，向镜头走来，背景霓虹灯忽明忽暗。",
        "Qwen 提示词扩写（图生视频）",
        [("image", 300), ("image_2", None), ("image_3", None), ("image_4", None)],
    )
    d["nodes"].append(qwen)

    # links：删 audio_vae(274)，改 prompt 槽位，补第一帧->Qwen
    new_links = []
    for l in d["links"]:
        lid, src, so, dst, di, t = l
        if lid == 274:
            continue                      # ImageToVideo 没有 audio_vae
        if dst == 136:
            remap = {279: 2, 276: 3, 277: 4, 275: 5, 278: 6, 282: 7}
            if lid in remap:
                di = remap[lid]
        if lid == 279:
            src, so = 200, 0              # prompt 改由扩写节点提供
        new_links.append([lid, src, so, dst, di, t])
    new_links.append([300, 137, 0, 200, 0, "IMAGE"])
    d["links"] = new_links

    nodes[137]["outputs"][0]["links"] = [278, 300]

    # 去掉参考图节点 139 到 136 之外的多余输出引用
    nodes[139]["outputs"][0]["links"] = [282]

    # ImageToVideo 没有 audio_vae，把音频 VAE 输出里那条已删除的 link 274 清掉
    nodes[120]["outputs"][0]["links"] = [250]

    d["last_node_id"] = 200
    d["last_link_id"] = 300
    d["extra"] = d.get("extra", {})
    return d


# ---------------------------------------------------------------- Ref2V 多图
def build_ref2v():
    d = json.load(open(TPL, encoding="utf-8"))
    d["id"] = str(uuid.uuid4())
    nodes = {n["id"]: n for n in d["nodes"]}

    common_model_swap(nodes, MDL["unet_ref2v"], MDL["lora_ref2v"], 8)

    n136 = nodes[136]
    n136["title"] = "MiniMax H3 多图参考生视频"
    n136["size"] = [400, 400]

    nodes[137]["widgets_values"] = [IMG_A, "image"]
    nodes[137]["title"] = "参考图 1"
    nodes[139]["widgets_values"] = [IMG_B, "image"]
    nodes[139]["title"] = "参考图 2"

    # 第三个参考图插槽原本是空的，接上一张真实图片
    for inp in n136["inputs"]:
        if inp.get("name") == "ref_images.ref_image_2":
            inp["link"] = 300
    nodes[201] = make_loadimage(201, [-350, 5960], IMG_C, "参考图 3")
    d["nodes"].append(nodes[201])

    d["nodes"] = [n for n in d["nodes"] if n["id"] != 138]
    qwen = make_qwen_node(
        200, [-1520, 6680], "reference",
        "以图1的人物外貌与服装、图2的人物、图3的雨夜街景为准，"
        "生成两人在霓虹街头对峙的镜头。",
        "Qwen 提示词扩写（多图参考）",
        [("image", 301), ("image_2", 302), ("image_3", 303), ("image_4", None)],
    )
    d["nodes"].append(qwen)

    new_links = []
    for l in d["links"]:
        lid, src, so, dst, di, t = l
        if lid == 279:
            src, so = 200, 0
        new_links.append([lid, src, so, dst, di, t])
    new_links += [
        [300, 201, 0, 136, 5, "IMAGE"],   # 参考图3 -> ref_image_2
        [301, 137, 0, 200, 0, "IMAGE"],   # 参考图1 -> Qwen image
        [302, 139, 0, 200, 1, "IMAGE"],   # 参考图2 -> Qwen image_2
        [303, 201, 0, 200, 2, "IMAGE"],   # 参考图3 -> Qwen image_3
    ]
    d["links"] = new_links

    nodes[137]["outputs"][0]["links"] = [278, 301]
    nodes[139]["outputs"][0]["links"] = [282, 302]
    nodes[201]["outputs"][0]["links"] = [300, 303]

    d["last_node_id"] = 201
    d["last_link_id"] = 303
    return d


def save(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("written:", p, len(json.dumps(obj)), "chars")


if __name__ == "__main__":
    save(build_i2v(), "h3_i2v_qwen35.json")
    save(build_ref2v(), "h3_ref2v_multi_qwen35.json")
    print("done")
