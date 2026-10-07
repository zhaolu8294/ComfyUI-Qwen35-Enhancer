# -*- coding: utf-8 -*-
"""对生成的两个 H3 工作流做结构与资源校验"""
import json
import os

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
# 原为工作区绝对路径；备份到仓库后改为「脚本自身所在目录」，
# 因为工作流 JSON 与脚本同级存放。
WF = os.path.dirname(os.path.abspath(__file__))

DIRS = {
    "UNETLoader": os.path.join(CV, "models", "diffusion_models", "minimax_h3"),
    "CLIPLoader": os.path.join(CV, "models", "text_encoders", "minimax_h3"),
    "VAELoader": os.path.join(CV, "models", "vae", "minimax_h3"),
    "LoraLoaderModelOnly": os.path.join(CV, "models", "loras", "H3"),
    "LoadImage": os.path.join(CV, "input"),
}

IDX = {
    "UNETLoader": 0,
    "CLIPLoader": 0,
    "VAELoader": 0,
    "LoraLoaderModelOnly": 0,
    "LoadImage": 0,
}

problems = []
notes = []


def check(fname):
    global problems, notes
    problems, notes = [], []
    # 绝对路径直接用；否则按 h3_workflow 目录找
    p = fname if os.path.isabs(fname) else os.path.join(WF, fname)
    d = json.load(open(p, encoding="utf-8"))
    nodes = {n["id"]: n for n in d["nodes"]}

    print("=" * 70)
    print(fname)
    print("=" * 70)

    # 1 节点唯一
    ids = [n["id"] for n in d["nodes"]]
    if len(ids) != len(set(ids)):
        problems.append("节点 id 重复")
    print("  节点数:", len(d["nodes"]), " 连线数:", len(d["links"]))

    # 2 link 端点
    for l in d["links"]:
        lid, src, so, dst, di, t = l
        if src not in nodes:
            problems.append(f"link {lid}: 源节点 {src} 不存在")
        if dst not in nodes:
            problems.append(f"link {lid}: 目标节点 {dst} 不存在")
            continue
        if src in nodes:
            outs = nodes[src].get("outputs", [])
            if so >= len(outs):
                problems.append(f"link {lid}: 源 {src} 输出槽 {so} 越界")
        ins = nodes[dst].get("inputs", [])
        if di >= len(ins):
            problems.append(f"link {lid}: 目标 {dst} 输入槽 {di} 越界（共 {len(ins)}）")
        else:
            if ins[di].get("link") != lid:
                problems.append(
                    f"link {lid}: 目标 {dst}.inputs[{di}]({ins[di].get('name')}) "
                    f"记录的 link 是 {ins[di].get('link')}，不一致")

    # 3 每个输入声明的 link 必须存在且指向自己
    linkmap = {l[0]: l for l in d["links"]}
    for n in d["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            lk = inp.get("link")
            if lk is None:
                continue
            if lk not in linkmap:
                problems.append(f"节点 {n['id']} 输入[{i}] 声明的 link {lk} 不存在")
                continue
            l = linkmap[lk]
            if l[3] != n["id"] or l[4] != i:
                problems.append(
                    f"节点 {n['id']} 输入[{i}]({inp.get('name')}) link {lk} "
                    f"实际指向 {l[3]}.{l[4]}")

    # 4 输出声明的 links 必须存在且源是自己
    for n in d["nodes"]:
        for j, out in enumerate(n.get("outputs", [])):
            for lk in (out.get("links") or []):
                if lk not in linkmap:
                    problems.append(f"节点 {n['id']} 输出[{j}] 声明的 link {lk} 不存在")
                    continue
                l = linkmap[lk]
                if l[1] != n["id"] or l[2] != j:
                    problems.append(
                        f"节点 {n['id']} 输出[{j}] link {lk} 实际源自 {l[1]}.{l[2]}")

    # 5 扩写节点 widgets
    for n in d["nodes"]:
        if n["type"] == "Qwen35PromptEnhancer":
            w = n.get("widgets_values", [])
            print(f"  Qwen 节点 id={n['id']}  widgets={len(w)}  mode={w[6]!r}  "
                  f"max_images={w[7]}  model={w[0]!r}")
            print(f"        show_progress={w[15]!r}  progress_interval={w[16]!r}  "
                  f"bilingual={w[17]!r}")
            if len(w) >= 5:
                print(f"        quantization={w[3]!r}  attention={w[4]!r}")
            if len(w) != 18:
                problems.append(f"Qwen 节点 widgets 应为 18 项，实际 {len(w)}")
            if len(w) >= 5 and w[4] not in ("auto", "flash_attention_2", "sdpa", "eager"):
                problems.append(
                    f"Qwen 节点 attention 非法: {w[4]!r}（应为 auto / flash_attention_2 / sdpa / eager）")
            if len(w) >= 5 and w[4] == "sdpa":
                notes.append(
                    "Qwen 节点 attention=sdpa：视觉塔会走逐图切块路径，图片 prefill 会明显偏慢；"
                    "装了 flash-attn 时应设为 auto")
            if len(w) >= 17 and w[15] not in (True, False):
                problems.append(f"Qwen 节点 show_progress 非布尔: {w[15]!r}")

            if len(w) >= 18 and w[17] not in ("off", "en_then_zh"):
                problems.append(f"Qwen 节点 bilingual 非法: {w[17]!r}")
            if w[6] not in ("text2video", "image2video", "reference"):
                problems.append(f"Qwen 节点 mode 非法: {w[6]!r}")
            if "integrated_multimodal_description" not in w[1]:
                problems.append("Qwen 节点 system_prompt 不含 H3 结构说明")
            if not n.get("inputs"):
                notes.append("Qwen 节点无输入槽（纯文本模式）")
            onames = [o.get("name") for o in n.get("outputs", [])]
            print(f"        输出端口: {onames}")
            if onames[:2] != ["prompt", "prompt_zh"]:
                problems.append(f"Qwen 节点输出端口应为 prompt/prompt_zh，实际 {onames}")

    # 5c 批量打标节点 widgets（widgets_values 必须与 INPUT_TYPES 的键顺序逐位对齐）
    for n in d["nodes"]:
        if n["type"] == "Qwen35BatchImageTagger":
            w = n.get("widgets_values", [])
            print(f"  BatchTagger 节点 id={n['id']}  widgets={len(w)}  model={w[0]!r}")
            if len(w) >= 14:
                print(f"        folder_path={w[1]!r}  preset={w[2]!r}  quantization={w[5]!r}  "
                      f"overwrite={w[9]!r}  format={w[13]!r}")
            if len(w) >= 27:
                print(f"        bilingual={w[25]!r}  max_output_chars={w[26]!r}")
            if len(w) != 27:
                problems.append(f"BatchTagger widgets 应为 27 项，实际 {len(w)}")
            if len(w) >= 14:
                # system_preset 是动态下拉（选项来自 presets/tagging_system_prompts.json），
                # 这里只能校验它非空；具体预设名是否有效由 test_qwen35_tagger.py 覆盖。
                if not isinstance(w[2], str) or not w[2]:
                    problems.append(f"BatchTagger system_preset 非法: {w[2]!r}")
                if w[9] not in ("skip", "overwrite"):
                    problems.append(f"BatchTagger overwrite 非法: {w[9]!r}")
                if w[13] not in ("tags_one_line", "raw"):
                    problems.append(f"BatchTagger output_format 非法: {w[13]!r}")
                # 末尾两个是后加的控件（双语开关 / 字符上限）；旧工作流升级后
                # widgets_values 可能只有 25 项，那种情况 ComfyUI 会用控件默认值补上，
                # 不算错误，这里仅在字段存在时校验取值。
                if len(w) >= 27:
                    if w[25] not in ("off", "en_then_zh"):
                        problems.append(f"BatchTagger bilingual 非法: {w[25]!r}")
                    if not isinstance(w[26], int) or w[26] < 0:
                        problems.append(f"BatchTagger max_output_chars 非法: {w[26]!r}")
                # 只有 preset=custom 时 system_prompt 才会被用到，此时它该是默认那份
                if w[2] == "custom" and w[3] and "danbooru" not in str(w[3]):
                    notes.append(
                        "BatchTagger 的 system_prompt 不是默认那份 —— 若是有意改的请忽略")
                elif w[2] != "custom":
                    notes.append(f"BatchTagger 使用预设 system_preset={w[2]!r}，"
                                 f"system_prompt 会被忽略")
            onames_b = [o.get("name") for o in n.get("outputs", [])]
            print(f"        输出端口: {onames_b}")
            if onames_b[:2] != ["report", "tagged"]:
                problems.append(f"BatchTagger 输出端口应为 report/tagged，实际 {onames_b}")
            if not isinstance(w[1], str) if w else False:
                problems.append(f"BatchTagger folder_path 非字符串: {w[1]!r}")
            if w and not w[1]:
                notes.append("BatchTagger 的 folder_path 还是空的，跑之前必须先填")

    # 5b 中文预览节点（接住 prompt_zh，否则中文输出在界面上看不见）
    for n in d["nodes"]:
        if n["type"] == "PreviewAny":
            srcs = [l for l in d.get("links", []) if l[3] == n["id"]]
            print(f"  PreviewAny id={n['id']}  上游 link={[l[0] for l in srcs]}")
            if not srcs:
                problems.append(f"PreviewAny(id={n['id']}) 没有输入连线")
            elif srcs[0][2] != 1:
                # 「接第 2 个输出」这条只针对扩写节点（要接住 prompt_zh）。
                # 批量打标节点只有 report/tagged，接第 1 个（report）才是对的。
                _src_type = nodes.get(srcs[0][1], {}).get("type")
                if _src_type == "Qwen35PromptEnhancer":
                    problems.append(
                        f"PreviewAny(id={n['id']}) 应接 Qwen 第 2 个输出，实际槽 {srcs[0][2]}")
                else:
                    print(f"    （上游是 {_src_type}，接第 {srcs[0][2] + 1} 个输出，跳过该检查）")

    # 6 H3 主节点
    for n in d["nodes"]:
        if n["type"] in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"):
            names = [i.get("name") for i in n.get("inputs", [])]
            print(f"  H3 节点 id={n['id']} type={n['type']}")
            print(f"    inputs: {names}")
            print(f"    widgets: {n.get('widgets_values')}")
            if n["type"] == "MiniMaxH3ReferenceToVideo":
                ri = [x for x in names if x and x.startswith("ref_images.")]
                print(f"    Autogrow 参考图槽: {ri}")

    # 7 资源文件存在性
    for n in d["nodes"]:
        t = n["type"]
        if t in DIRS:
            w = n.get("widgets_values") or []
            if not w:
                continue
            fn = w[IDX[t]]
            full = os.path.join(DIRS[t], fn)
            ok = os.path.isfile(full)
            print(f"  {'OK ' if ok else 'MISS'} {t:<22} {fn}")
            if not ok:
                problems.append(f"{t} 引用的文件不存在: {fn}")

    # 8 调用了哪些模型
    print("  连线摘要:")
    for l in d["links"]:
        lid, src, so, dst, di, t = l
        sn = nodes[src]["type"] if src in nodes else "?"
        dn = nodes[dst]["type"] if dst in nodes else "?"
        if "Qwen" in sn or "Qwen" in dn or "MiniMax" in dn or "LoadImage" in sn:
            print(f"    [{lid}] {sn}({src}).{so} -> {dn}({dst}).{di}  ({t})")

    print()
    if problems:
        print("  !! 问题:")
        for x in problems:
            print("    -", x)
    else:
        print("  结构校验全部通过")
    if notes:
        for x in notes:
            print("  note:", x)
    print()
    return len(problems)


total = 0
for f in ("h3_i2v_qwen35.json", "h3_ref2v_multi_qwen35.json",
          # 批量打标节点的示例：widgets_values 必须与 INPUT_TYPES 逐位对齐，
          # 错位时 ComfyUI 不报错、只是值悄悄串位，所以在这里也卡一道。
          os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer", "examples",
                       "batch_tagger_qwen35.json")):
    total += check(f)
print("=" * 70)
print("总问题数:", total)
