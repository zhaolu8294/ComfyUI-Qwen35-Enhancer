# -*- coding: utf-8 -*-
"""描述长度档位（desc_length）真机对比。

单元测试只能证明「提示词里那句长度条款被改对了」；**模型会不会照着写长**
只有真机能说明。这个脚本在同一张图、同一套预设上只换 desc_length，
逐档打印输出、字数与 token 数。

跑一次加载 27B Q4 约 110s（冷启动），之后每档几秒。

用法：
    python bench_tag_length.py                       # 造一张室内图，跑 preset/long/extra_long
    python bench_tag_length.py --image my.jpg        # 用自己的图
    python bench_tag_length.py --preset photoreal_zh --levels preset,long,extra_long
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
sys.path.insert(0, NODE_DIR)

import gguf_backend as gb  # noqa: E402


def load_length_helpers():
    """从 nodes.py 里抠出「描述长度档位」那一整段。

    只为几个纯函数把整个节点模块拉进来（连带 comfy / torch）不划算，
    而且这个脚本要在没有 ComfyUI 的进程里也能跑。
    """
    src = open(os.path.join(NODE_DIR, "nodes.py"), encoding="utf-8").read()
    start = src.rindex("# ---", 0, src.index("# 描述长度档位"))
    end = src.index("_IMAGE_EXTS_DEFAULT = ")
    ns = {"re": re}
    exec(compile(src[start:end], "nodes_length_section", "exec"), ns)
    return ns["_resolve_desc_length"], ns["_apply_desc_length"]


def make_scene_png(path, w=768, h=576):
    """造一张有点内容的室内图：窗、木桌、红杯子、椅子、绿植。

    不是照片，但元素够多，足以看出「描述长短」的差别。
    """
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (222, 214, 198))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(h * 0.62), w, h], fill=(150, 118, 84))          # 木地板
    d.rectangle([40, 50, 250, 250], fill=(168, 204, 232),
                outline=(120, 120, 120), width=4)                       # 窗
    d.line([145, 50, 145, 250], fill=(120, 120, 120), width=4)
    d.line([40, 150, 250, 150], fill=(120, 120, 120), width=4)
    d.rectangle([330, 300, 560, 380], fill=(126, 88, 56))               # 桌面
    d.rectangle([345, 380, 360, 470], fill=(96, 66, 42))
    d.rectangle([530, 380, 545, 470], fill=(96, 66, 42))
    d.ellipse([400, 262, 452, 300], fill=(196, 62, 52))                 # 红杯子
    d.rectangle([600, 300, 720, 460], fill=(110, 122, 96))              # 椅子
    d.ellipse([70, 430, 190, 540], fill=(74, 120, 62))                  # 绿植
    im.save(path, "PNG")
    return path


def clause_of(text):
    """把提示词里那句长度条款揪出来，方便直观看这次改成了什么。"""
    for line in str(text or "").split("\n"):
        if ("sentences" in line and "words" in line) or ("句" in line and "字" in line):
            return line.strip()
    return "(没找到长度条款 -> 走的是末尾追加)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="用自己的图；不给就造一张")
    ap.add_argument("--preset", default="scene", help="用哪套打标预设")
    ap.add_argument("--levels", default="preset,long,extra_long")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    resolve, apply_len = load_length_helpers()

    pj = os.path.join(NODE_DIR, "presets", "tagging_system_prompts.json")
    presets = json.load(open(pj, encoding="utf-8"))["presets"]
    if args.preset not in presets:
        print(f"!! 没有预设 {args.preset!r}，可选：{sorted(presets)}")
        return 2
    rec = presets[args.preset]
    lang = rec.get("lang")
    base_prompt = rec["prompt"]

    exe = gb.find_llama_server()
    if not exe:
        print("!! 找不到 llama-server.exe")
        return 2
    mains, projs = gb.list_gguf()
    if not mains:
        print("!! 找不到 gguf")
        return 2
    model = list(mains.values())[0]
    mmproj = list(projs.values())[0] if projs else ""

    img_path = args.image
    if not img_path:
        tmpd = tempfile.mkdtemp(prefix="qwen35_taglen_")
        img_path = make_scene_png(os.path.join(tmpd, "scene.png"))
    with open(img_path, "rb") as fh:
        blob = fh.read()
    ext = os.path.splitext(img_path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")

    print("=" * 74)
    print("preset :", args.preset, f"(lang={lang})")
    print("model  :", os.path.basename(model))
    print("mmproj :", os.path.basename(mmproj) if mmproj else "(无)")
    print("image  :", img_path)
    print("=" * 74)

    cfg = {
        "server_exe": exe, "model": model, "mmproj": mmproj,
        "n_gpu_layers": -1, "context_size": args.ctx, "parallel": 1,
        "kv_cache_type": "q8_0", "flash_attn": True,
        "batch": 2048, "ubatch": 512,
        "reasoning_format": "deepseek", "extra_args": "",
    }
    t0 = time.time()
    srv = gb.acquire(cfg)
    try:
        srv.ensure_ready(on_status=lambda s: print("  " + s, flush=True), timeout=900)
    except gb.GgufError as e:
        print("\n!! 启动失败：\n", e)
        return 3
    print(f"[耗时] 加载 {time.time() - t0:.1f}s\n")

    rows = []
    for lvl in [s.strip() for s in args.levels.split(",") if s.strip()]:
        spec = resolve(lvl, "")
        sys_prompt, how = apply_len(base_prompt, spec, lang)
        print("-" * 74)
        print(f"desc_length = {lvl}   ({spec})   处理方式 = {how}")
        print(f"  长度条款 : {clause_of(sys_prompt)}")
        if how == "append":
            print(f"  追加指令 : {sys_prompt.splitlines()[-1].strip()}")
        try:
            r = srv.chat(system=sys_prompt, user="给这张图打标。",
                         images=[(mime, blob)],
                         max_tokens=args.max_tokens, temperature=0.2)
        except gb.GgufError as e:
            print(f"  !! 推理失败：{e}")
            continue
        txt = (r["text"] or "").strip()
        n_char = len(txt)
        n_word = len(txt.split())
        print(f"  输出     : {n_char} 字符 / {n_word} 词 / "
              f"{r['completion_tokens']} tok")
        print(f"  {txt[:600]}")
        if len(txt) > 600:
            print(f"  …（共 {n_char} 字符）")
        rows.append((lvl, n_char, n_word, r["completion_tokens"]))
        print()

    print("=" * 74)
    print("汇总")
    print(f"  {'档位':<12}{'字符':>8}{'词':>8}{'tok':>8}")
    base = rows[0][1] if rows else 0
    for lvl, c, w, t in rows:
        ratio = f"  (x{c / base:.2f})" if base else ""
        print(f"  {lvl:<12}{c:>8}{w:>8}{t:>8}{ratio}")
    if len(rows) >= 2 and rows[-1][1] > rows[0][1]:
        print(f"\n  [OK] 档位确实把输出拉长了（{rows[0][1]} -> {rows[-1][1]} 字符）")
    elif len(rows) >= 2:
        print("\n  [!!] 变长档位没有让输出明显变长，值得人工看一眼")
    return 0


if __name__ == "__main__":
    code = main()
    print("\n-- 关闭 server --")
    gb.release(force=True)
    print("完成，退出码", code)
    sys.exit(code)
