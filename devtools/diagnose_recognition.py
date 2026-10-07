# -*- coding: utf-8 -*-
"""诊断：打标为什么会「认错东西」。

用户的原话是「提示词经常会认错东西，有没有办法，自我迭代一下，比如多输出几轮，
对比迭代之类的」。这个脚本把互相纠缠的四个变量拆开单独看：

  A 现状        max_side=1280 + JPEG q92 二次压缩 + thinking off + temp 0.2，跑 1 次
  B 原图无损    max_side=0  -> 直接送原 PNG（不缩、不重编码）+ thinking off
  C 开思考      同 A，但 enable_thinking=True
  D 三路采样    同 A，但 temperature=0.8、seed 42/43/44 -> 看同一张图三次说法差多少
  E 带图自检    拿 A 的输出，把「图 + 描述」一起送回去，要求逐条核验并只改错处
  F 二次自检    拿 E 的结果再核一遍 -> 看会不会收敛（还是越改越走样）

为什么要分开测：这四个变量经常被混为一谈，但治法完全不同。
  · 如果是 B 明显比 A 准 -> 病根在「输入被压糊了」，多跑几轮也救不回来，要先修输入。
  · 如果是 C 明显比 A 准 -> 病根是「没想就直接答」，一行参数就能治。
  · 如果 D 的三次互相打架 -> 说明模型在这张图上本来就不确定，「投票/比对」才有意义。
  · 如果 E 能在 A 的基础上改对 -> 自我迭代成立，可以做成节点选项。

判分靠人眼：脚本只负责把每档的输出并排打出来，ground truth 由人对照图片确认。

用法：
    python diagnose_recognition.py                       # 默认两张素材图
    python diagnose_recognition.py --preset photoreal
    python diagnose_recognition.py --images a.png b.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
sys.path.insert(0, NODE_DIR)

import gguf_backend as gb  # noqa: E402

PRESET_JSON = os.path.join(NODE_DIR, "presets", "tagging_system_prompts.json")
USER_PROMPT = "给这张图打标。"          # 与 DEFAULT_TAGGER_USER_PROMPT 一致

# 自检用的系统提示词。三条设计要点：
#   1) 强制「逐条列出可核查的断言」—— 而不是问它「你觉得对不对」。
#      泛泛地问，模型会顺着自己刚写的话往下圆（self-consistency bias），
#      把错的也说成对的；拆成一条条具体断言，它才不得不回图上看。
#   2) 每条都要写「图里实际是什么」—— 逼它重新看一眼，而不是复述自己的文字。
#   3) 只改错处、明确禁止顺手润色 —— 否则它会把这道工序当成「再写一遍」，
#      没认错的地方也被改掉，等于白折腾。
CHECK_SYSTEM = """You are a meticulous fact-checker for image captions.

You will be given an image and a caption that was written for it.
Your job is to verify the caption AGAINST THE IMAGE, claim by claim.

Step 1 - List every concrete, checkable claim in the caption. Break it down:
how many subjects, hair style and colour, eye colour, skin tone, expression,
directon of gaze, each piece of clothing, each accessory, each held object,
pose, and anything about the background or location.

Step 2 - For each claim, look at the image again and decide:
  SUPPORTED     the image clearly shows this
  CONTRADICTED  the image clearly shows something else
  UNCLEAR       the image does not let you tell
Always write what the image actually shows, in your own words.

Step 3 - Output a corrected caption. Keep exactly the same style, tone, length
and language as the original. Change ONLY the claims that were CONTRADICTED.
Do NOT add new details. Do NOT re-describe the image from scratch. Do NOT
rewrite sentences that were already correct. If nothing was contradicted,
output the original caption unchanged.

Output format, exactly:

CHECKS:
- <claim> -> <SUPPORTED|CONTRADICTED|UNCLEAR>: <what the image actually shows>
- ...

CORRECTED:
<the final caption>"""


def load_preset(key):
    with open(PRESET_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    presets = data.get("presets") or {}
    if key not in presets:
        raise SystemExit(f"!! 预设 {key!r} 不存在，可选：{list(presets)}")
    rec = presets[key]
    return str(rec.get("prompt") or ""), str(rec.get("lang") or "")


def text_of(r):
    """从响应里取正文，顺带把思考块单独拎出来。"""
    txt = (r.get("text") or "").strip()
    think = (r.get("reasoning") or "").strip()
    return txt, think


def show(idx, title, r, dt, kb=None):
    txt, think = text_of(r)
    n_p = int(r.get("prompt_tokens") or 0)
    n_c = int(r.get("completion_tokens") or 0)
    extra = f"  图片 {kb:.0f} KB" if kb is not None else ""
    print(f"\n[{idx} {title}]")
    print(f"    prompt {n_p} tok / prefill {float(r.get('prefill_s') or 0):.2f}s"
          f" | 生成 {n_c} tok / {dt:.2f}s{extra}")
    if think:
        print(f"    (思考 {len(think)} 字，节选) {think[:180]!r}")
    print(f"    >>> {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="character",
                    help="用哪套预设的系统提示词（默认 character）")
    ap.add_argument("--images", nargs="*", default=[
        r"E:\AI\ComfyUI-aki-v3\ComfyUI\output\Qwen_image_2.1_00249.png",
        r"E:\AI\ComfyUI-aki-v3\ComfyUI\output\Qwen_image_2.1_00259.png",
    ])
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--ctx", type=int, default=16384)
    args = ap.parse_args()

    sys_prompt, lang = load_preset(args.preset)
    print("=" * 74)
    print(f"预设        : {args.preset} (lang={lang})，system {len(sys_prompt)} 字符")
    print(f"user 侧文本 : {USER_PROMPT!r}")
    print(f"对照档位    : A 现状 / B 原图无损 / C 开思考 / D 三路采样 / E 自检 / F 二次自检")
    print("=" * 74)

    exe = gb.find_llama_server()
    mains, projs = gb.list_gguf()
    if not (exe and mains and projs):
        print("!! 缺 llama-server.exe / 主干 gguf / mmproj，先补齐再跑")
        return 2

    cfg = {
        "server_exe": exe,
        "model": list(mains.values())[0],
        "mmproj": list(projs.values())[0],
        "n_gpu_layers": -1,
        "context_size": args.ctx,
        "parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attn": True,
        "batch": 2048,
        "ubatch": 512,
        "reasoning_format": "deepseek",
        "extra_args": "",
    }
    print(f"模型        : {os.path.basename(cfg['model'])}")
    print(f"mmproj      : {os.path.basename(cfg['mmproj'])}")

    # 生产默认参数：打标节点是 temperature=0.2 / max_new_tokens=256 / seed=42
    base = dict(max_tokens=256, temperature=0.2, top_p=0.9, top_k=20,
                min_p=0.0, presence_penalty=0.0, repeat_penalty=1.05, seed=42)

    t0 = time.time()
    srv = gb.acquire(cfg)
    try:
        srv.ensure_ready(on_status=lambda s: print("   " + s, flush=True), timeout=900)
    except gb.GgufError as e:
        print("\n!! 启动失败：\n", e)
        return 3
    print(f"[加载] {time.time() - t0:.1f}s\n")

    try:
        for path in args.images:
            if not os.path.isfile(path):
                print(f"\n!! 跳过（找不到）: {path}")
                continue
            name = os.path.basename(path)
            print("\n" + "#" * 74)
            print(f"# {name}")
            print("#" * 74)

            mime_a, blob_a = gb.encode_image_file(path, args.max_side)
            mime_b, blob_b = gb.encode_image_file(path, 0)      # 原图，不缩不重编码
            ka, kb_ = len(blob_a) / 1024.0, len(blob_b) / 1024.0

            # ---- A 现状 ----
            t = time.time()
            ra = srv.chat(system=sys_prompt, user=USER_PROMPT,
                          images=[(mime_a, blob_a)], enable_thinking=False, **base)
            show("A", f"现状 {args.max_side}px/JPEG 二次编码/thinking off/temp 0.2",
                 ra, time.time() - t, ka)
            cap_a = text_of(ra)[0]

            # ---- B 原图无损 ----
            t = time.time()
            rb = srv.chat(system=sys_prompt, user=USER_PROMPT,
                          images=[(mime_b, blob_b)], enable_thinking=False, **base)
            show("B", "原图无损（不缩、不重编码，PNG 直送）", rb, time.time() - t, kb_)

            # ---- C 开思考 ----
            t = time.time()
            rc = srv.chat(system=sys_prompt, user=USER_PROMPT,
                          images=[(mime_a, blob_a)], enable_thinking=True, **base)
            show("C", "开思考（enable_thinking=True，其余同 A）",
                 rc, time.time() - t, ka)

            # ---- D 三路采样 ----
            print(f"\n[D 三路采样] temperature=0.8，seed 42/43/44（其余同 A）")
            for s in (42, 43, 44):
                kw = dict(base)
                kw.update(temperature=0.8, seed=s)
                t = time.time()
                rd = srv.chat(system=sys_prompt, user=USER_PROMPT,
                              images=[(mime_a, blob_a)], enable_thinking=False, **kw)
                txt, _ = text_of(rd)
                print(f"    D[{s}] {txt}")

            # ---- E 带图自检 ----
            if cap_a:
                q = ("CAPTION TO CHECK:\n" + cap_a)
                t = time.time()
                re_ = srv.chat(system=CHECK_SYSTEM, user=q,
                               images=[(mime_a, blob_a)],
                               max_tokens=768, temperature=0.1, top_p=0.9,
                               top_k=20, min_p=0.0, presence_penalty=0.0,
                               repeat_penalty=1.05, seed=42, enable_thinking=False)
                show("E", "带图自检（拿 A 的文本，逐条核验后只改错处）",
                     re_, time.time() - t, ka)
                body = text_of(re_)[0]

                # ---- F 二次自检：还会不会继续改 ----
                m = body.split("CORRECTED:")
                corrected = (m[-1].strip() if len(m) > 1 else "")
                if corrected:
                    t = time.time()
                    rf = srv.chat(system=CHECK_SYSTEM,
                                  user="CAPTION TO CHECK:\n" + corrected,
                                  images=[(mime_a, blob_a)],
                                  max_tokens=768, temperature=0.1, top_p=0.9,
                                  top_k=20, min_p=0.0, presence_penalty=0.0,
                                  repeat_penalty=1.05, seed=42,
                                  enable_thinking=False)
                    show("F", "二次自检（再核一遍，看是否收敛）",
                         rf, time.time() - t, ka)
                    body2 = text_of(rf)[0]
                    same = body2.split("CORRECTED:")[-1].strip() == corrected
                    print(f"    -> 与 E 的结果{'一致（已收敛）' if same else '**不一致（未收敛，在来回改）**'}")
    finally:
        gb.release(force=True)

    print("\n" + "=" * 74)
    print("判分靠人眼：请对照图片确认 A/B/C 哪档认对了，D 的三次是否互相打架，")
    print("E 是否把 A 的错处改对、F 是否又改回去（来回震荡 = 自检不可靠）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
