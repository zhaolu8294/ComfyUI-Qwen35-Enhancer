# -*- coding: utf-8 -*-
"""真机验证：双语互同步（改中文 -> 英文自动跟上）。

单测用的是假模型，只能证明**编排**是对的；这个脚本证明**模型真的会照做**：
走完整 tag_folder 链路（真 GGUF 模型 + 真图片 + 真写盘），看三件事 ——

  1. 手改中文后，英文那份被重写成**中文校订之后**的内容（不是它自己的旧稿）
  2. 同步确实是「翻译」：中文里新增的画面事实会出现在英文里
  3. 幂等：第三次跑零生成，不会把中文反复重翻

用法：
    python smoke_bilingual_sync.py            # 跑完删掉临时目录
    python smoke_bilingual_sync.py --keep     # 保留目录，方便自己翻文件
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

TMP = tempfile.mkdtemp(prefix="qwen35_bisync_")

# ---- ComfyUI 桩：节点 import 时需要 ----
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
fp.get_input_directory = lambda: TMP
fp.get_output_directory = lambda: TMP
sys.modules["folder_paths"] = fp


class _PB:
    def __init__(self, *a, **k):
        pass

    def update_absolute(self, *a, **k):
        pass


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = _PB
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu

mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda *a, **k: None
mm.free_memory = lambda *a, **k: None
mm.soft_empty_cache = lambda *a, **k: None
mm.throw_exception_if_processing_interrupted = lambda: None


class _IPE(BaseException):
    pass


mm.InterruptProcessingException = _IPE
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import nodes as QM          # noqa: E402
import gguf_backend as gb   # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402


def make_image(path):
    """白底 + 左边红方块 + 右边蓝圆。

    故意选这种「一眼能说出、且两个字幕都容易写对」的内容 ——
    判分靠关键词，图太复杂就没法自动判了。
    """
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([70, 150, 250, 330], fill=(200, 40, 35))
    d.ellipse([390, 140, 570, 320], fill=(40, 80, 200))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    exe = gb.find_llama_server()
    mains, projs = gb.list_gguf()
    if not exe or not mains:
        print("!! 找不到 llama-server 或 gguf —— 先按 README 备好")
        return 2

    be = QM.Qwen35GGUFServer().provide(
        model=list(mains.keys())[0],
        mmproj=list(projs.keys())[0] if projs else gb._NONE_MMPROJ,
        server_exe="auto", context_size=args.ctx, auto_download=False,
    )[0]
    print(be["report"])
    if not be["cfg"].get("mmproj"):
        print("!! 没带 mmproj，图片会被丢掉，这个验证没意义")
        return 2

    node = QM.Qwen35BatchImageTagger()
    common = dict(
        model_name="", folder_path=TMP, system_preset="photoreal",
        system_prompt="",                # 选了预设时它被忽略，但签名上是必填
        backend=be, show_progress=False, unload_other_models=False,
        keep_model_loaded=True,          # 三次调用共用同一个 server，别每次重载 109s
        bilingual="en_then_zh", caption_mode="refine",
        temperature=0.2, enable_thinking=False, overwrite="skip",
    )

    img = os.path.join(TMP, "shot01.png")
    make_image(img)
    en = QM._txt_path_for(img)
    zh = QM._txt_path_for(img, "_zh")
    DRAFT_EN = "a photo, masterpiece, a blue circle on the right"
    DRAFT_ZH = "一张照片，杰作，右边有一个蓝色圆形"
    with open(en, "w", encoding="utf-8") as fh:
        fh.write(DRAFT_EN)
    with open(zh, "w", encoding="utf-8") as fh:
        fh.write(DRAFT_ZH)

    print("\n" + "=" * 72)
    print("第 1 次：双语 refine —— 两份各自校订")
    print("=" * 72)
    t0 = time.perf_counter()
    r1 = node.tag_folder(**common)
    print(r1["result"][0])
    en1 = open(en, encoding="utf-8").read().strip()
    zh1 = open(zh, encoding="utf-8").read().strip()
    print(f"\n  EN -> {en1}")
    print(f"  ZH -> {zh1}")
    print(f"  用时 {time.perf_counter() - t0:.1f}s")

    # ---- 用户的实际动作：只改中文（顺手），加一条画面里确实有的事实 ----
    zh_edit = zh1.rstrip("。. ") + "，左边有一个红色方块。"
    with open(zh, "w", encoding="utf-8") as fh:
        fh.write(zh_edit)
    print("\n" + "=" * 72)
    print("手改中文（只动这一份）")
    print("=" * 72)
    print(f"  ZH -> {zh_edit}")

    print("\n" + "=" * 72)
    print("第 2 次：中文校订 + 英文以中文为源同步")
    print("=" * 72)
    t0 = time.perf_counter()
    r2 = node.tag_folder(**common)
    print(r2["result"][0])
    en2 = open(en, encoding="utf-8").read().strip()
    zh2 = open(zh, encoding="utf-8").read().strip()
    print(f"\n  EN -> {en2}")
    print(f"  ZH -> {zh2}")
    print(f"  用时 {time.perf_counter() - t0:.1f}s")

    print("\n" + "=" * 72)
    print("第 3 次：什么都不改 —— 应当零生成")
    print("=" * 72)
    t0 = time.perf_counter()
    r3 = node.tag_folder(**common)
    print(r3["result"][0])
    print(f"  用时 {time.perf_counter() - t0:.1f}s")

    # ---- 判分 ----
    rep1, rep2, rep3 = r1["result"][0], r2["result"][0], r3["result"][0]
    results = [
        ("第 1 次（两份都还没有输出记录）没被误判成「双边手改」",
         "双边手改" not in rep1),
        ("第 2 次发生了 1 次跨语言同步",
         "跨语言同步 1 次" in rep2),
        ("英文被重写成新内容（与第一次不同）",
         en2 and en2 != en1),
        ("英文里出现了中文新增的画面事实（翻译而非重新描述）",
         any(k in en2.lower() for k in ("red", "square", "block"))),
        ("中文那份保留了你手改的内容",
         "红" in zh2),
        ("第三次零生成（幂等）",
         "待处理      : 0 张 / 0 次生成" in rep3),
        ("第三次那份工作流里没有「双边手改」误报",
         "双边手改" not in rep3),
    ]
    print("\n" + "=" * 72)
    print("判分")
    print("=" * 72)
    bad = 0
    for name, ok in results:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1

    gb.release()                     # 收掉 llama-server，把显存还回去

    print()
    print(f"  临时目录: {TMP}")
    if args.keep:
        print("  （--keep：保留，自己翻里面的 .txt / .orig / .q35state）")
    else:
        shutil.rmtree(TMP, ignore_errors=True)
        print("  已清理")

    print()
    print("全部通过" if not bad else f"存在 {bad} 项问题")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
