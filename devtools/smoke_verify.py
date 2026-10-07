# -*- coding: utf-8 -*-
"""真机验证：一轮带图自检（verify）在完整链路上真的能改对。

单测用的是假模型，只能证明**编排**是对的；这个脚本证明**模型真的会照做** ——
走完整 tag_folder 链路（真 GGUF 后端 + 真图片 + 真写盘），把同一批图在
verify=off / once 下的输出并排打出来，判分靠人眼（ground truth 由人对照图确认）。

顺便验证「输入保真度」那两项改动确实生效：缩图输出 PNG 无损 + 默认 1536。

用法：
    python smoke_verify.py                  # 跑完删临时目录
    python smoke_verify.py --keep           # 保留目录，方便自己翻文件
    python smoke_verify.py --preset scene   # 换预设
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

TMP = tempfile.mkdtemp(prefix="qwen35_verify_")

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

DEFAULT_SRC = [
    r"E:\AI\ComfyUI-aki-v3\ComfyUI\output\Qwen_image_2.1_00249.png",
    r"E:\AI\ComfyUI-aki-v3\ComfyUI\output\Qwen_image_2.1_00259.png",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--preset", default="character")
    ap.add_argument("--src", nargs="*", default=DEFAULT_SRC)
    ap.add_argument("--max-side", type=int, default=1536)
    args = ap.parse_args()

    exe = gb.find_llama_server()
    mains, projs = gb.list_gguf()
    if not exe or not mains:
        print("!! 找不到 llama-server 或 gguf —— 先按 README 备好")
        return 2
    if not projs:
        print("!! 没找到 mmproj，图片会被丢掉，这个验证没意义")
        return 2

    be = QM.Qwen35GGUFServer().provide(
        model=list(mains.keys())[0], mmproj=list(projs.keys())[0],
        server_exe="auto", context_size=args.ctx, auto_download=False,
    )[0]

    # ---- 保真度：确认缩图后是无损 PNG，且尺寸按 max_image_side 走 ----
    print("=" * 74)
    print("输入保真度")
    print("=" * 74)
    for s in args.src:
        if os.path.isfile(s):
            mime, blob = gb.encode_image_file(s, args.max_side)
            from PIL import Image
            import io
            with Image.open(io.BytesIO(blob)) as im:
                wh = im.size
            orig_kb = os.path.getsize(s) / 1024.0
            print(f"  {os.path.basename(s)}: {orig_kb:.0f} KB 原图 -> "
                  f"{mime} {len(blob) / 1024:.0f} KB  长边 {max(wh)}")

    imgs = []
    for s in args.src:
        if os.path.isfile(s):
            dst = os.path.join(TMP, os.path.basename(s))
            shutil.copy2(s, dst)
            imgs.append(dst)
    if not imgs:
        print("!! 一张可用素材都没有")
        return 2

    node = QM.Qwen35BatchImageTagger()
    common = dict(
        model_name="", folder_path=TMP, system_preset=args.preset,
        system_prompt="",                 # 选了预设时它被忽略，但签名上是必填
        backend=be, show_progress=False, unload_other_models=False,
        keep_model_loaded=True,           # 几轮共用同一个 server，别每次重载
        temperature=0.2, enable_thinking=False, overwrite="skip",
        max_image_side=args.max_side,
    )

    def read_all():
        out = {}
        for p in imgs:
            t = QM._txt_path_for(p)
            out[os.path.basename(p)] = (
                open(t, encoding="utf-8").read().strip() if os.path.isfile(t) else ""
            )
        return out

    def wipe():
        for p in imgs:
            t = QM._txt_path_for(p)
            if os.path.isfile(t):
                os.remove(t)

    # ---- 预热：第一次调用要启动 llama-server（冷启动约 109s），不计入对比 ----
    print("\n[预热] 启动 llama-server 并加载模型…", flush=True)
    t0 = time.perf_counter()
    node.tag_folder(verify="off", **common)
    print(f"[预热] 完成，用时 {time.perf_counter() - t0:.1f}s（含模型加载）")
    wipe()

    # ---- 第 1 轮：off ----
    print("\n" + "=" * 74)
    print("第 1 轮：verify=off（生成即定稿）")
    print("=" * 74)
    t0 = time.perf_counter()
    r1 = node.tag_folder(verify="off", **common)
    dt1 = time.perf_counter() - t0
    print(r1["result"][0])
    base = read_all()
    print(f"  >>> 用时 {dt1:.1f}s")

    # ---- 第 2 轮：once ----
    wipe()
    print("\n" + "=" * 74)
    print("第 2 轮：verify=once（生成后再带图核验一轮）")
    print("=" * 74)
    t0 = time.perf_counter()
    r2 = node.tag_folder(verify="once", **common)
    dt2 = time.perf_counter() - t0
    print(r2["result"][0])
    fixed = read_all()
    print(f"  >>> 用时 {dt2:.1f}s")

    # ---- 并排对照 ----
    print("\n" + "=" * 74)
    print("并排对照（判分请对照原图）")
    print("=" * 74)
    n_changed = 0
    for p in imgs:
        k = os.path.basename(p)
        a, b = base.get(k, ""), fixed.get(k, "")
        same = (a == b)
        if not same:
            n_changed += 1
        print(f"\n[{k}]  {'自检改动了' if not same else '自检没动（判定无错）'}")
        print(f"  off  -> {a}")
        print(f"  once -> {b}")

    # ---- 自动判分（能自动判的部分）----
    print("\n" + "=" * 74)
    print("自动判分")
    print("=" * 74)
    results = [
        ("off 那轮报告写明自检是关的", "一轮自检    : 关" in r1["result"][0]),
        ("once 那轮报告写明自检是开的", "一轮自检    : 开" in r2["result"][0]),
        ("once 那轮有自检统计行（跑了 N 次 / M 条被修正）",
         "一轮自检    : 跑了" in r2["result"][0]),
        ("once 比 off 慢（多了一次带图推理）", dt2 > dt1 * 1.4),
        ("两张图都成功写出 txt", all(v for v in fixed.values())),
        ("once 那轮没有把内容写空", all(fixed[k] for k in fixed)),
    ]
    bad = 0
    for name, ok in results:
        print(f"  [{'OK ' if ok else '!! '}] {name}")
        bad += (not ok)
    print(f"\n  off {dt1:.1f}s  vs  once {dt2:.1f}s"
          f"（{dt2 / max(dt1, 1e-9):.2f}x）；{n_changed}/{len(imgs)} 张被自检改动")
    print("\n人眼判分要点：once 那份改动的部分，是不是**对照原图确实改对了**？")
    print("有没有把本来写对的地方也改掉（这是自检最大的风险）？")
    if n_changed == 0:
        print("\n注：这次一张都没被改动。可能是描述本来就没硬错（好事），")
        print("    也可能是自检没按格式输出而回退了 —— 看上面报告里的「未生效」计数。")
    if args.keep:
        print(f"\n(临时目录保留: {TMP})")
    else:
        shutil.rmtree(TMP, ignore_errors=True)
        print("\n(临时目录已清理)")
    return 1 if bad else 0


if __name__ == "__main__":
    code = main()
    gb.release(force=True)
    sys.exit(code)
