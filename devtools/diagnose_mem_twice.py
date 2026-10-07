# -*- coding: utf-8 -*-
"""端到端复现「跑两次卡内存」：照搬 ComfyUI 里那两轮的顺序，每一步量内存。

comfyui.log 里的现场：
    20:05:45  第 1 轮 HF   9B（character 预设）开始，可用内存 41.9 GiB
    20:10:17  第 2 轮 GGUF 27B（character 预设 + refine）开始
    20:16:06  第 3 轮 HF   9B 开始，可用内存 **19.4 GiB**   <- 少了 22.5 GiB

两个后端各自已被单独证清白：
    · llama-server —— 起了又关，可用内存 42563MB -> 42739MB；独立进程退出，
      OS 直接回收映射，不留残渣。
    · HF 加载/卸载 —— 连跑两轮，卸载后都精确回到 41.07 GiB，零累积。
      （18 GiB 权重走 mmap 直接上显存，几乎不占 CPU 内存。）

所以在「单个都干净」的前提下，要验的是**两者连着跑、来回切**会不会留下东西。
本脚本复刻的就是这个组合，每步都把 可用内存 / 可用显存 / 进程 RSS 打出来。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_mem_twice.py
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
NAMES = ["CellWorks39.jpg", "Qwen_image_2.1_00026.png"]
HF_MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated"

TMP = tempfile.mkdtemp(prefix="qwen35_twice_")

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
import torch                # noqa: E402
import psutil               # noqa: E402

_G = 1024.0 ** 3
BASE = None


def probe(tag):
    global BASE
    vm = psutil.virtual_memory()
    free_cpu = vm.available / _G
    rss = psutil.Process().memory_info().rss / _G
    try:
        free_gpu, _t = torch.cuda.mem_get_info()
        free_gpu /= _G
    except Exception:
        free_gpu = 0.0
    if BASE is None:
        BASE = (free_cpu, rss)
    print(f"  {tag:<22} 可用内存 {free_cpu:6.2f} GiB ({free_cpu - BASE[0]:+6.2f})"
          f" | 可用显存 {free_gpu:6.2f} GiB | RSS {rss:6.2f} GiB ({rss - BASE[1]:+6.2f})",
          flush=True)
    return free_cpu, free_gpu, rss


def stage():
    """2 张图 + 英文初稿（模拟 refine：旁边已有 txt）。"""
    for n in NAMES:
        base = os.path.splitext(n)[0]
        shutil.copyfile(os.path.join(SRC, n), os.path.join(TMP, n))
        src_txt = os.path.join(SRC, base + ".txt")
        if os.path.isfile(src_txt):
            shutil.copyfile(src_txt, os.path.join(TMP, base + ".txt"))


def main():
    stage()
    node = QM.Qwen35BatchImageTagger()
    common = dict(
        folder_path=TMP, system_preset="character", system_prompt="",
        caption_mode="refine", desc_length="preset", bilingual="off",
        max_new_tokens=1024, temperature=0.2, seed=42, max_image_side=1536,
        overwrite="overwrite", show_progress=False,
        unload_other_models=True, enable_thinking=False,
    )

    print(f"临时目录: {TMP}")
    print()
    print("=" * 88)
    print("STEP 0  基线")
    print("=" * 88)
    probe("基线")
    print()

    # ---- 第 1 轮：HF 9B ----
    print("=" * 88)
    print("STEP 1  HF 9B（transformers，进程内）")
    print("=" * 88)
    probe("HF 轮-开始前")
    t0 = time.perf_counter()
    try:
        r = node.tag_folder(model_name=HF_MODEL, quantization="none",
                            attention="auto", keep_model_loaded=False,
                            backend=None, **common)
        print("  --- HF 轮完整报告 ---")
        for ln in r["result"][0].splitlines():
            print("  | " + ln)
    except BaseException as e:
        import traceback
        print(f"  !! HF 轮异常 {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"  （耗时 {time.perf_counter() - t0:.1f}s）")
    gc.collect()
    time.sleep(2)
    probe("HF 轮-结束后")
    # 直接点名：还有多少 CUDA tensor 活着、合计多大 —— 用来确认
    # 「显存没还」到底是还有引用，还是别的进程在占
    _refs = [o for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda]
    _tot = sum(o.numel() * o.element_size() for o in _refs)
    print(f"  [残留] 活着的 CUDA tensor: {len(_refs)} 个，合计 {_tot / _G:.2f} GiB")
    # 显式再释放一次：用来区分「tag_folder 里根本没调 _release」
    # 还是「调了、但还有别的引用拽着 tensor 不放」
    node._release(force=True)
    gc.collect()
    time.sleep(2)
    probe("HF 轮-显式再释放")
    _refs = [o for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda]
    _tot = sum(o.numel() * o.element_size() for o in _refs)
    print(f"  [残留] 活着的 CUDA tensor: {len(_refs)} 个，合计 {_tot / _G:.2f} GiB")
    print()

    # ---- 第 2 轮：GGUF 27B ----
    print("=" * 88)
    print("STEP 2  GGUF 27B（llama-server 独立进程）")
    print("=" * 88)
    mains, projs = gb.list_gguf()
    be = QM.Qwen35GGUFServer().provide(
        model=list(mains.keys())[0], mmproj=list(projs.keys())[0],
        server_exe="auto", context_size=8192, auto_download=False,
    )[0]
    probe("GGUF 轮-开始前")
    t0 = time.perf_counter()
    try:
        r = node.tag_folder(model_name="", quantization="none", attention="auto",
                            keep_model_loaded=False, backend=be, **common)
        print("  --- GGUF 轮完整报告 ---")
        for ln in r["result"][0].splitlines():
            print("  | " + ln)
    except BaseException as e:
        import traceback
        print(f"  !! GGUF 轮异常 {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"  （耗时 {time.perf_counter() - t0:.1f}s）")
    gc.collect()
    time.sleep(2)
    probe("GGUF 轮-结束后")
    print()

    print("=" * 88)
    print("STEP 3  静置 30s，看有没有延迟回收")
    print("=" * 88)
    time.sleep(30)
    probe("静置后")
    print()

    print("=" * 88)
    print("结论")
    print("=" * 88)
    print("  · 若「GGUF 轮-结束后」比「基线」低 20 GiB 量级 -> 复现成功，凶手就在换后端这一步")
    print("  · 若基本回到基线 -> 这条链路本身不泄漏；现场那 22.5 GiB 另有来源")
    gb.release(force=True)
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
