# -*- coding: utf-8 -*-
"""诊断：HF（transformers）路径加载/卸载模型后，内存到底还不还。

现场数据（comfyui.log）：
    20:05:45  第 1 轮开始（HF 9B）    可用内存 41.9 GiB
    20:16:06  第 3 轮开始（HF 9B）    可用内存 19.4 GiB      <- 少了 22.5 GiB

已经排除 llama-server：另一次实测里它起了又关，
可用内存 42563MB -> 42739MB，**干净归还**（独立进程退出，OS 直接回收映射）。

于是嫌疑落在 HF 路径：Safetensors 的 from_pretrained 默认走 mmap，
18GB 权重会被映射进**本进程**的地址空间；而 _release() 只做了
    del self._cache[...] -> gc.collect() -> torch.cuda.empty_cache()
这三步全在显存侧，一个字节的 CPU 内存都没还。

本脚本用真实的 _load() / _release() 连跑两轮（对应「跑两次」），
把每一步的 可用内存 / 可用显存 / 进程 RSS 都量出来，看有没有累积。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_mem_hf.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
MODEL_DIR = os.path.join(CV, "models", "text_encoders")
TARGET = "huihui-ai_Huihui-Qwen3.5-9B-abliterated"

# ---- ComfyUI 桩 ----
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
fp.get_input_directory = lambda: MODEL_DIR
fp.get_output_directory = lambda: MODEL_DIR
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
import torch                # noqa: E402
import psutil               # noqa: E402

_GIB = 1024.0 ** 3
BASE = None


def probe(tag):
    global BASE
    vm = psutil.virtual_memory()
    free_cpu = vm.available / _GIB
    rss = psutil.Process().memory_info().rss / _GIB
    try:
        free_gpu, _tot = torch.cuda.mem_get_info()
        free_gpu /= _GIB
    except Exception:
        free_gpu = 0.0
    if BASE is None:
        BASE = (free_cpu, rss)
    d_cpu = free_cpu - BASE[0]
    d_rss = rss - BASE[1]
    print(f"  {tag:<16} 可用内存 {free_cpu:6.2f} GiB ({d_cpu:+6.2f})"
          f" | 可用显存 {free_gpu:6.2f} GiB"
          f" | 进程 RSS {rss:6.2f} GiB ({d_rss:+6.2f})")
    return free_cpu, free_gpu, rss


def main():
    path = os.path.join(MODEL_DIR, TARGET)
    if not os.path.isdir(path):
        print(f"!! 找不到模型目录：{path}")
        return 2
    print(f"模型: {path}")
    print(f"目录大小: {sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / _GIB:.2f} GiB")
    print()

    node = QM.Qwen35BatchImageTagger()

    print("=" * 78)
    print("基线（未加载任何模型）")
    print("=" * 78)
    probe("基线")
    print()

    for i in (1, 2):
        print("=" * 78)
        print(f"第 {i} 轮：加载 -> 卸载")
        print("=" * 78)
        probe(f"第{i}轮-加载前")
        t0 = time.perf_counter()
        node._load(path, "none", "auto")
        print(f"  （加载耗时 {time.perf_counter() - t0:.1f}s）")
        probe(f"第{i}轮-加载后")
        node._release(force=True)
        gc.collect()
        time.sleep(3)
        probe(f"第{i}轮-卸载后")
        print()

    print("=" * 78)
    print("解读")
    print("=" * 78)
    print("  · 「卸载后」若明显低于「基线」且两轮持续走低 -> HF 路径有累积泄漏")
    print("  · 只掉一次、第二轮回到同一水位 -> 是缓存/首次分配效应，不是泄漏")
    print("  · 进程 RSS 不降 而 可用内存也不降 -> 内存留在本进程内（mmap 或堆 arena）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
