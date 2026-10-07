# -*- coding: utf-8 -*-
"""HF 9B：ComfyUI 内 8.5 tok/s vs 独立脚本 21.6 tok/s —— 试着定位这 2.7 倍。

同一台机器、同一个模型、同样的预设与上限，两边解码速度差 2.7 倍。
prefill 却几乎一致（1.0~1.7s vs 1.12s），说明差异出在**逐 token 路径**上 ——
而这条路正是 GatedDeltaNet 的 torch 回退：每输出 token 要发 3365 个小 kernel
（diagnose_hf_speed.py 实测），全靠 CPU 逐个发射。

ComfyUI 启动日志里三条设置与独立脚本不同：
    [22:12:02] Device: cuda:0 RTX 4090 D : cudaMallocAsync   <- 脚本用 native
    [22:12:02] Using async weight offloading with 2 streams
    [22:12:02] Enabled pinned memory 26116.0

本脚本验第一条：分配器。它是**进程级**的，可以用环境变量精确复现。

用法（两次各跑一遍，比 tok/s）：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_alloc.py
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync ... python.exe diagnose_alloc.py

只跑 HF，2 张图，不碰现场文件（复制到临时目录）。
"""
from __future__ import annotations

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

ALLOC = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or \
    os.environ.get("PYTORCH_ALLOC_CONF") or "（默认 native）"

# ---------------------------------------------------------------- 桩
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
fp.get_input_directory = lambda: os.environ.get("TEMP", ".")
fp.get_output_directory = lambda: os.environ.get("TEMP", ".")
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


class _IPE(BaseException):
    pass


mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda *a, **k: None
mm.free_memory = lambda *a, **k: None
mm.soft_empty_cache = lambda *a, **k: None
mm.throw_exception_if_processing_interrupted = lambda: None
mm.InterruptProcessingException = _IPE
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import torch                                              # noqa: E402
import nodes as QM                                        # noqa: E402


def main():
    print("=" * 84)
    print("步速对照：分配器")
    print("=" * 84)
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {ALLOC}")
    try:
        print(f"  torch 实际分配器后端      = {torch.cuda.get_allocator_backend()}")
    except Exception as e:
        print(f"  （读不到后端: {e}）")

    tmp = tempfile.mkdtemp(prefix="qwen35_alloc_")
    for n in NAMES:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))

    node = QM.Qwen35BatchImageTagger()
    t0 = time.perf_counter()
    r = node.tag_folder(
        model_name=HF_MODEL, folder_path=tmp,
        system_preset="character", system_prompt="",
        user_prompt=QM.DEFAULT_TAGGER_USER_PROMPT,
        quantization="none", attention="auto",
        enable_thinking=False, overwrite="overwrite",
        output_format="raw", max_new_tokens=784,
        temperature=0.2, seed=42, max_image_side=1536,
        keep_model_loaded=False, unload_other_models=True,
        show_progress=False, bilingual="off", caption_mode="off",
        desc_length="extra_long", verify="off", backend=None,
    )
    elapsed = time.perf_counter() - t0
    print()
    print("  --- 报告关键行 ---")
    for ln in r["result"][0].splitlines():
        s = ln.strip()
        if any(s.startswith(k) for k in
               ("成功 / 失败", "输出 token", "打标耗时", "加载模型", "合计")):
            print("  | " + s)
    print(f"\n  墙钟总耗时 {elapsed:.1f}s")
    print("\n  基线参考（默认 native，3 张图）: 21.6 tok/s  [diagnose_ab_backend.py]")


if __name__ == "__main__":
    main()
