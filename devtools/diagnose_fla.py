# -*- coding: utf-8 -*-
"""装 fla 能不能救回 HF 9B 的解码速度？（隔离加载，不改 ComfyUI 环境）

背景：Qwen3.5 是混合架构，32 层里 24 层是 GatedDeltaNet 线性注意力。
transformers 顶部 `try: from fla...`，缺 fla 时四个融合函数全为 None，
各自回落到 torch 实现 —— 逐 token 解码因此要发 **3365 个 CUDA kernel/token**
（diagnose_hf_speed.py 实测），这是 8~21 tok/s 的机制来源。

其中**解码主循环只依赖 `fused_recurrent_gated_delta_rule`（来自 fla）**：
    self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_...
所以先单独验 fla；causal-conv1d 是另一条（需编译），本次不碰。

做法：fla 用 `pip install --target` 装到独立目录，脚本里 sys.path 前置，
**完全不进 ComfyUI 的 site-packages**。依赖 triton-windows 3.7.0 / einops 本机已有。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_fla.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import types

FLA_DIR = r"C:\Users\ADMIN\WorkBuddy\2026-10-06-18-56-50\h3_workflow\_fla_probe02.2"
CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
NAMES = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png"]
HF_MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated"

# ★ 必须在 import transformers 的模型模块之前生效
sys.path.insert(0, FLA_DIR)

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


def hr(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def main():
    hr("STEP 0  fla 可导入性")
    print(f"  隔离目录: {FLA_DIR}")
    print(f"  目录存在: {os.path.isdir(FLA_DIR)}")
    try:
        import fla
        print(f"  ✓ import fla 成功  版本 {getattr(fla, '__version__', '?')}")
    except Exception as e:
        import traceback
        print(f"  ✗ import fla 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n  -> fla 这条路在当前环境走不通，结论：换 GGUF 后端。")
        return
    try:
        import triton
        print(f"  triton {triton.__version__} (triton-windows)")
    except Exception as e:
        print(f"  triton 不可用: {e}")

    hr("STEP 1  transformers 是否认到 fast path")
    try:
        import torch  # noqa: F401
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        print(f"  is_fast_path_available = {M.is_fast_path_available}")
        for nm in ("causal_conv1d_fn", "causal_conv1d_update",
                   "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                   "FusedRMSNormGated"):
            v = getattr(M, nm, None)
            print(f"    {nm:<34} = {'None（回退 torch）' if v is None else '✓ ' + getattr(v, '__module__', str(type(v)))}")
        print("  注：解码主循环用的是 fused_recurrent_gated_delta_rule，"
              "它非 None 就已经吃到主要收益。")
    except Exception as e:
        print(f"  检查失败: {type(e).__name__}: {e}")

    hr("STEP 2  加载 HF 9B 并打标（同 diagnose_ab_backend 的配置）")
    sys.path.insert(0, NODE_DIR)
    import nodes as QM

    tmp = tempfile.mkdtemp(prefix="qwen35_fla_")
    for n in NAMES:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))
    print(f"  临时目录: {tmp}")

    node = QM.Qwen35BatchImageTagger()
    t0 = time.perf_counter()
    try:
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
        rep = r["result"][0]
    except BaseException as e:
        elapsed = time.perf_counter() - t0
        import traceback
        print(f"  !! 异常 {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n  -> fla 装上了但跑不通（用时 {elapsed:.1f}s），结论：换 GGUF 后端。")
        return

    for ln in rep.splitlines():
        if ln.strip():
            print("  | " + ln)
    print(f"\n  墙钟总耗时 {elapsed:.1f}s")

    hr("STEP 3  与基线对比")
    n_tok = None
    for ln in rep.splitlines():
        s = ln.strip()
        if s.startswith("输出 token"):
            try:
                n_tok = float(s.split(":")[1].split("（")[0].strip())
            except Exception:
                pass
    print("  基线（无 fla）：858 tok / 39.77s = 21.6 tok/s  [diagnose_ab_backend.py]")
    if n_tok:
        print(f"  本次（有 fla）：{n_tok:.0f} tok（见上行打标耗时）")
    print("  说明：两者都用同一脚本环境，可比。ComfyUI 进程内另有一套损耗"
          "（见 diagnose_hf_speed.py 的对照）")


if __name__ == "__main__":
    main()
