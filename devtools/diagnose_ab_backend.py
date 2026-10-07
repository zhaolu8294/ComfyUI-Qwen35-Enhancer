# -*- coding: utf-8 -*-
"""同配置 A/B：HF 9B vs GGUF 27B，量同一个打标任务的真实耗时。

现场（2026-10-07 22:15，comfyui.log）用的是 HF 9B：
    7 张、235 tok/次、30.40s/次、7.7 tok/s、合计 231s
profile（diagnose_hf_speed.py）已定位：Qwen3.5 是混合架构（32 层里 24 层
GatedDeltaNet 线性注意力），fla 未装 -> transformers 走 torch 回退 ->
**每输出 token 发 3365 个 CUDA kernel**（平均每个只跑 17us）。

本脚本回答「换 GGUF 后端能省多少」：同一批图、同一预设、同一长度档，
两个后端各跑一遍，把 token 数 / prefill / 解码 / tok/s 并排列出来。

不碰现场文件：图复制到临时目录，txt 也写在临时目录。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_ab_backend.py
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
NAMES = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png"]
HF_MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated"

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
import gguf_backend as gb                                 # noqa: E402

FAIL_STATUS = ("no_mark", "empty", "bad_len")


def hr(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def stage():
    tmp = tempfile.mkdtemp(prefix="qwen35_ab_")
    for n in NAMES:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))
    return tmp


def free_vram():
    try:
        f, _t = torch.cuda.mem_get_info()
        return f / 1024 ** 3
    except Exception:
        return 0.0


def run(tag, node, tmp, backend):
    hr(tag)
    print(f"  跑前可用显存 {free_vram():.2f} GiB")
    t0 = time.perf_counter()
    try:
        r = node.tag_folder(
            model_name=(HF_MODEL if backend is None else ""),
            folder_path=tmp,
            system_preset="character", system_prompt="",
            user_prompt=QM.DEFAULT_TAGGER_USER_PROMPT,
            quantization="none", attention="auto",
            enable_thinking=False,
            overwrite="overwrite",
            output_format="raw",
            max_new_tokens=784,
            temperature=0.2, seed=42, max_image_side=1536,
            keep_model_loaded=False, unload_other_models=True,
            show_progress=False,
            bilingual="off", caption_mode="off",
            desc_length="extra_long",
            verify="off",
            backend=backend,
        )
        elapsed = time.perf_counter() - t0
        rep = r["result"][0]
    except BaseException as e:
        elapsed = time.perf_counter() - t0
        import traceback
        print(f"  !! 异常 {type(e).__name__}: {e}")
        traceback.print_exc()
        return elapsed, None, None
    print(f"  跑完耗时 {elapsed:.1f}s，跑后可用显存 {free_vram():.2f} GiB")
    print("  --- 报告 ---")
    for ln in rep.splitlines():
        if ln.strip():
            print("  | " + ln)
    # 从报告里抓关键数字
    n_tok = n_ok = None
    dec_s = None
    for ln in rep.splitlines():
        s = ln.strip()
        if s.startswith("输出 token"):
            try:
                n_tok = float(s.split(":")[1].split("（")[0].strip())
            except Exception:
                pass
        elif s.startswith("成功 / 失败"):
            try:
                n_ok = s.split(":")[1].strip()
            except Exception:
                pass
        elif s.startswith("打标耗时"):
            try:
                dec_s = s.split(":")[1].strip().split("（")[0].strip().rstrip("s")
                dec_s = float(dec_s)
            except Exception:
                pass
    if n_tok and dec_s:
        print(f"  >>> 合计 {n_tok:.0f} tok / {dec_s:.1f}s = "
              f"{n_tok / dec_s:.1f} tok/s   （成功率 {n_ok}）")
    return elapsed, n_tok, dec_s


def main():
    tmp = stage()
    node = QM.Qwen35BatchImageTagger()
    print(f"临时目录: {tmp}")
    print(f"素材: {NAMES}")
    print("配置: character 预设 / extra_long / raw / 上限 784 / 1536 长边 / 思考关 / 自检关")

    res = {}
    res["HF 9B"] = run("A  HF 9B（transformers 进程内）", node, tmp, None)
    # HF 跑完显存要还干净，GGUF 才有可比条件
    time.sleep(3)

    mains, projs = gb.list_gguf()
    hr("B  GGUF 27B（llama-server 独立进程）")
    print(f"  主干: {list(mains.keys())[0]}")
    be = QM.Qwen35GGUFServer().provide(
        model=list(mains.keys())[0], mmproj=list(projs.keys())[0],
        server_exe="auto", context_size=8192, auto_download=False,
    )[0]
    res["GGUF 27B"] = run("B  GGUF 27B 跑标", node, tmp, be)

    hr("汇总")
    print(f"  {'后端':<12}{'总耗时s':>10}{'输出tok':>10}{'解码s':>10}{'tok/s':>10}")
    for k, (el, ntok, dsec) in res.items():
        speed = f"{ntok / dsec:.1f}" if (ntok and dsec) else "-"
        print(f"  {k:<12}{el:>10.1f}"
              f"{(f'{ntok:.0f}' if ntok else '-'):>10}"
              f"{(f'{dsec:.1f}' if dsec else '-'):>10}{speed:>10}")
    if res["HF 9B"][2] and res["GGUF 27B"][2]:
        a = res["HF 9B"][1] / res["HF 9B"][2]
        b = res["GGUF 27B"][1] / res["GGUF 27B"][2]
        print(f"\n  GGUF 比 HF 快 {a / b:.1f} 倍（{a:.1f} -> {b:.1f} tok/s）")

    try:
        gb.release()
    except Exception:
        pass
    print("\n  已释放后端")


if __name__ == "__main__":
    main()
