# -*- coding: utf-8 -*-
"""复现并定位：GGUF 后端 + refine 模式 → 7 张全「输出为空」。

现场参数（comfyui.log 20:10:17 那轮）：
    推理后端   : llama.cpp / GGUF（27B Q4_K_XL + mmproj）
    系统提示词 : character 预设
    打标模式   : refine（把旁边已有 .txt 当初稿）
    描述长度   : preset（desc_length=custom 解析失败后退回的）
    上限       : 1024 tok
    结果       : 7 张全部 RuntimeError: 输出为空

已经用裸 HTTP 排除掉的东西（见 diagnose_gguf_empty.py）：
    · chat template **认** enable_thinking，显式传 False 就不思考
    · 不传该 kwarg 时模板按 `is undefined` 走**思考**分支
    · 带图 + 初稿 + 显式 False 的组合本身是好的（4.2s / 123 tok / 正常正文）

所以嫌疑收窄成一句：**这一轮到底有没有把 enable_thinking=False 传下去**。
本脚本就用真实预设 + 真实 refine 指令 + 真实素材，把这个开关两种状态各跑一遍。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_gguf_refine.py
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

TMP = tempfile.mkdtemp(prefix="qwen35_refine_")

# ---- ComfyUI 桩（节点 import 与 _free_vram 需要）----
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


def snapshot():
    """把原始图 + 初稿收进内存，便于两轮之间复位。"""
    keep = {}
    for n in NAMES:
        for ext in (".jpg", ".png"):
            p = os.path.join(SRC, os.path.splitext(n)[0] + ext)
            if os.path.isfile(p):
                keep[n] = (p, open(p, "rb").read())
                break
        t = os.path.join(SRC, os.path.splitext(n)[0] + ".txt")
        if os.path.isfile(t):
            keep["txt:" + n] = (t, open(t, "rb").read())
    return keep


def reset(keep):
    for n in NAMES:
        base = os.path.splitext(n)[0]
        # 清掉上一轮可能留下的所有产物
        for suf in ("", ".orig", ".q35state"):
            for ext in (".jpg", ".png", ".txt", ".txt" + suf):
                p = os.path.join(TMP, base + ext)
                if os.path.isfile(p):
                    os.remove(p)
        img_p, blob = keep[n]
        shutil.copyfile(img_p, os.path.join(TMP, n))
        t_p, t_blob = keep["txt:" + n]
        open(os.path.join(TMP, base + ".txt"), "wb").write(t_blob)


def main():
    exe = gb.find_llama_server()
    mains, projs = gb.list_gguf()
    if not exe or not mains:
        print("!! 找不到 llama-server 或 gguf")
        return 2

    keep = snapshot()
    print(f"临时目录 : {TMP}")
    print(f"素材     : {', '.join(NAMES)}")
    for n in NAMES:
        t_p, blob = keep["txt:" + n]
        print(f"  初稿 {n}: {len(blob)} 字节")
    print()

    be = QM.Qwen35GGUFServer().provide(
        model=list(mains.keys())[0], mmproj=list(projs.keys())[0],
        server_exe="auto", context_size=8192, auto_download=False,
    )[0]

    node = QM.Qwen35BatchImageTagger()

    common = dict(
        model_name="", folder_path=TMP, system_preset="character",
        system_prompt="",
        backend=be,
        caption_mode="refine",          # ← 现场就是这个
        desc_length="preset",           # ← 现场 custom 解析失败后退回 preset
        bilingual="off",                # 失败发生在英文侧，先砍掉双语噪声
        max_new_tokens=1024,            # ← 现场
        temperature=0.2, seed=42, max_image_side=1536,
        overwrite="overwrite",          # 强制重做，不被指纹跳过
        show_progress=False,
        keep_model_loaded=True,         # 两轮共用一个 server
        unload_other_models=False,
    )

    # ---- 预热：第一次要把 16.19GB 权重拉起来 ----
    print("=" * 74)
    print("预热（启动 llama-server，不计入判定）")
    print("=" * 74)
    t0 = time.perf_counter()
    try:
        node.tag_folder(enable_thinking=False, **common)
    except BaseException as e:
        print(f"  预热异常（继续）：{type(e).__name__}: {e}")
    print(f"  预热完成 {time.perf_counter() - t0:.1f}s\n")

    for et in (False, True):
        reset(keep)
        print("=" * 74)
        print(f"enable_thinking = {et}")
        print("=" * 74)
        t0 = time.perf_counter()
        try:
            r = node.tag_folder(enable_thinking=et, **common)
            dt = time.perf_counter() - t0
            rep = r["result"][0]
            print(rep)
            print(f"  >>> 用时 {dt:.1f}s")
        except BaseException as e:
            dt = time.perf_counter() - t0
            print(f"  !! 抛异常 {type(e).__name__}: {e}  （{dt:.1f}s）")
        # 结果 txt
        for n in NAMES:
            p = os.path.join(TMP, os.path.splitext(n)[0] + ".txt")
            body = open(p, encoding="utf-8").read().strip() if os.path.isfile(p) else "(无)"
            print(f"  {n} -> {len(body)} 字: {body[:150]!r}")
        print()

    gb.release(force=True)
    print(f"临时目录保留在 {TMP}（可自行翻看）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
