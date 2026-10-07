# -*- coding: utf-8 -*-
"""在 ComfyUI 进程内跑一次同样的打标 —— 定位「脚本 20.5 tok/s vs ComfyUI 8.5 tok/s」。

已确证的事实：
    · 独立进程跑同一个节点、同一批图、同一配置：**20.5 tok/s**（native 分配器）
      19.4 tok/s（cudaMallocAsync）→ 分配器只值 5%，已排除
    · ComfyUI 内 22:15 那次：**8.5 tok/s**（同样 bf16 / character / extra_long / raw / 784）
    · 凌晨 04:32 / 04:56 的 ComfyUI 内是 **23.1 / 20.9 tok/s**（也没装 fla）
    · 所以：不是模型、不是代码、不是分配器 —— 是 ComfyUI 这个进程**现在**的状态

做法：走 ComfyUI 的 HTTP 接口（/prompt）提交同一个打标任务。这样代码路径、
进程、CUDA 上下文全都是「真身」，不是模拟。素材复制到临时目录，不碰现场文件。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_comfyui_run.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.request

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
NAMES = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png"]
API = "http://127.0.0.1:8188"
# model_name 的取值必须与 COMBO 选项逐字一致（含后缀）
MODELS = [
    "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]",
    "Qwen3-VL-8B-Instruct  [Qwen3VLForConditionalGeneration]",
]


def post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read())


def run_one(model, tmp, label):
    print()
    print("=" * 84)
    print(f"{label}   {model}")
    print("=" * 84)
    st = get("/system_stats")
    d = st["devices"][0]
    print(f"跑前  显存 {d['vram_free']/1024**3:.2f} GiB 可用 / "
          f"内存 {st['system']['ram_free']/1024**3:.2f} GiB 可用")

    inputs = {
        "model_name": model,
        "folder_path": tmp,
        "system_preset": "character",
        "system_prompt": "",
        "user_prompt": "给这张图打标。",
        "quantization": "none",
        "attention": "auto",
        "enable_thinking": False,
        "recursive": False,
        "overwrite": "overwrite",
        "image_exts": ".png,.jpg,.jpeg,.webp,.bmp",
        "output_suffix": "",
        "output_encoding": "utf-8",
        "output_format": "raw",
        "max_new_tokens": 784,
        "temperature": 0.2,
        "seed": 42,
        "max_image_side": 1536,
        "limit": 0,
        "dry_run": False,
        "keep_model_loaded": False,
        "unload_other_models": True,
        "custom_model_path": "",
        "show_progress": True,
        "progress_interval": 2.0,
        "bilingual": "off",
        "max_output_chars": 0,
        "caption_mode": "off",
        "desc_length": "extra_long",
        "desc_words": "",
        "bilingual_sync": "auto",
        "verify": "off",
    }
    t0 = time.perf_counter()
    try:
        res = post("/prompt", {"prompt": {
            "1": {"class_type": "Qwen35BatchImageTagger", "inputs": inputs}}})
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        print(f"!! 提交失败 {type(e).__name__}: {e}\n{body[:800]}")
        return
    pid = res.get("prompt_id")
    print(f"已提交 prompt_id = {pid}")

    # 轮询完成
    rep = None
    for i in range(200):                      # 最多 ~10 分钟
        time.sleep(3)
        try:
            h = get(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            entry = h[pid]
            if entry.get("outputs"):
                rep = entry
                break
            if entry.get("status", {}).get("status_str") == "error":
                rep = entry
                break
    dt = time.perf_counter() - t0
    print(f"\n墙钟总耗时（含排队/加载/卸载）: {dt:.1f}s")

    if not rep:
        print("!! 没等到结果（超时或仍在跑）")
        return
    stt = rep.get("status", {})
    print(f"状态: {stt.get('status_str')}  completed={stt.get('completed')}")
    for msg in stt.get("messages", []) or []:
        if msg and msg[0] in ("execution_error",):
            print("  !! 执行错误:", json.dumps(msg[1], ensure_ascii=False)[:600])
    outs = rep.get("outputs", {})
    for nid, o in outs.items():
        for t in (o.get("text") or []):
            print("  --- 节点返回的报告 ---")
            for ln in str(t).splitlines():
                if ln.strip():
                    print("  | " + ln)
    print()


def main():
    tmp = tempfile.mkdtemp(prefix="qwen35_api_")
    for n in NAMES:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))
    print(f"临时目录: {tmp}")
    print("配置: character / extra_long / raw / 上限 784 / 1536 / 思考关 / 自检关")
    st = get("/system_stats")
    print(f"ComfyUI argv = {st['system']['argv']}")
    for i, m in enumerate(MODELS, 1):
        run_one(m, tmp, f"[{i}/{len(MODELS)}] ComfyUI 进程内")
    print()
    print("基线：独立进程同配置 20.5 tok/s（native）/ 19.4（cudaMallocAsync）")
    print("      凌晨 ComfyUI 内同模型 20.9~23.1 tok/s")


if __name__ == "__main__":
    main()
