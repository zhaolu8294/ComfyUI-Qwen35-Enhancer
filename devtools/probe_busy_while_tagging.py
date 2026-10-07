# -*- coding: utf-8 -*-
"""ComfyUI 里 HF 打标恒为 8 tok/s —— 看打标期间 GPU 与 CPU 谁在忙。

已确证：
    · ComfyUI 内 Qwen3.5-9B（24 层线性注意力）= 8.0~8.4 tok/s
    · ComfyUI 内 Qwen3-VL-8B（纯全注意力）  = 7.9~8.6 tok/s
      **架构完全不同，速度分毫不差** → 不是"计算慢"，是每 token 有固定等待
    · 独立进程同模型同配置 = 20.5 tok/s
    · GGUF 后端（llama-server 独立进程）在 ComfyUI 内 = 33.9~37.1 tok/s，不受影响

判据：
    GPU 低 + ComfyUI CPU 高   -> CPU 抢占（HF 路径是 CPU 发射密集，最怕这个）
    GPU 低 + ComfyUI CPU 也低 -> 卡在同步/等待（有东西每 token 同步一次）
    GPU 高                    -> 计算饱和（那就是 kernel 效率问题）

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_busy_while_tagging.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request

import psutil

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
NAMES = ["CellWorks39.jpg", "CellWorks42.jpg"]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"


def gpu_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        u, m = out.split(",")
        return float(u), float(m)
    except Exception:
        return -1.0, -1.0


def main():
    tmp = tempfile.mkdtemp(prefix="qwen35_busy_")
    for n in NAMES:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))

    # 找 ComfyUI 进程
    comfy = None
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        cl = p.info["cmdline"] or []
        if any("ComfyUI" in str(a) and "main.py" in str(a) for a in cl):
            comfy = psutil.Process(p.info["pid"])
            break
    print(f"ComfyUI 进程: {comfy.pid if comfy else '未找到'}")

    inputs = {
        "model_name": MODEL, "folder_path": tmp,
        "system_preset": "character", "system_prompt": "",
        "user_prompt": "给这张图打标。", "quantization": "none", "attention": "auto",
        "enable_thinking": False, "recursive": False, "overwrite": "overwrite",
        "image_exts": ".png,.jpg,.jpeg,.webp,.bmp", "output_suffix": "",
        "output_encoding": "utf-8", "output_format": "raw",
        "max_new_tokens": 784, "temperature": 0.2, "seed": 42,
        "max_image_side": 1536, "limit": 0, "dry_run": False,
        "keep_model_loaded": False, "unload_other_models": True,
        "custom_model_path": "", "show_progress": True, "progress_interval": 2.0,
        "bilingual": "off", "max_output_chars": 0, "caption_mode": "off",
        "desc_length": "extra_long", "desc_words": "", "bilingual_sync": "auto",
        "verify": "off",
    }
    req = urllib.request.Request(
        API + "/prompt",
        data=json.dumps({"prompt": {"1": {
            "class_type": "Qwen35BatchImageTagger", "inputs": inputs}}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    print(f"已提交 prompt_id = {pid}（2 张图，约 70s）")

    # 采样
    print()
    print(f"  {'时刻':<8}{'GPU%':>7}{'显存MB':>9}{'ComfyUI CPU%':>14}{'线程':>6}"
          f"{'最多忙的线程':>28}")
    if comfy:
        comfy.cpu_percent(None)
    peak = {"gpu": 0.0, "cpu": 0.0}
    t0 = time.perf_counter()
    done = False
    while time.perf_counter() - t0 < 150:
        time.sleep(1.5)
        u, m = gpu_util()
        cpu = 0.0
        nthr = 0
        top = "-"
        if comfy:
            try:
                cpu = comfy.cpu_percent(None)
                th = comfy.threads()
                nthr = len(th)
                # 单次快照看不出增量，用瞬时值代替（两次采样间 psutil 内部会算）
                top = f"{len(th)} 线程快照"
            except Exception:
                pass
        peak["gpu"] = max(peak["gpu"], u)
        peak["cpu"] = max(peak["cpu"], cpu)
        el = time.perf_counter() - t0
        print(f"  {el:>5.1f}s{u:>7.0f}{m:>9.0f}{cpu:>14.1f}{nthr:>6}{top:>28}")
        try:
            h = json.loads(urllib.request.urlopen(
                f"{API}/history/{pid}", timeout=10).read())
            if pid in h and h[pid].get("outputs"):
                done = True
                break
        except Exception:
            pass

    print()
    print(f"  峰值：GPU {peak['gpu']:.0f}%  /  ComfyUI 进程 CPU {peak['cpu']:.0f}%")
    print("  判据：GPU 低 + CPU 高 → CPU 抢占；两者都低 → 每 token 有同步等待；"
          "GPU 高 → 计算饱和")
    print(f"  完成={done}")


if __name__ == "__main__":
    main()
