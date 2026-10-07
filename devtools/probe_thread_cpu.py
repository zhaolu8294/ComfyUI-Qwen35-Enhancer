# -*- coding: utf-8 -*-
"""地面真值：解码期间，ComfyUI 进程里**每个线程**实际吃了多少 CPU。

为什么不信 py-spy：它在 Windows 上 `--native` 采样会报 "N behind in sampling"，
样本可能向某些栈倾斜。这里改用 Windows 自己维护的每线程 CPU 时间（psutil 可读），
没有任何采样偏差。

关键判别：
    监控线程（Thread-8 startMonitorLoop）吃 40~60% 的核
        -> GIL 争用成立（HF 解码是纯 Python 发射，最怕被抢 GIL）
    监控线程只吃 1~5%
        -> py-spy 那个结论是假的，得另找原因

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_thread_cpu.py --images 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import urllib.request

import psutil

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
ALL = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png",
       "Qwen_image_2.1_00027.png"]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"


def find_comfyui():
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        s = " ".join(str(a) for a in (p.info["cmdline"] or []))
        if "main.py" in s and "ComfyUI" in s:
            return psutil.Process(p.info["pid"])
    return None


def snap(proc):
    """{tid: cpu_seconds}  —— user+system"""
    out = {}
    for t in proc.threads():
        out[t.id] = (t.user_time or 0.0) + (t.system_time or 0.0)
    return out


# 线程名（Python 层）通过 py-spy 拿不到时就靠这一行：
# comfyui 里 Thread-8 是 startMonitorLoop（py-spy dump 已确认）
NAME_HINT = {
    "prompt_worker": "解码/执行线程",
    "startMonitorLoop": "Crystools 监控线程",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="qwen35_tcpu_")
    for n in ALL[:a.images]:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))

    proc = find_comfyui()
    if proc is None:
        print("!! 没找到 ComfyUI 进程")
        return
    print(f"ComfyUI PID {proc.pid}   逻辑核 {psutil.cpu_count(logical=True)}")

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
    print(f"已提交 pid={pid}   {a.images} 张图")
    print()

    base = snap(proc)
    prev = dict(base)
    t0 = time.perf_counter()
    last = t0
    print(f"{'t':>6}{'进程CPU%':>10}   每线程增量 CPU%（只列 >0.5% 的）")
    print("-" * 96)
    acc = {}          # tid -> 累计 CPU 秒
    done = False
    while time.perf_counter() - t0 < 260:
        time.sleep(a.interval)
        now = time.perf_counter()
        dt = now - last
        last = now
        cur = snap(proc)
        rows = []
        tot_cpu = 0.0
        for tid, c in cur.items():
            d = c - prev.get(tid, c)
            tot_cpu += d
            acc[tid] = acc.get(tid, 0.0) + d
            pct = d / dt * 100.0
            if pct > 0.5:
                rows.append((pct, tid))
        prev = cur
        rows.sort(reverse=True)
        s = "  ".join(f"tid{tid}:{pct:.0f}%" for pct, tid in rows[:6])
        print(f"{now-t0:>6.1f}{tot_cpu/dt*100:>10.1f}   {s}")

        try:
            with urllib.request.urlopen(f"{API}/history/{pid}", timeout=10) as r:
                h = json.loads(r.read())
            if pid in h and (h[pid].get("outputs")
                             or h[pid].get("status", {}).get("status_str") == "error"):
                done = True
                break
        except Exception:
            pass

    print()
    print("=" * 96)
    wall = time.perf_counter() - t0
    print(f"完成={done}  观察窗口 {wall:.1f}s")
    print(f"整段窗口内各线程累计 CPU（CPU 秒 / 占窗口比）：")
    for tid, c in sorted(acc.items(), key=lambda x: -x[1])[:8]:
        if c < 0.2:
            continue
        print(f"   tid {tid:<8} {c:8.2f} s   = 平均 {c/wall*100:5.1f}% 个核")
    print()
    print("注：psutil 在 Windows 上给不出线程名。逐个核对的办法：")
    print("    在任务管理器『详细信息 → 右键列 → 选择列 → 线程 ID』里对 tid。")


if __name__ == "__main__":
    main()
