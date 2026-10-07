# -*- coding: utf-8 -*-
"""用节点自己的下载函数把 mmproj 取回来。

和「GGUF 后端」节点缺件时走的是同一个 `gguf_backend.ensure_file()` ——
所以这个脚本跑通就等于节点的下载路径在真实网络上跑通了。
"""
import os
import sys
import time

NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
sys.path.insert(0, NODE_DIR)
import gguf_backend as gb  # noqa: E402

DEST_DIR = r"H:\AI\models\LLM"
URL = gb.DEFAULT_MMPROJ_URL

os.makedirs(DEST_DIR, exist_ok=True)
print("URL     :", URL)
print("目录    :", DEST_DIR)
print("远端大小:", f"{gb.remote_size(URL):,} 字节")
print("-" * 60, flush=True)

t0 = time.time()
last = [0.0]


def status(done, total, el):
    now = time.time()
    if now - last[0] < 3.0 and total > 0 and done < total:
        return
    last[0] = now
    sp = done / el / (1024 * 1024) if el > 0.5 else 0.0
    if total > 0:
        print(f"  {done * 100.0 / total:5.1f}%  {done / 1048576:7.1f} / "
              f"{total / 1048576:.1f} MB  {sp:5.1f} MB/s", flush=True)
    else:
        print(f"  {done / 1048576:7.1f} MB  {sp:5.1f} MB/s", flush=True)


path = gb.ensure_file(URL, DEST_DIR, on_status=status)
print("-" * 60)
print("落地    :", path)
print("实际大小:", f"{os.path.getsize(path):,} 字节")
print("耗时    :", f"{time.time() - t0:.1f}s")
print("残留    :", "有 .part（异常）" if os.path.exists(path + ".part") else "无 .part（正常）")
