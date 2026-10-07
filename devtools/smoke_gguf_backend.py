# -*- coding: utf-8 -*-
"""GGUF 后端真机冒烟测试。

目的只有一个：证明 27B Q4 在 4090D 24GB 上**真的整份进了显存**，
而不是被静默摊到 CPU。判据是 tok/s 的量级 ——
  · 全 GPU：decode 通常 25~40 tok/s
  · 有层掉到 CPU：掉到 0.6~3.4 tok/s（而且日志里一个错都不会报）

同时校验 gguf_backend 的真实代码路径：发现 exe、建 server、等 /health、
发 chat 请求、解析 timings。

用法：
    python smoke_gguf_backend.py            # 默认只跑纯文本
    python smoke_gguf_backend.py --mmproj   # 若有 mmproj，一并验证视觉
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
sys.path.insert(0, NODE_DIR)

import gguf_backend as gb  # noqa: E402


def vram():
    """返回 (已用MiB, 总量MiB, 进程数说明)。拿不到就 (0,0,'')。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()
        used, total = [int(x) for x in out[0].split(",")]
        return used, total, ""
    except Exception as e:
        return 0, 0, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", action="store_true", help="连视觉一起验")
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    exe = gb.find_llama_server()
    if not exe:
        print("!! 找不到 llama-server.exe"); return 2
    mains, projs = gb.list_gguf()
    if not mains:
        print("!! 找不到 gguf"); return 2
    model = list(mains.values())[0]

    mmproj = ""
    if args.mmproj:
        if not projs:
            print("!! 指定了 --mmproj 但扫不到 mmproj 文件"); return 2
        mmproj = list(projs.values())[0]

    print("=" * 66)
    print("server :", exe)
    print("model  :", model)
    print("mmproj :", mmproj or "(无，纯文本)")
    print("=" * 66)

    u0, tot, _ = vram()
    print(f"[显存] 启动前已用 {u0} MiB / {tot} MiB")

    cfg = {
        "server_exe": exe,
        "model": model,
        "mmproj": mmproj,
        "n_gpu_layers": -1,          # -1 = 不传 -ngl，让 llama.cpp 自己 fit
        "context_size": args.ctx,
        "parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attn": True,
        "batch": 2048,
        "ubatch": 512,
        "reasoning_format": "deepseek",
        "extra_args": "",
    }

    t0 = time.time()
    srv = gb.acquire(cfg)
    try:
        srv.ensure_ready(on_status=lambda s: print("  " + s, flush=True), timeout=900)
    except gb.GgufError as e:
        print("\n!! 启动失败：\n", e)
        return 3
    load_s = time.time() - t0
    u1, tot, _ = vram()
    print(f"[显存] 加载后已用 {u1} MiB / {tot} MiB（净增 {u1 - u0} MiB）")
    print(f"[耗时] 加载 {load_s:.1f}s")

    # ---- 纯文本 ----
    print("\n--- 纯文本推理 ---")
    try:
        r = srv.chat(
            system="You are a helpful assistant. Answer in one short sentence.",
            user="What is the capital of France?",
            max_tokens=64, temperature=0.3,
        )
    except gb.GgufError as e:
        print("!! 推理失败：\n", e)
        print("\nserver 日志尾部：\n", srv.log_tail(30))
        return 4
    ct = r["completion_tokens"]
    dec = r["decode_s"] or r["seconds"]
    tps = ct / dec if dec > 0 else 0.0
    print(f"  回复     : {r['text'][:200]!r}")
    if r["reasoning"]:
        print(f"  思考块   : {r['reasoning'][:120]!r}  (已被引到独立字段)")
    print(f"  prompt   : {r['prompt_tokens']} tok / {r['prefill_s']:.2f}s")
    print(f"  生成     : {ct} tok / {dec:.2f}s  =  {tps:.1f} tok/s")

    if tps and tps < 8:
        print(f"\n  [!!] {tps:.1f} tok/s 偏低 —— 大概率有层被摊到 CPU，"
              "看上面日志有没有 'offloaded ... to CPU' 字样")
    elif tps:
        print(f"\n  [OK] {tps:.1f} tok/s，量级正常，权重确实在显存里")

    # ---- 视觉 ----
    if args.mmproj:
        print("\n--- 视觉推理 ---")
        # 造一张纯色测试图，问它主色
        try:
            from PIL import Image
            import io
            im = Image.new("RGB", (512, 512), (230, 30, 30))
            buf = io.BytesIO(); im.save(buf, "PNG")
            blob = buf.getvalue()
        except Exception as e:
            print("  (PIL 不可用，跳过)", e); blob = None
        if blob:
            t1 = time.time()
            try:
                rv = srv.chat(
                    system="Describe the dominant color of the image in one word.",
                    user="What color is this image?",
                    images=[("image/png", blob)],
                    max_tokens=48, temperature=0.2,
                )
            except gb.GgufError as e:
                print("!! 视觉推理失败：\n", e); return 5
            dt = time.time() - t1
            print(f"  回复     : {rv['text'][:200]!r}")
            print(f"  prompt   : {rv['prompt_tokens']} tok / {rv['prefill_s']:.2f}s")
            print(f"  总耗时   : {dt:.1f}s   -> 单图打标就是这个量级")
            if "red" in rv["text"].lower() or "红" in rv["text"]:
                print("  [OK] 色觉正确，视觉塔工作正常")
            else:
                print("  [??] 没识别出红色，需要人工看一眼")

    u2, tot, _ = vram()
    print(f"\n[显存] 推理后 {u2} MiB / {tot} MiB")
    return 0


if __name__ == "__main__":
    code = main()
    print("\n-- 关闭 server --")
    gb.release(force=True)
    print("完成，退出码", code)
    sys.exit(code)
