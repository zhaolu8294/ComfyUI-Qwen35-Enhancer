# -*- coding: utf-8 -*-
"""本机 P-core 与 E-core 在「Python 派发型」负载上的真实性能比。

为什么需要这个数：HF 解码在本机是**单核 + 逐个小算子派发**受限
（profile 实测 3365 个 CUDA kernel / 输出 token）。如果同一个核
在 P-core 上跑 21 tok/s、在 E-core 上跑 8 tok/s，那 2.56 倍就必须由
「P/E 单核性能比」来兑现 —— 这个比值必须亲自量，不能凭印象。

三个微基准（都尽量贴近真实负载，不测纯算术峰值）：
  1. py_int_loop   ：Python 解释器紧循环（对象分配 + 整数运算 + 列表索引）
                      —— 贴近 transformers 的 Python 层开销
  2. small_linear  ：tiny tensor 上反复调 F.linear（CPU）
                      —— 贴近「每个小算子一次派发」的成本
  3. alloc_loop    ：反复创建/销毁小对象和 tensor —— 贴近每 token 的临时张量

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe bench_core_type.py
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe bench_core_type.py --seconds 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import psutil

P_CORES = list(range(16))        # i7-14700KF：逻辑 0-15（8 个 P-core，HT）
E_CORES = list(range(16, 28))    # 逻辑 16-27（12 个 E-core）

ARMS = [
    ("不限制（全部 28 逻辑核）", None),
    ("P-core ×16（逻辑 0-15）", P_CORES),
    ("E-core ×12（逻辑 16-27）", E_CORES),
    ("单 P-core（逻辑 2）", [2]),
    ("单 E-core（逻辑 20）", [20]),
]


# ----------------------------------------------------------------- 微基准
def bm_int_loop(deadline):
    n = 0
    x = 0
    d = {"a": 0}
    while time.perf_counter() < deadline:
        for _ in range(20000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            d["a"] = d.get("a", 0) + (x & 7)
            n += 1
    return n


def bm_small_linear(deadline, torch):
    import torch.nn.functional as F
    x = torch.randn(1, 256)
    w = torch.randn(256, 256)
    n = 0
    while time.perf_counter() < deadline:
        for _ in range(200):
            y = F.linear(x, w)
            n += 1
    return n


def bm_alloc_loop(deadline, torch):
    n = 0
    while time.perf_counter() < deadline:
        for _ in range(2000):
            t = torch.empty(64)
            l = [float(i) for i in range(8)]
            n += 1
    return n


# -----------------------------------------------------------------
def run_arm(label, mask, seconds, torch):
    p = psutil.Process()
    saved = p.cpu_affinity()
    if mask is not None:
        try:
            p.cpu_affinity(mask)
        except Exception as e:
            print(f"    !! 设置亲和性失败: {e}")
            return None
    time.sleep(0.2)
    res = {}
    for name, fn in (("py_int_loop", lambda dl: bm_int_loop(dl)),
                     ("small_linear", lambda dl: bm_small_linear(dl, torch)),
                     ("alloc_loop", lambda dl: bm_alloc_loop(dl, torch))):
        try:
            fn(time.perf_counter() + 0.4)          # 预热
            r = []
            for _ in range(3):
                t = time.perf_counter()
                n = fn(t + seconds)
                r.append(n / (time.perf_counter() - t))
            res[name] = statistics.median(r)
        except Exception as e:
            res[name] = float("nan")
            print(f"    !! {name} 失败: {e}")
    try:
        p.cpu_affinity(saved)
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)          # 单核可比：HF 解码就是单核受限
    print(f"CPU: {psutil.cpu_count(logical=True)} 逻辑 / "
          f"{psutil.cpu_count(logical=False)} 物理；torch.set_num_threads(1)")
    print(f"每个微基准跑 3 次取中位数，每次 {a.seconds}s")
    print()

    out = {}
    for label, mask in ARMS:
        got = [c for c in (mask or psutil.Process().cpu_affinity())]
        print(f"  {label}   （实际可用核 {len(got)} 个："
              f"{got[:6]}{'...' if len(got) > 6 else ''}）")
        out[label] = run_arm(label, mask, a.seconds, torch)

    print()
    hdr = f"  {'亲和性':<30}{'py_int_loop':>14}{'small_linear':>14}{'alloc_loop':>13}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    base = out.get(ARMS[0][0]) or {}
    for label, _ in ARMS:
        r = out.get(label)
        if not r:
            continue
        print(f"  {label:<30}{r['py_int_loop']:>14,.0f}"
              f"{r['small_linear']:>14,.0f}{r['alloc_loop']:>13,.0f}")

    p_single = out.get("单 P-core（逻辑 2）") or {}
    e_single = out.get("单 E-core（逻辑 20）") or {}
    if p_single and e_single:
        print()
        print("  单核 P / E 性能比（>1 表示 P 更快；这就是 HF 解码 2.56 倍差距能取到的上限）：")
        for k in ("py_int_loop", "small_linear", "alloc_loop"):
            try:
                print(f"    {k:<16} P/E = {p_single[k]/e_single[k]:.2f}x"
                      f"   （P {p_single[k]:,.0f}  vs  E {e_single[k]:,.0f}）")
            except Exception:
                pass
        print()
        print("  判读：若 P/E ≈ 2.5  -> 单靠「线程落在哪个核」就能解释 8 vs 21 tok/s")
        print("        若 P/E ≈ 1.3~1.8 -> 还有第二个因素，需继续查")


if __name__ == "__main__":
    main()
