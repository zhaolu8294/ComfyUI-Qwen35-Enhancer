# -*- coding: utf-8 -*-
"""裸算子微基准：定位 Conv3d 慢在哪一档。

背景：Qwen3-VL 视觉塔的 patch_embed 是 Conv3d(3->1152, kernel/stride=(2,16,16))，
输入 (3360, 3, 2, 16, 16) 输出 (3360, 1152, 1, 1, 1) —— 数学上就是个
(3360x1536)@(1536x1152) 的 GEMM，理论 ~12 GFLOP，4090D 上应该 10ms 量级。
实测视觉塔 29s，疑似 bucket 全在这个卷积上。

每步都 flush=True，被超时掐断也能看到卡在哪一档。
"""
import sys
import time

import torch

print("torch", torch.__version__,
      "| cudnn", torch.backends.cudnn.version(),
      "| cudnn.enabled", torch.backends.cudnn.enabled,
      "| benchmark", torch.backends.cudnn.benchmark,
      "| tf32", torch.backends.cudnn.allow_tf32, flush=True)

N, C, T, P = 3360, 3, 2, 16


def bench(fn, warmup=2, rep=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / rep


for name, dt in (("float32", torch.float32),
                 ("float16", torch.float16),
                 ("bfloat16", torch.bfloat16)):
    print(f"\n--- Conv3d {name} ---", flush=True)
    try:
        x = torch.randn(N, C, T, P, P, device="cuda", dtype=dt)
        conv = torch.nn.Conv3d(C, 1152, (T, P, P), (T, P, P), bias=True).cuda().to(dt)
        conv.eval()
        print(f"  构造 OK，权重 dtype {conv.weight.dtype}", flush=True)
        with torch.inference_mode():
            t1 = time.perf_counter()
            y = conv(x)
            torch.cuda.synchronize()
            print(f"  首次调用 {time.perf_counter() - t1:.2f}s  输出 {tuple(y.shape)}",
                  flush=True)
            s = bench(lambda: conv(x))
        print(f"  稳态 {s * 1000:.1f} ms/次", flush=True)
        del x, conv, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败 {type(e).__name__}: {str(e)[:120]}", flush=True)

for name, dt in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
    print(f"\n--- 等价 Linear {name}（同一计算量，但不是卷积）---", flush=True)
    try:
        xf = torch.randn(N, C * T * P * P, device="cuda", dtype=dt)
        lin = torch.nn.Linear(C * T * P * P, 1152, bias=True).cuda().to(dt)
        with torch.inference_mode():
            s = bench(lambda: lin(xf))
        print(f"  稳态 {s * 1000:.1f} ms/次", flush=True)
        del xf, lin
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  失败 {type(e).__name__}: {str(e)[:120]}", flush=True)

print("\n完成", flush=True)
