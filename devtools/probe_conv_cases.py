# -*- coding: utf-8 -*-
"""逐档探测 Conv3d 为什么在 fp16/bf16 下病态地慢。

单进程模式（argv[1] = 档位名）：跑一档，打印 JSON 结果。
驱动模式（argv[1] 省略或 = 'all'）：每档**单独起一个子进程**并设硬超时，
所以某一档卡死不会拖垮整轮 —— 这很关键，因为 fp16 Conv3d 的首次调用
在实测中 45 秒都没返回。

档位：
  fp32_cudnn / fp16_cudnn / bf16_cudnn     —— 三种 dtype，cuDNN 开启
  fp16_nocudnn / bf16_nocudnn              —— cuDNN 关闭（走 PyTorch 原生实现）
  bf16_benchmark                           —— cuDNN benchmark 打开（让 cuDNN 搜算法）
  bf16_matmul                              —— 等价 GEMM（F.linear），验证数学等价
  bf16_conv2d                              —— 2D 版对照，看是不是 3D 特有
"""
import json
import os
import subprocess
import sys
import time

N, C, T, P = 3360, 3, 2, 16
OUT_DIM = 1152
BUDGET = 60          # 单档子进程硬超时（秒）

CASES = ["fp32_cudnn", "fp16_cudnn", "bf16_cudnn", "fp16_nocudnn",
         "bf16_nocudnn", "bf16_benchmark", "bf16_matmul", "bf16_conv2d"]


def run_case(name):
    import torch

    if name.endswith("nocudnn"):
        torch.backends.cudnn.enabled = False
    if name == "bf16_benchmark":
        torch.backends.cudnn.benchmark = True

    dt = {"fp32": torch.float32, "fp16": torch.float16,
          "bf16": torch.bfloat16}[name.split("_")[0]]
    dev = "cuda"

    if name.endswith("matmul"):
        # Conv3d(kernel=stride=(T,P,P)) 在 1x1x1 输出上 == (N, C*T*P*P) @ (C*T*P*P, 1152)
        x = torch.randn(N, C * T * P * P, device=dev, dtype=dt)
        w = torch.randn(OUT_DIM, C * T * P * P, device=dev, dtype=dt)
        b = torch.randn(OUT_DIM, device=dev, dtype=dt)
        fn = lambda: torch.nn.functional.linear(x, w, b)      # noqa: E731
    elif name.endswith("conv2d"):
        x = torch.randn(N, C, T * P, P, device=dev, dtype=dt)
        conv = torch.nn.Conv2d(C, OUT_DIM, (T * P, P), (T * P, P)).to(dev).to(dt).eval()
        fn = lambda: conv(x)                                   # noqa: E731
    else:
        x = torch.randn(N, C, T, P, P, device=dev, dtype=dt)
        conv = torch.nn.Conv3d(C, OUT_DIM, (T, P, P), (T, P, P)).to(dev).to(dt).eval()
        fn = lambda: conv(x)                                   # noqa: E731

    with torch.inference_mode():
        t = time.perf_counter()
        y = fn()
        torch.cuda.synchronize()
        first = time.perf_counter() - t
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        steady = (time.perf_counter() - t) / 3
    return {"first_s": round(first, 3), "steady_ms": round(steady * 1000, 1),
            "out": list(y.shape), "dtype": str(y.dtype)}


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        try:
            print(json.dumps({"case": sys.argv[1], "ok": True,
                              **run_case(sys.argv[1])}))
        except Exception as e:
            print(json.dumps({"case": sys.argv[1], "ok": False,
                              "err": f"{type(e).__name__}: {str(e)[:150]}"}))
        return

    here = os.path.abspath(__file__)
    print(f"{'档位':<18} {'首次调用':>10} {'稳态':>12}   结果")
    print("-" * 78)
    for case in CASES:
        try:
            r = subprocess.run([sys.executable, "-u", here, case],
                               capture_output=True, text=True, timeout=BUDGET)
            line = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
            if not line:
                print(f"{case:<18} {'-':>10} {'-':>12}   无输出 rc={r.returncode} "
                      f"{r.stderr.strip()[-70:]}")
                continue
            d = json.loads(line[-1])
            if d.get("ok"):
                print(f"{case:<18} {d['first_s']:>9.3f}s {d['steady_ms']:>10.1f}ms   "
                      f"{d['out']} {d['dtype']}")
            else:
                print(f"{case:<18} {'-':>10} {'-':>12}   失败 {d['err']}")
        except subprocess.TimeoutExpired:
            print(f"{case:<18} {'>%ds' % BUDGET:>10} {'-':>12}   ⚠ 卡死，未在 {BUDGET}s 内返回")
        except Exception as e:
            print(f"{case:<18} {'-':>10} {'-':>12}   驱动异常 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
