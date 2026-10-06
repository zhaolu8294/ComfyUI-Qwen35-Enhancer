# -*- coding: utf-8 -*-
"""
bitsandbytes 量化档位 —— 裸算子微基准（Qwen3.5-9B 的真实层形状）

背景
----
日志实测：同一模型同参数，只改 quantization
    none(bf16) 解码 18.5~21.1 tok/s / 整轮 51.85 s
    8bit       解码  2.3~6.4 tok/s / 整轮 119~319 s
且 8bit 自己不稳定（6.1 / 2.3 / 2.6 三种结果）。

本机 GPU 被 H3 占着 22.5/24.5 GiB，装不下 9B，所以改用**单层 Linear** 做
受控微基准 —— 这正是当初找出 Conv3d 病态路径的方法。

读 bnb 0.49.2 源码（autograd/_functions.py:MatMul8bitLt.forward）得到两个开关：
    if state.threshold > 0.0:
        output, subA = int8_mixed_scaled_mm(...)   # 离群列分解（transformers 默认 6.0）
    else:
        output = int8_scaled_mm(...)              # 纯融合 int8 内核
所以 threshold=0.0 是一条**源码里明确写着**的路径切换，不是我的猜测。

测量项
------
每个 (形状, batch) 组合跑 6 个变体：
    bf16            基线
    int8 t=6.0      当前默认（混合内核）
    int8 t=0.0      纯 int8 融合内核
    int8 t=0.0 fp16 同上但输入已是 fp16（省掉逐层 bf16->fp16 cast）
    int8 t=1e9      全部当离群（反证：证明混合路径有多贵）
    4bit nf4        bnb 的另一条路
另附：真实分布下阈值 6.0 会挑出多少离群列（解释"不稳定"的来源）。

运行：python -u bench_bnb_quant.py > bench_bnb_quant.out 2>&1
"""
import os
import sys
import time
import statistics

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

torch.manual_seed(0)
DEV = "cuda"
WARMUP = 3
REPS = 10

# Qwen3.5-9B 的真实形状：hidden 4096 / intermediate 12288 / 32 层
SHAPES = [
    ("mlp_up  4096->12288", 4096, 12288),
    ("mlp_down 12288->4096", 12288, 4096),
    ("linattn 4096->4096 ", 4096, 4096),
]
BATCHES = [(1, "decode"), (256, "prefill")]


def make_layer(kind, in_f, out_f, dtype):
    if kind == "bf16":
        return torch.nn.Linear(in_f, out_f, bias=False).to(DEV).to(dtype).eval()
    if kind.startswith("int8"):
        thr = float(kind.split("t=")[1].split()[0])
        layer = bnb.nn.Linear8bitLt(in_f, out_f, bias=False,
                                    has_fp16_weights=False, threshold=thr)
        return layer.to(DEV).eval()
    if kind == "4bit":
        layer = bnb.nn.Linear4bit(in_f, out_f, bias=False,
                                  quant_type="nf4", compute_dtype=torch.bfloat16)
        return layer.to(DEV).eval()
    raise ValueError(kind)


CASES = [
    ("bf16            ", "bf16", torch.bfloat16),
    ("int8 t=6.0      ", "int8 t=6.0", torch.bfloat16),
    ("int8 t=0.0      ", "int8 t=0.0", torch.bfloat16),
    ("int8 t=0.0 fp16 ", "int8 t=0.0 fp16", torch.float16),
    ("int8 t=1e9      ", "int8 t=1e9", torch.bfloat16),
    ("4bit nf4        ", "4bit", torch.bfloat16),
]


def timeit(layer, x):
    with torch.no_grad():
        for _ in range(WARMUP):
            layer(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            layer(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))          # ms
    return statistics.median(ts), min(ts)


def weight_bytes(kind, in_f, out_f):
    """按实际存储估算权重字节数，用来算等效带宽。"""
    n = in_f * out_f
    if kind == "bf16":
        return n * 2
    if kind == "4bit":
        return n * 0.5
    return n * 1.0                                # int8（未把 fp16 离群算进去）


def main():
    print("=" * 96)
    print("bitsandbytes 量化档位微基准")
    print("=" * 96)
    print("torch %s  cuda %s  bnb %s  gpu %s"
          % (torch.__version__, torch.version.cuda, bnb.__version__,
             torch.cuda.get_device_name(0)))
    free, total = torch.cuda.mem_get_info()
    print("显存：可用 %.2f GiB / 共 %.2f GiB（注意：H3 常驻，可用小）"
          % (free / 1024**3, total / 1024**3))
    print()

    # ---------- 先解释"不稳定"：阈值 6.0 会挑出多少离群列 ----------
    print("-" * 96)
    print("【离群列检测】threshold=6.0（transformers 默认）在真实量级的激活上挑出多少列")
    print("-" * 96)
    for name, in_f, out_f in SHAPES:
        for bs, tag in BATCHES:
            a = (torch.randn(bs, in_f, device=DEV, dtype=torch.bfloat16) * 3.0)
            rows = []
            for scale in (1.0, 3.0, 6.0):
                x = (a.to(torch.float16) * scale)
                _, _, idx = F.int8_vectorwise_quant(x, 6.0)
                n_out = 0 if idx is None else int(idx.numel())
                rows.append((scale, n_out, n_out / in_f * 100))
            print("  %-22s %-8s " % (name, tag)
                  + " | ".join("scale=%.0f σ: %4d 列 (%.1f%%)" % r for r in rows))
    print()
    print("  ↳ 离群列数**随输入分布变化** → 混合内核的工作量也随之变化，")
    print("    这就是 8bit 三次跑出 6.1/2.3/2.6 tok/s 的形态来源。")
    print()

    # ---------- 主表 ----------
    results = {}
    for name, in_f, out_f in SHAPES:
        for bs, tag in BATCHES:
            print("-" * 96)
            print("%s   batch=%d (%s)" % (name, bs, tag))
            print("-" * 96)
            print("  %-18s %10s %10s %12s %10s" % ("变体", "中位 ms", "最快 ms", "等效带宽", "相对 bf16"))
            base = None
            for label, kind, dtype in CASES:
                real_kind = kind.split()[0] + (" t=" + kind.split("t=")[1] if "t=" in kind else "")
                try:
                    layer = make_layer(kind, in_f, out_f, dtype)
                    x = torch.randn(bs, in_f, device=DEV, dtype=dtype)
                    med, best = timeit(layer, x)
                    wb = weight_bytes(real_kind.split()[0] if "t=" not in real_kind else "int8",
                                      in_f, out_f)
                    bw = wb / (med / 1000) / 1024**3          # GB/s
                    if base is None:
                        base = med
                    print("  %-18s %10.3f %10.3f %9.0f GB/s %9.2fx"
                          % (label, med, best, bw, med / base))
                    results[(name, bs, label.strip())] = med
                    del layer, x
                    torch.cuda.empty_cache()
                except Exception as ex:
                    print("  %-18s 失败: %s: %s" % (label, type(ex).__name__, str(ex)[:70]))
            print()

    # ---------- 汇总 ----------
    print("=" * 96)
    print("汇总：相对 bf16 的倍数（>1 表示比 bf16 慢）")
    print("=" * 96)
    hdr = "  %-18s" % "变体"
    for name, bs, _ in [(n, b, t) for n, _, _ in SHAPES for b, t in BATCHES]:
        hdr += "%14s" % ("%s/b%d" % (name.split()[0], bs))
    print(hdr)
    for label, _, _ in CASES:
        line = "  %-18s" % label
        for name, in_f, out_f in SHAPES:
            for bs, tag in BATCHES:
                b = results.get((name, bs, "bf16            ".strip()))
                v = results.get((name, bs, label.strip()))
                line += "%14s" % ("%.2fx" % (v / b) if (v and b) else "-")
        print(line)
    print()
    print("跑完：%s" % time.strftime("%F %T"))


if __name__ == "__main__":
    main()
