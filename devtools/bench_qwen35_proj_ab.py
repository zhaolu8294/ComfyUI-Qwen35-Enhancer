# -*- coding: utf-8 -*-
"""
Qwen3.5-9B 投影成本 —— 严格 A/B 交替法（v2）

为什么要 v2
-----------
v1 单独顺序测各变体，结果不可重复：
    MLP 的 4bit 一次 5.73ms、一次 12.70ms；全注意力 int8 t=0.0 一次 4.07、一次 12.69。
而同期 bf16 两次稳定（MLP 784/794 GB/s，线性注意力 698/733 GB/s）。
→ GPU 本身没问题，**抖动来自 bnb 路径自身**（以及桌面合成/ComfyUI 常驻的背景负载）。
所以单独报 bnb 的绝对耗时没有意义，必须做受控对照。

v2 的做法
--------
1. 同一个形状集合下，**同时持有** bf16 与候选变体（内存够，见下）；
2. 每个 pass 内背靠背交替计时 bf16 / 变体，反复 ROUNDS 轮；
3. 各取**最小值**（最小值最能反映"没有被打扰时的真实速度"），只报比值。
这样两者共享同一段 GPU 负载状态，漂移被抵消。

形状按 Qwen3.5-9B 的真实 config（hidden 4096 / intermediate 12288 / 32 层），
输入按 post-RMSNorm 量级（单位 RMS）构造。
"""
import os
import gc
import time
import statistics

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

torch.manual_seed(0)
REPS = 8          # 每个样本内连测次数
ROUNDS = 5        # 交替轮数

H = 4096
GROUPS = [
    ("线性注意力投影×24", [
        ("in_proj_qkv", H, 8192, 24),
        ("in_proj_z", H, H, 24),
        ("in_proj_b", H, 32, 24),
        ("in_proj_a", H, 32, 24),
        ("out_proj", H, H, 24),
    ]),
    ("全注意力投影×8", [
        ("q", H, H, 8), ("k", H, 1024, 8), ("v", H, 1024, 8), ("o", H, H, 8),
    ]),
    ("MLP×32", [
        ("gate", H, 12288, 32), ("up", H, 12288, 32), ("down", 12288, H, 32),
    ]),
]

VARIANTS = [
    ("int8 t=6.0", "int8", 6.0),
    ("int8 t=0.0", "int8", 0.0),
    ("4bit nf4", "4bit", 0.0),
]


def make(kind, in_f, out_f, thr):
    if kind == "bf16":
        lin = torch.nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.mul_(0.02)
        return lin.eval()
    if kind == "int8":
        m = bnb.nn.Linear8bitLt(in_f, out_f, bias=False,
                                has_fp16_weights=False, threshold=thr)
        m.weight = bnb.nn.Int8Params(
            torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02,
            requires_grad=False, has_fp16_weights=False)
        return m.to("cuda").eval()
    if kind == "4bit":
        m = bnb.nn.Linear4bit(in_f, out_f, bias=False, quant_type="nf4",
                              compute_dtype=torch.bfloat16)
        m.weight = bnb.nn.Params4bit(
            torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02,
            requires_grad=False, quant_type="nf4")
        return m.to("cuda").eval()
    raise ValueError(kind)


def build(spec, kind, thr):
    mods = []
    for _n, in_f, out_f, cnt in spec:
        for _ in range(cnt):
            mods.append(make(kind, in_f, out_f, thr))
    return mods


def params(spec):
    return sum(in_f * out_f * cnt for _n, in_f, out_f, cnt in spec)


def sweep(mods, xs, spec):
    i = 0
    out = None
    for _n, in_f, _o, cnt in spec:
        x = xs[in_f]
        for _ in range(cnt):
            out = mods[i](x)
            i += 1
    return out


def once(mods, xs, spec):
    """连测 REPS 次，返回最小值（ms）。"""
    with torch.no_grad():
        ts = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = sweep(mods, xs, spec)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return min(ts), bool(torch.isfinite(out).all())


def ab_group(title, spec):
    print("=" * 96)
    print("%s   （%d 次 Linear 调用，%.2f B 参数）"
          % (title, sum(c for *_r, c in spec), params(spec) / 1e9))
    print("=" * 96)
    base = build(spec, "bf16", 0.0)
    good = True
    rows = []
    for label, kind, thr in VARIANTS:
        try:
            vm = build(spec, kind, thr)
        except Exception as ex:
            print("  %-12s 构建失败: %s: %s" % (label, type(ex).__name__, str(ex)[:70]))
            continue
        print("  %-12s" % label)
        for batch in (1, 256):
            xs = {}
            for _n, in_f, _o, _c in spec:
                xs.setdefault(in_f, torch.randn(batch, in_f, device="cuda",
                                                dtype=torch.bfloat16))
            with torch.no_grad():                        # 预热
                sweep(base, xs, spec)
                sweep(vm, xs, spec)
            torch.cuda.synchronize()
            bs, vs, oks = [], [], []
            for _ in range(ROUNDS):
                b, ok1 = once(base, xs, spec)
                v, ok2 = once(vm, xs, spec)
                bs.append(b); vs.append(v); oks.append(ok1 and ok2)
            bmin, vmin = min(bs), min(vs)
            wb = params(spec) * (0.5 if kind == "4bit" else 1.0)
            print("    batch=%-4d bf16 %8.3f ms (%4.0f GB/s) | %s %8.3f ms (%4.0f GB/s)"
                  "  ->  %5.2fx  %s"
                  % (batch, bmin, params(spec) * 2 / (bmin / 1000) / 1024**3,
                     label, vmin, wb / (vmin / 1000) / 1024**3,
                     vmin / bmin, "有限值OK" if all(oks) else "**出现非有限值**"))
            rows.append((title, batch, label, vmin / bmin))
            del xs
            torch.cuda.empty_cache()
        del vm
        gc.collect()
        torch.cuda.empty_cache()
    del base
    gc.collect()
    torch.cuda.empty_cache()
    print()
    return rows


def main():
    print("=" * 96)
    print("Qwen3.5-9B 投影成本 严格 A/B 交替法  REPS=%d ROUNDS=%d" % (REPS, ROUNDS))
    print("=" * 96)
    print("torch %s  cuda %s  bnb %s  gpu %s"
          % (torch.__version__, torch.version.cuda, bnb.__version__,
             torch.cuda.get_device_name(0)))
    f, t = torch.cuda.mem_get_info()
    print("可用显存 %.2f GiB / %.2f GiB" % (f / 1024**3, t / 1024**3))
    print()

    # 离群列：解释 threshold 的实际含义
    print("【threshold=6.0 的含义】batch=1 时 absmax 就是那个值本身，|x|>6 才算离群")
    for w in (4096, 8192, 12288):
        x1 = torch.randn(1, w, device="cuda", dtype=torch.bfloat16)
        _, _, i1 = F.int8_vectorwise_quant(x1.to(torch.float16), 6.0)
        x3 = torch.randn(1, w, device="cuda", dtype=torch.bfloat16) * 3
        _, _, i3 = F.int8_vectorwise_quant(x3.to(torch.float16), 6.0)
        n1 = 0 if i1 is None else int(i1.numel())
        n3 = 0 if i3 is None else int(i3.numel())
        print("  宽度 %-6d  RMS=1: %3d 列(%.2f%%)   RMS=3: %3d 列(%.2f%%)"
              % (w, n1, n1 / w * 100, n3, n3 / w * 100))
    print()

    allrows = []
    for title, spec in GROUPS:
        allrows += ab_group(title, spec)

    print("=" * 96)
    print("汇总：相对 bf16 的倍数（>1 慢，<1 快）")
    print("=" * 96)
    print("  %-26s %10s %10s %10s" % ("组", "int8 t=6.0", "int8 t=0.0", "4bit nf4"))
    for title, _s in GROUPS:
        line = "  %-26s" % title
        for label, _, _ in VARIANTS:
            vals = [r[3] for r in allrows if r[0] == title and r[2] == label]
            line += "%10s" % ("/".join("%.2fx" % v for v in vals) if vals else "-")
        print(line)
    print()
    print("  说明：每个格里是 batch=1 / batch=256 两个比值。")
    print()
    print("跑完 %s" % time.strftime("%F %T"))


if __name__ == "__main__":
    main()
