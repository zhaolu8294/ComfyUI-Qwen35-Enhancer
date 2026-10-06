# -*- coding: utf-8 -*-
"""
Qwen3.5-9B 解码期 Linear 投影的成本分解 —— 真实形状，不加载完整模型。

为什么要这么测
--------------
系统内存只剩 4.9 GB（ComfyUI 占 ~29 GB 工作集），18 GB 的 safetensors mmap 不起来
（OSError 1455 页面文件太小），所以完整 9B 这轮加载不了。
但**解码期的耗时几乎全部在 Linear 投影上**，而投影的成本只取决于形状 + 数量 + dtype，
不取决于权重数值。所以直接按 config 的真实形状把全部投影建出来，在 seq=1 上测。

Qwen3.5-9B 的真实结构（hidden 4096 / intermediate 12288 / 32 层）
---------------------------------------------------------------
线性注意力层 × 24（layer_types 里 linear_attention，full_attention_interval=4）：
    in_proj_qkv  4096 -> 8192   (conv_dim = key_dim*2 + value_dim = 2048*2+4096)
    in_proj_z    4096 -> 4096
    in_proj_b    4096 -> 32     ← 131k 参数，也要走一遍 bnb 内核
    in_proj_a    4096 -> 32     ← 同上
    out_proj     4096 -> 4096
    = 5 次 Linear 调用/层 → 24 层 = 120 次
全注意力层 × 8：
    q 4096->4096, k 4096->1024, v 4096->1024, o 4096->4096  → 32 次
MLP × 32：gate/up 4096->12288, down 12288->4096 → 96 次
合计 248 次 bnb 内核调用 / token。

输入按 post-RMSNorm 的量级构造（单位 RMS），这样离群列阈值的比较才有意义。
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
WARMUP = 3
REPS = 8

H = 4096
LIN_SPEC = [
    ("in_proj_qkv", H, 8192, 24),
    ("in_proj_z  ", H, H, 24),
    ("in_proj_b  ", H, 32, 24),
    ("in_proj_a  ", H, 32, 24),
    ("out_proj   ", H, H, 24),
]
ATTN_SPEC = [("q  ", H, H, 8), ("k  ", H, 1024, 8), ("v  ", H, 1024, 8), ("o  ", H, H, 8)]
MLP_SPEC = [("gate", H, 12288, 32), ("up  ", H, 12288, 32), ("down", 12288, H, 32)]

# 8bit 里 threshold 只在读 state 时用；4bit 无此概念
VARIANTS = [
    ("bf16      ", "bf16", 0.0),
    ("int8 t=6.0", "int8", 6.0),
    ("int8 t=0.0", "int8", 0.0),
    ("4bit nf4  ", "4bit", 0.0),
]


def make(kind, in_f, out_f, thr, dtype):
    if kind == "bf16":
        w = torch.randn(out_f, in_f, device="cuda", dtype=dtype) * 0.02
        lin = torch.nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(w)
        del w
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


def build(spec, kind, thr, dtype, batch=1):
    mods, xs = [], {}
    for name, in_f, out_f, cnt in spec:
        for _ in range(cnt):
            mods.append(make(kind, in_f, out_f, thr, dtype))
        if in_f not in xs:
            xs[in_f] = torch.randn(batch, in_f, device="cuda", dtype=dtype)
    return mods, xs


def run_once(mods, xs, spec):
    i = 0
    out = None
    for name, in_f, out_f, cnt in spec:
        x = xs[in_f]
        for _ in range(cnt):
            out = mods[i](x)
            i += 1
    return out


def timed(mods, xs, spec, dtype):
    with torch.no_grad():
        for _ in range(WARMUP):
            out = run_once(mods, xs, spec)
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = run_once(mods, xs, spec)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ok = bool(torch.isfinite(out).all())
    return statistics.median(ts), min(ts), ok


def total_params(spec):
    return sum(in_f * out_f * cnt for _, in_f, out_f, cnt in spec)


def measure_group(title, spec, dtype, batch=1):
    print("-" * 92)
    print("%s   （batch=%d，%d 次 Linear 调用 / token，%.2f B 参数）"
          % (title, batch, sum(c for *_r, c in spec), total_params(spec) / 1e9))
    print("-" * 92)
    print("  %-12s %10s %10s %12s %10s %8s" % ("变体", "中位 ms", "最快 ms", "等效带宽", "相对bf16", "有限值"))
    res = {}
    base = None
    for label, kind, thr in VARIANTS:
        try:
            mods, xs = build(spec, kind, thr, dtype, batch)
            med, best, ok = timed(mods, xs, spec, dtype)
            wb = total_params(spec) * (2 if kind == "bf16" else (0.5 if kind == "4bit" else 1.0))
            bw = wb / (med / 1000) / 1024**3
            if base is None:
                base = med
            print("  %-12s %10.3f %10.3f %9.0f GB/s %9.2fx %8s"
                  % (label, med, best, bw, med / base, "是" if ok else "**否**"))
            res[label.strip()] = med
            del mods, xs
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as ex:
            print("  %-12s 失败: %s: %s" % (label, type(ex).__name__, str(ex)[:80]))
    print()
    return res


def outlier_probe(dtype):
    print("-" * 92)
    print("【离群列检测】threshold=6.0 在单位 RMS 的激活上挑出多少列（batch=1 decode）")
    print("-" * 92)
    for shape in (4096, 8192, 12288):
        x = torch.randn(1, shape, device="cuda", dtype=torch.bfloat16)
        _, _, idx = F.int8_vectorwise_quant(x.to(torch.float16), 6.0)
        n = 0 if idx is None else int(idx.numel())
        x3 = (torch.randn(1, shape, device="cuda", dtype=torch.bfloat16) * 3)
        _, _, idx3 = F.int8_vectorwise_quant(x3.to(torch.float16), 6.0)
        n3 = 0 if idx3 is None else int(idx3.numel())
        print("  宽度 %-6d  RMS=1: %4d 列 (%.2f%%)   RMS=3: %4d 列 (%.2f%%)"
              % (shape, n, n / shape * 100, n3, n3 / shape * 100))
    print("  ↳ batch=1 时 absmax 就是单个值，阈值 6.0 意味着 |x|>6 才算离群。")
    print("    post-RMSNorm 的激活 RMS≈1，|x|>6 极罕见 → 解码时混合路径基本不触发。")
    print()


def main():
    print("=" * 92)
    print("Qwen3.5-9B 解码期 Linear 投影成本分解（真实形状）")
    print("=" * 92)
    print("torch %s  cuda %s  bnb %s  gpu %s"
          % (torch.__version__, torch.version.cuda, bnb.__version__,
             torch.cuda.get_device_name(0)))
    f, t = torch.cuda.mem_get_info()
    print("可用显存 %.2f GiB / %.2f GiB" % (f / 1024**3, t / 1024**3))
    print("输入按 post-RMSNorm 量级构造（单位 RMS）")
    print()

    outlier_probe(torch.bfloat16)

    GROUPS = [("线性注意力投影 ×24 层", LIN_SPEC),
              ("全注意力投影 ×8 层", ATTN_SPEC),
              ("MLP ×32 层", MLP_SPEC)]
    res = {}                                   # (batch, group) -> {variant: ms}
    for batch in (1, 256):
        print()
        print("#" * 92)
        print("# batch = %d   (%s)" % (batch, "decoder 单步" if batch == 1 else "prefill 一块"))
        print("#" * 92)
        print()
        for title, spec in GROUPS:
            res[(batch, title)] = measure_group(title, spec, torch.bfloat16, batch)

    for batch in (1, 256):
        print("=" * 92)
        print("合并：batch=%d 时整个模型的投影成本（三组相加）" % batch)
        print("=" * 92)
        print("  %-12s %12s %12s %8s" % ("变体", "合计 ms", "相对bf16", "(batch=1 → tok/s 上界)" if batch == 1 else ""))
        base = None
        for label, _, _ in VARIANTS:
            key = label.strip()
            vals = [res[(batch, g[0])].get(key) for g in GROUPS]
            if any(v is None for v in vals):
                print("  %-12s %12s" % (key, "数据不全"))
                continue
            tot = sum(vals)
            if base is None:
                base = tot
            tail = ("%8.2f tok/s" % (1000.0 / tot)) if batch == 1 else ""
            print("  %-12s %12.1f %8.2fx %s" % (key, tot, tot / base, tail))
        print()

    # ---- 按用户实际工作负载加权（04:55 那次：prefill ~3000 tok ×2，decode 543 tok）----
    PREFILL_TOK = 3000
    DECODE_TOK = 543
    print("=" * 92)
    print("按你 04:55 那次的真实负载折算投影时间（prefill %d tok ×2 段，decode %d tok）"
          % (PREFILL_TOK, DECODE_TOK))
    print("=" * 92)
    print("  %-12s %14s %14s %12s %10s" % ("变体", "prefill ms", "decode ms", "合计 ms", "相对bf16"))
    base = None
    for label, _, _ in VARIANTS:
        key = label.strip()
        try:
            pre = sum(res[(256, g[0])][key] for g in GROUPS) * (PREFILL_TOK / 256.0) * 2
            dec = sum(res[(1, g[0])][key] for g in GROUPS) * DECODE_TOK
        except KeyError:
            print("  %-12s 数据不全" % key)
            continue
        tot = pre + dec
        if base is None:
            base = tot
        print("  %-12s %14.1f %14.1f %12.1f %9.2fx" % (key, pre, dec, tot, tot / base))
    print()
    print("注：只算 Linear 投影，不含线性注意力的门控/卷积/逐 token 递推（实测 24.2 ms/token）、")
    print("    embedding、lm_head、采样。所以这是投影部分的占比，不是整轮耗时。")
    print()
    print("跑完 %s" % time.strftime("%F %T"))


if __name__ == "__main__":
    main()
