# -*- coding: utf-8 -*-
"""
真实 Qwen3.5-9B 上的量化档位 A/B —— 直接量解码 tok/s，不再靠外推。

用法：
    python -u bench_real_quant.py 8bit     # 加载 8bit，运行时翻转 threshold
    python -u bench_real_quant.py 4bit
    python -u bench_real_quant.py none

为什么翻转 threshold 而不用重载
------------------------------
bnb 0.49.2 的 MatMul8bitLt.forward 每次调用都现读 state.threshold：
    if state.threshold > 0.0:  int8_mixed_scaled_mm(...)   # 离群列分解
    else:                      int8_scaled_mm(...)         # 纯融合内核
所以改 state.threshold 立刻生效，同一份权重、同一份输入，完全受控 ——
和之前 A/B 注意力后端用的是同一个手法。

其它被测的量
------------
* Linear8bitLt 模块个数 → 每 token 要付多少次 bnb 内核调用
* 各 weight 的 dtype 分布 → 确认视觉塔 / lm_head 有没有被排除
* prefill（seq=512）与 decode（128 步）分别计时
"""
import os
import sys
import time
import json
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

MODE = (sys.argv[1] if len(sys.argv) > 1 else "8bit").lower()
MODEL = r"E:/AI/ComfyUI-aki-v3/ComfyUI/models/text_encoders/huihui-ai_Huihui-Qwen3.5-9B-abliterated"
PROMPT_LEN = 512
NEW_TOKENS = 128


def hdr(s):
    print()
    print("=" * 92)
    print(s)
    print("=" * 92, flush=True)


def load():
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    cfg = None
    if MODE == "8bit":
        cfg = BitsAndBytesConfig(load_in_8bit=True,
                                 llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    elif MODE == "4bit":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_skip_modules=["model.visual", "visual", "lm_head"])

    kw = dict(device_map="auto", attn_implementation="sdpa",
              max_memory={0: "21000MiB", "cpu": "0MiB"})
    if cfg is not None:
        kw["quantization_config"] = cfg
        kw["dtype"] = "auto"
    else:
        kw["dtype"] = torch.bfloat16

    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
    model.eval()
    print("[load] %.1f s" % (time.perf_counter() - t0), flush=True)
    return model


def report_modules(model):
    hdr("模块统计")
    import bitsandbytes as bnb
    counts = {}
    for m in model.modules():
        counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    for k in ("Linear8bitLt", "Linear4bit", "Linear", "Conv1d"):
        if k in counts:
            extra = ""
            if k == "Linear8bitLt":
                thr = {round(float(m.state.threshold), 3) for m in model.modules()
                       if isinstance(m, bnb.nn.Linear8bitLt)}
                extra = "  threshold=%s" % sorted(thr)
            if k == "Linear":
                extra = "  (未被量化)"
            print("  %-16s %4d 个%s" % (k, counts[k], extra))

    # dtype / device 分布
    dt, dev = {}, {}
    for n, p in model.named_parameters():
        dt[str(p.dtype)] = dt.get(str(p.dtype), 0) + p.numel()
        dev[str(p.device)] = dev.get(str(p.device), 0) + p.numel()
    tot = sum(dt.values())
    print("  ---- 参数 dtype 分布（共 %.2f B 参数）----" % (tot / 1e9))
    for k, v in sorted(dt.items(), key=lambda x: -x[1]):
        print("    %-16s %7.2f B  (%.1f%%)" % (k, v / 1e9, v / tot * 100))
    print("  ---- 参数 device 分布 ----")
    for k, v in sorted(dev.items(), key=lambda x: -x[1]):
        print("    %-16s %7.2f B  (%.1f%%)" % (k, v / 1e9, v / tot * 100))
    return counts


def set_threshold(model, val):
    import bitsandbytes as bnb
    n = 0
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear8bitLt):
            m.state.threshold = float(val)
            n += 1
    return n


def measure_decode(model, label):
    """强制生成恰好 NEW_TOKENS 个 token，测纯解码速率。"""
    random.seed(0)
    ids = torch.randint(0, 200000, (1, PROMPT_LEN), device="cuda")
    with torch.no_grad():
        # 预热一次，把 KV cache / 内核都点着
        model.generate(input_ids=ids, max_new_tokens=8, min_new_tokens=8,
                       do_sample=False, eos_token_id=None, pad_token_id=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(input_ids=ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                             do_sample=False, eos_token_id=None, pad_token_id=0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    n = out.shape[1] - PROMPT_LEN
    rate = n / dt
    print("  %-26s %7.2f s / %3d tok  ->  %6.2f tok/s   (%.1f ms/token)"
          % (label, dt, n, rate, dt / max(n, 1) * 1000), flush=True)
    return rate


def measure_prefill(model, label):
    ids = torch.randint(0, 200000, (1, PROMPT_LEN), device="cuda")
    with torch.no_grad():
        for _ in range(2):
            model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3
    print("  %-26s prefill(seq=%d) %7.3f s" % (label, PROMPT_LEN, dt), flush=True)
    return dt


def main():
    hdr("真实模型量化 A/B   mode=%s" % MODE)
    print("torch %s  cuda %s  gpu %s" % (torch.__version__, torch.version.cuda,
                                        torch.cuda.get_device_name(0)))
    f, t = torch.cuda.mem_get_info()
    print("载入前可用显存 %.2f GiB / %.2f GiB" % (f / 1024**3, t / 1024**3))

    model = load()
    f2, _ = torch.cuda.mem_get_info()
    print("载入后可用显存 %.2f GiB（模型占用约 %.2f GiB）"
          % (f2 / 1024**3, (f - f2) / 1024**3), flush=True)

    report_modules(model)
    measure_prefill(model, "初始")

    hdr("解码速率对比")
    if MODE == "8bit":
        for val in (6.0, 0.0, 1e9):
            n = set_threshold(model, val)
            print("  [已把 %d 个 Linear8bitLt 的 threshold 设为 %s]" % (n, val), flush=True)
            if val == 1e9:
                # 1e9 会让离群检测挑出几乎所有列 -> 反证用；可能极慢，先小样本试
                print("  （threshold=1e9 是反证项，跳过以免耗时）")
                continue
            measure_decode(model, "int8 threshold=%s" % val)
        set_threshold(model, 6.0)
    else:
        measure_decode(model, "quant=%s" % MODE)

    print()
    print("跑完 %s  mode=%s" % (time.strftime("%F %T"), MODE), flush=True)


if __name__ == "__main__":
    main()
