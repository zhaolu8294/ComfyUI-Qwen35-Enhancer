"""fla 内核 vs transformers 自带 torch 回退 —— A/B 实测（修正版）。

第一版写错了调用方式：fla 的签名是
    fused_recurrent_gated_delta_rule(q, k, v, g=None, gk=None, gv=None, beta=None, ...)
位置参数会把 beta 塞进 gk 槽位 -> 输出 NaN、耗时不成立。
本版一律用关键字参数。

用法：
    PYTHONPATH=H:/_fla_test E:/AI/ComfyUI-aki-v3/python/python.exe -u bench_fla_vs_torch.py
"""
import os
import time

import torch

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\text_encoders\huihui-ai_Huihui-Qwen3.5-9B-abliterated"
DEV = "cuda"
DT = torch.bfloat16
REF_TOK_S = 20.9
REF_MS_PER_TOKEN = 1000.0 / REF_TOK_S


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


torch.cuda.init()
print("=" * 78)
print("GPU :", torch.cuda.get_device_name(0))
try:
    import fla
    print("fla :", fla.__version__)
except Exception as e:
    print("fla : 未安装 -", e)
print("=" * 78)

import transformers.models.qwen3_5.modeling_qwen3_5 as Q5

has_fla = (Q5.fused_recurrent_gated_delta_rule is not None
           and Q5.fused_recurrent_gated_delta_rule is not Q5.torch_recurrent_gated_delta_rule)
print("fla 内核启用 :", has_fla)
print("  Q5.fused_recurrent_gated_delta_rule ->",
      getattr(Q5.fused_recurrent_gated_delta_rule, "__module__", None))
print("  Q5.chunk_gated_delta_rule           ->",
      getattr(Q5.chunk_gated_delta_rule, "__module__", None))
print("  Q5.FusedRMSNormGated                ->", Q5.FusedRMSNormGated)
print()

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL_DIR)
tc = cfg.text_config
n_lin = sum(1 for t in tc.layer_types if t == "linear_attention")
H, K, V = tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim
print(f"线性注意力 {n_lin}/{tc.num_hidden_layers} 层；基准 {REF_TOK_S} tok/s = "
      f"{REF_MS_PER_TOKEN:.1f} ms/token")
print()

torch.manual_seed(0)
q = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
k = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
v = torch.randn(1, 1, H, V, device=DEV, dtype=DT)
g = -torch.rand(1, 1, H, device=DEV, dtype=DT) * 0.1
beta = torch.rand(1, 1, H, device=DEV, dtype=DT)
st = torch.zeros(1, H, K, V, device=DEV, dtype=DT)

# ---------------------------------------------------------------- 裸算子
print("-" * 78)
print("[1] 裸算子 recurrent 规则（decode 路径，seq=1）")
print("-" * 78)
t_tor = bench(lambda: Q5.torch_recurrent_gated_delta_rule(
    q, k, v, g, beta, st, False, use_qk_l2norm_in_kernel=True))
print(f"  torch 回退 : {t_tor:8.3f} ms/层  × {n_lin} = {t_tor*n_lin:7.2f} ms/token"
      f"  ({t_tor*n_lin/REF_MS_PER_TOKEN*100:5.1f}% of 基准)")

if has_fla:
    with torch.no_grad():
        o1, _ = Q5.torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, st, False, use_qk_l2norm_in_kernel=True)
        o2, _ = Q5.fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=st,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
    d = (o1.float() - o2.float()).abs().max()
    den = o1.float().abs().max().clamp_min(1e-12)
    print(f"  数值检查   : torch finite={bool(torch.isfinite(o1).all())} "
          f"fla finite={bool(torch.isfinite(o2).all())} "
          f"相对误差={float(d/den):.2e}")

    t_fla = bench(lambda: Q5.fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=st,
        output_final_state=False, use_qk_l2norm_in_kernel=True))
    print(f"  fla 内核   : {t_fla:8.3f} ms/层  × {n_lin} = {t_fla*n_lin:7.2f} ms/token"
          f"  ({t_fla*n_lin/REF_MS_PER_TOKEN*100:5.1f}% of 基准)")
    print(f"  >>> 单层 {t_tor/t_fla:.2f}×，{n_lin} 层省 {(t_tor-t_fla)*n_lin:.1f} ms/token")

# 张量搬运视角
state_mb = st.numel() * 2 / 1024 / 1024
print()
print(f"  （参考）单层 state {state_mb:.2f} MiB(bf16)；若按流量算，"
      f"torch 版等效带宽仅 {state_mb*4*1024*1024/1073741824/(t_tor/1000):.1f} GB/s"
      f" = 峰值 1008 GB/s 的 {state_mb*4*1024*1024/1073741824/(t_tor/1000)/1008*100:.1f}%")
print()

# ---------------------------------------------------------------- 真层
print("-" * 78)
print("[2] Qwen3_5GatedDeltaNet 真层 decode 前向（同权重，只换 delta 规则）")
print("-" * 78)
layer = Q5.Qwen3_5GatedDeltaNet(tc, 0).to(DEV, DT).eval()
h1 = torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT)


def make_cache_and_prime():
    c = Q5.DynamicCache(config=cfg)
    with torch.no_grad():
        layer(torch.randn(1, 96, tc.hidden_size, device=DEV, dtype=DT), cache_params=c)
    torch.cuda.synchronize()
    return c


layer.recurrent_gated_delta_rule = Q5.torch_recurrent_gated_delta_rule
layer.chunk_gated_delta_rule = Q5.torch_chunk_gated_delta_rule
c_t = make_cache_and_prime()
t_layer_tor = bench(lambda: layer(h1, cache_params=c_t))
print(f"  torch 回退 : {t_layer_tor:8.3f} ms/层  × {n_lin} = "
      f"{t_layer_tor*n_lin:7.2f} ms/token  "
      f"({t_layer_tor*n_lin/REF_MS_PER_TOKEN*100:5.1f}% of 基准)")

if has_fla:
    layer.recurrent_gated_delta_rule = Q5.fused_recurrent_gated_delta_rule
    layer.chunk_gated_delta_rule = Q5.chunk_gated_delta_rule
    c_f = make_cache_and_prime()
    t_layer_fla = bench(lambda: layer(h1, cache_params=c_f))
    print(f"  fla 内核   : {t_layer_fla:8.3f} ms/层  × {n_lin} = "
          f"{t_layer_fla*n_lin:7.2f} ms/token  "
          f"({t_layer_fla*n_lin/REF_MS_PER_TOKEN*100:5.1f}% of 基准)")
    print(f"  >>> 真层 {t_layer_tor/t_layer_fla:.2f}×，"
          f"{n_lin} 层省 {(t_layer_tor-t_layer_fla)*n_lin:.1f} ms/token")
    print()
    print(f"  换算成整轮：解码 25.6 s（中英各约 12.8 s）里，"
          f"线性注意力部分省 {(t_layer_tor-t_layer_fla)*n_lin/1000*(263+280):.1f} s")

print()
print("=" * 78)
