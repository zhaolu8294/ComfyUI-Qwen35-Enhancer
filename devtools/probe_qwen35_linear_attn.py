"""Qwen3.5 解码慢的第二病灶探针：线性注意力走了 torch 回退实现。

背景
----
Qwen3.5-9B 是**混合架构**：32 层里 24 层是 Gated DeltaNet 线性注意力，
只有 8 层是全注意力（full_attention_interval=4）。

transformers 5.9.0 的 modeling_qwen3_5.py 里：

    is_fast_path_available = all(
        (causal_conv1d_fn, causal_conv1d_update,
         chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
    )

前两个来自 causal_conv1d，后两个来自 fla（flash-linear-attention）。
两者都没装 -> 四个符号全为 None -> 全部退回 modeling 文件里自带的 torch 实现：

    self.chunk_gated_delta_rule     = chunk_gated_delta_rule     or torch_chunk_gated_delta_rule
    self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

其中 decode 走的是 torch_recurrent_gated_delta_rule，它的实现是：

    query, key, value, beta, g = [x.transpose(1,2).contiguous().to(torch.float32) for x in (...)]
    ...
    for i in range(sequence_length):     # <-- Python 循环
        ...
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        ...
即 fp32、逐 token、一串小张量算子。

本探针回答一个问题：**这 24 层回退实现，每 token 到底吃掉多少毫秒？**
（对照基准：bf16 实测解码 20.9 tok/s = 47.6 ms/token）

用法
----
    E:/AI/ComfyUI-aki-v3/python/python.exe probe_qwen35_linear_attn.py
"""
import os
import sys
import time
import json

import torch

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\text_encoders\huihui-ai_Huihui-Qwen3.5-9B-abliterated"
DEV = "cuda"
DT = torch.bfloat16

# 实测基准（bf16 / quant=none / 05:07 那次日志）
REF_MS_PER_TOKEN = 1000.0 / 20.9


def bench(fn, n=50, warm=10):
    """返回单次调用的毫秒数（GPU 同步）。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    torch.cuda.init()
    print("=" * 72)
    print("GPU :", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "cuda", torch.version.cuda)
    print("=" * 72)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config

    n_layer = tc.num_hidden_layers
    types = tc.layer_types
    n_lin = sum(1 for t in types if t == "linear_attention")
    n_full = n_layer - n_lin
    print(f"层数 {n_layer}：linear_attention {n_lin} / full_attention {n_full}")
    print(f"linear: k_heads={tc.linear_num_key_heads} v_heads={tc.linear_num_value_heads} "
          f"k_dim={tc.linear_key_head_dim} v_dim={tc.linear_value_head_dim} "
          f"conv_kernel={tc.linear_conv_kernel_dim}  mamba_ssm_dtype={tc.mamba_ssm_dtype}")
    print(f"full  : heads={tc.num_attention_heads} kv_heads={tc.num_key_value_heads} head_dim={tc.head_dim}")
    print(f"参考基准（bf16 实测解码 20.9 tok/s）: {REF_MS_PER_TOKEN:.1f} ms/token")
    print()

    import transformers.models.qwen3_5.modeling_qwen3_5 as Q5

    # ------------------------------------------------------------------
    # 快路径可用性
    # ------------------------------------------------------------------
    print("-" * 72)
    print("[1] 快路径可用性")
    print("-" * 72)
    print(f"  is_fast_path_available              = {Q5.is_fast_path_available}")
    print(f"  causal_conv1d_fn                    = {Q5.causal_conv1d_fn}")
    print(f"  causal_conv1d_update                = {Q5.causal_conv1d_update}")
    print(f"  chunk_gated_delta_rule (fla)        = {Q5.chunk_gated_delta_rule}")
    print(f"  fused_recurrent_gated_delta_rule    = {Q5.fused_recurrent_gated_delta_rule}")
    print(f"  FusedRMSNormGated (fla)             = {Q5.FusedRMSNormGated}")
    print()

    H = tc.linear_num_value_heads
    K = tc.linear_key_head_dim
    V = tc.linear_value_head_dim
    B = 1
    T = 1

    # ------------------------------------------------------------------
    # [2] 裸测 torch_recurrent_gated_delta_rule（decode 实际走的那条）
    # ------------------------------------------------------------------
    print("-" * 72)
    print("[2] torch_recurrent_gated_delta_rule @ decode 真实形状（裸算子，脱离模型）")
    print("-" * 72)
    q = torch.randn(B, T, H, K, device=DEV, dtype=DT)
    k = torch.randn(B, T, H, K, device=DEV, dtype=DT)
    v = torch.randn(B, T, H, V, device=DEV, dtype=DT)
    g = -torch.rand(B, T, H, device=DEV, dtype=DT) * 0.1
    beta = torch.rand(B, T, H, device=DEV, dtype=DT)
    st = torch.zeros(B, H, K, V, device=DEV, dtype=DT)   # 模型里 cache 存的就是 bf16

    t_rec = bench(lambda: Q5.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, st, False, use_qk_l2norm_in_kernel=True))
    print(f"  单层单 token 一次调用       : {t_rec:8.3f} ms")
    print(f"  × {n_lin} 层（每 token）        : {t_rec * n_lin:8.2f} ms"
          f"   占 {t_rec * n_lin / REF_MS_PER_TOKEN * 100:.1f}%")
    print()

    # 只算张量搬运的量级：state 是 (1,32,128,128)
    # 注意：模型里 DynamicCache 存的是 bf16，本探针早期误用 fp32，见 verify_fla_equivalence.py
    state_mb = st.numel() * st.element_size() / 1024 / 1024
    traffic_mb = state_mb * 4            # 单层一次 ≈ 读 2 次 + 写 2 次
    print(f"  单层 recurrent state 体积  : {state_mb:.2f} MiB ({st.dtype})")
    print(f"  单层一次 ≈ 读 2 次 + 写 2 次 state ≈ {traffic_mb:.1f} MiB 流量")
    print(f"  × {n_lin} 层 ≈ {traffic_mb * n_lin:.0f} MiB / token")
    gbps = traffic_mb * 1024 * 1024 / (t_rec / 1000.0) / 1e9
    print(f"  -> 等效带宽约 {gbps:.1f} GB/s（4090D 峰值约 1008 GB/s）"
          f"  = {gbps / 1008 * 100:.1f}% 利用率")
    print("     （利用率这么低 = 瓶颈不在带宽，而在小算子数量/启动延迟）")
    print()

    # ------------------------------------------------------------------
    # [3] 真层前向：priming + 逐步 decode
    # ------------------------------------------------------------------
    print("-" * 72)
    print(f"[3] Qwen3_5GatedDeltaNet 真层前向（layer_idx=0，模型目录里的真配置）")
    print("-" * 72)
    try:
        layer = Q5.Qwen3_5GatedDeltaNet(tc, 0).to(DEV, DT).eval()
        nparam = sum(p.numel() for p in layer.parameters())
        print(f"  单层参数量 {nparam/1e6:.1f} M（bf16 ≈ {nparam*2/1024/1024:.0f} MiB）")

        # 关键：确认它到底挂的是 fla 还是 torch 回退
        print(f"  layer.recurrent_gated_delta_rule = "
              f"{getattr(layer.recurrent_gated_delta_rule, '__module__', '?')}."
              f"{getattr(layer.recurrent_gated_delta_rule, '__name__', '?')}")
        print(f"  layer.chunk_gated_delta_rule     = "
              f"{getattr(layer.chunk_gated_delta_rule, '__module__', '?')}."
              f"{getattr(layer.chunk_gated_delta_rule, '__name__', '?')}")
        print(f"  layer.norm                       = {type(layer.norm).__name__}")

        cache = Q5.DynamicCache(config=cfg)
        with torch.no_grad():
            # priming：用一段 prefill 建立 conv/recurrent state
            pre = torch.randn(1, 128, tc.hidden_size, device=DEV, dtype=DT)
            layer(pre, cache_params=cache)
            torch.cuda.synchronize()
            primed = cache.has_previous_state(0)
        print(f"  priming 完成，cache.has_previous_state(0) = {primed}")

        h1 = torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT)

        def one_step():
            with torch.no_grad():
                layer(h1, cache_params=cache)

        t_layer = bench(one_step, n=30, warm=5)
        print()
        print(f"  单层单 token 前向（decode 路径）: {t_layer:8.3f} ms")
        print(f"  × {n_lin} 层（每 token）          : {t_layer * n_lin:8.2f} ms"
              f"   占 {t_layer * n_lin / REF_MS_PER_TOKEN * 100:.1f}%")
        print()

        # 拆开看：投影 vs 注意力核心
        hs = torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT)
        def proj_only():
            with torch.no_grad():
                layer.in_proj_qkv(hs); layer.in_proj_z(hs)
                layer.in_proj_b(hs); layer.in_proj_a(hs)
        t_proj = bench(proj_only, n=30, warm=5)
        print(f"  其中 4 个投影 Linear        : {t_proj:8.3f} ms"
              f"   占单层 {t_proj / t_layer * 100:.0f}%")
        print(f"  其余（conv + gated delta + norm + out_proj）: {t_layer - t_proj:8.3f} ms"
              f"   占 {100 - t_proj / t_layer * 100:.0f}%")
        print()

        # 全模型粗算
        total_lin_ms = t_layer * n_lin
        print(f"  >>> 24 层线性注意力合计约 {total_lin_ms:.1f} ms/token"
              f"（基准 {REF_MS_PER_TOKEN:.1f} ms/token）")

        # 权重显存带宽下限：全模型 bf16 权重
        wbytes = 0
        with torch.no_grad():
            for nm, p in layer.named_parameters():
                wbytes += p.numel() * 2
        full_model_bytes = 17.5 * 1024 ** 3   # 日志实测
        print(f"  全模型权重 {full_model_bytes/1024**3:.1f} GiB -> 纯带宽下限 "
              f"{full_model_bytes/1008e9*1000:.1f} ms/token @1008GB/s")
        print(f"  线性注意力层权重占比 {(wbytes/1024**3)/(full_model_bytes/1024**3)*100:.1f}%"
              f"（{wbytes*n_lin/1024**3:.2f} GiB）")
    except Exception as e:
        import traceback
        print("  !! 真层测试失败:", type(e).__name__, e)
        traceback.print_exc()

    print()
    print("=" * 72)
    print("完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
