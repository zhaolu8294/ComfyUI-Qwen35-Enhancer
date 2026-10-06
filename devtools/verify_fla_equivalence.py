"""fla 内核 vs torch 回退：数值等价性验证。

上一轮 A/B 里 max|Δ| = nan —— 先查清是 fla 的问题还是我测试的问题
（我把 initial_state 传成了 fp32，而模型里 DynamicCache 存的是 bf16）。

用法：
    PYTHONPATH=H:/_fla_test E:/AI/ComfyUI-aki-v3/python/python.exe -u verify_fla_equivalence.py
"""
import torch

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\text_encoders\huihui-ai_Huihui-Qwen3.5-9B-abliterated"
DEV = "cuda"
DT = torch.bfloat16

OK = []
BAD = []


def check(name, cond, extra=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))


def main():
    torch.cuda.init()
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q5

    has_fla = (Q5.fused_recurrent_gated_delta_rule is not None
               and Q5.fused_recurrent_gated_delta_rule is not Q5.torch_recurrent_gated_delta_rule)
    print("=" * 74)
    print("fla 内核已启用 :", has_fla)
    print("=" * 74)
    if not has_fla:
        print("没启用 fla，本脚本需要在 PYTHONPATH=H:/_fla_test 下运行")
        return

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tc = cfg.text_config
    H = tc.linear_num_value_heads
    K = tc.linear_key_head_dim
    V = tc.linear_value_head_dim

    # ------------------------------------------------------------------
    # [1] 裸算子：recurrent（decode）—— 换 dtype 找出 nan 的来源
    # ------------------------------------------------------------------
    print()
    print("-" * 74)
    print("[1] recurrent 规则：initial_state 的 dtype 是关键")
    print("-" * 74)
    torch.manual_seed(0)
    q = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
    k = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
    v = torch.randn(1, 1, H, V, device=DEV, dtype=DT)
    g = -torch.rand(1, 1, H, device=DEV, dtype=DT) * 0.1
    beta = torch.rand(1, 1, H, device=DEV, dtype=DT)

    for st_dtype, label in ((torch.float32, "fp32 state"), (DT, "bf16 state")):
        st = torch.zeros(1, H, K, V, device=DEV, dtype=st_dtype)
        with torch.no_grad():
            o_t, s_t = Q5.torch_recurrent_gated_delta_rule(
                q, k, v, g, beta, st, True, use_qk_l2norm_in_kernel=True)
            o_f, s_f = Q5.fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=st.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
        finite_t = bool(torch.isfinite(o_t).all())
        finite_f = bool(torch.isfinite(o_f).all())
        if finite_t and finite_f:
            d = (o_t.float() - o_f.float()).abs()
            denom = o_t.float().abs().max().clamp_min(1e-12)
            print(f"  {label:12s}: torch finite={finite_t} fla finite={finite_f} "
                  f"max|Δ|={float(d.max()):.3e} 相对={float(d.max()/denom):.3e}")
            if st_dtype == DT:
                check("bf16 state 下 recurrent 输出数值等价（相对误差 < 1e-2）",
                      float(d.max() / denom) < 1e-2, f"{float(d.max()/denom):.2e}")
        else:
            print(f"  {label:12s}: torch finite={finite_t} fla finite={finite_f}  <-- nan 来源在这里")
            if st_dtype == torch.float32:
                check("fp32 state 会让 fla 产出 nan（说明我上一轮测错了 dtype）",
                      not finite_f)
    print()
    print("  >>> 结论：模型里 DynamicCache 存的是 bf16 state，所以等价性要看 bf16 那一行")
    print()

    # ------------------------------------------------------------------
    # [2] 裸算子：chunk（prefill 走这条）
    # ------------------------------------------------------------------
    print("-" * 74)
    print("[2] chunk 规则（prefill 路径）")
    print("-" * 74)
    T = 64
    torch.manual_seed(1)
    qc = torch.randn(1, T, H, K, device=DEV, dtype=DT)
    kc = torch.randn(1, T, H, K, device=DEV, dtype=DT)
    vc = torch.randn(1, T, H, V, device=DEV, dtype=DT)
    gc = -torch.rand(1, T, H, device=DEV, dtype=DT) * 0.1
    bc = torch.rand(1, T, H, device=DEV, dtype=DT)
    stc = torch.zeros(1, H, K, V, device=DEV, dtype=DT)
    try:
        with torch.no_grad():
            o_tc, _ = Q5.torch_chunk_gated_delta_rule(
                qc, kc, vc, gc, bc, initial_state=stc.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
            o_fc, _ = Q5.chunk_gated_delta_rule(
                qc, kc, vc, gc, bc, initial_state=stc.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True)
        dc = (o_tc.float() - o_fc.float()).abs()
        den = o_tc.float().abs().max().clamp_min(1e-12)
        print(f"  torch finite={bool(torch.isfinite(o_tc).all())} "
              f"fla finite={bool(torch.isfinite(o_fc).all())}")
        print(f"  max|Δ|={float(dc.max()):.3e} 相对={float(dc.max()/den):.3e}")
        check("chunk 规则输出数值等价（相对误差 < 1e-2）",
              float(dc.max() / den) < 1e-2, f"{float(dc.max()/den):.2e}")
    except Exception as e:
        import traceback
        print("  chunk 对比失败:", type(e).__name__, e)
        traceback.print_exc()
    print()

    # ------------------------------------------------------------------
    # [3] 真层端到端：同一份权重，只换 delta 规则
    # ------------------------------------------------------------------
    print("-" * 74)
    print("[3] Qwen3_5GatedDeltaNet 真层端到端（同权重，只换 delta 规则）")
    print("-" * 74)
    try:
        layer = Q5.Qwen3_5GatedDeltaNet(tc, 0).to(DEV, DT).eval()
        cache = Q5.DynamicCache(config=cfg)
        x = torch.randn(1, 96, tc.hidden_size, device=DEV, dtype=DT)
        with torch.no_grad():
            out_fla = layer(x, cache_params=cache)
        torch.cuda.synchronize()
        st_f = cache.layers[0].recurrent_states
        print(f"  fla 路径 prefill 输出 finite={bool(torch.isfinite(out_fla).all())}, "
              f"state dtype={st_f.dtype}, shape={tuple(st_f.shape)}")

        # 同一个层，把两条规则换回 torch 实现，重跑
        layer.recurrent_gated_delta_rule = Q5.torch_recurrent_gated_delta_rule
        layer.chunk_gated_delta_rule = Q5.torch_chunk_gated_delta_rule
        cache2 = Q5.DynamicCache(config=cfg)
        with torch.no_grad():
            out_torch = layer(x, cache_params=cache2)
        torch.cuda.synchronize()
        st_t = cache2.layers[0].recurrent_states
        print(f"  torch 路径 prefill 输出 finite={bool(torch.isfinite(out_torch).all())}, "
              f"state dtype={st_t.dtype}")

        d = (out_torch.float() - out_fla.float()).abs()
        den = out_torch.float().abs().max().clamp_min(1e-12)
        print(f"  prefill 输出 max|Δ|={float(d.max()):.3e} 相对={float(d.max()/den):.3e}")
        ds = (st_t.float() - st_f.float()).abs()
        dens = st_t.float().abs().max().clamp_min(1e-12)
        print(f"  state   输出 max|Δ|={float(ds.max()):.3e} 相对={float(ds.max()/dens):.3e}")
        check("真层 prefill 输出等价（相对误差 < 2e-2）",
              float(d.max() / den) < 2e-2, f"{float(d.max()/den):.2e}")

        # 再走一次 decode（seq=1）
        h1 = torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT)
        with torch.no_grad():
            od_f = layer(h1, cache_params=cache2)   # 此时 layer 已被换成 torch 实现
        layer.recurrent_gated_delta_rule = Q5.fused_recurrent_gated_delta_rule
        layer.chunk_gated_delta_rule = Q5.chunk_gated_delta_rule
        with torch.no_grad():
            od_t = layer(h1, cache_params=cache)    # fla
        dd = (od_t.float() - od_f.float()).abs()
        dend = od_f.float().abs().max().clamp_min(1e-12)
        print(f"  decode  输出 max|Δ|={float(dd.max()):.3e} 相对={float(dd.max()/dend):.3e}")
        check("真层 decode 输出等价（相对误差 < 5e-2）",
              float(dd.max() / dend) < 5e-2, f"{float(dd.max()/dend):.2e}")
    except Exception as e:
        import traceback
        print("  !! 真层测试失败:", type(e).__name__, e)
        traceback.print_exc()
        BAD.append("真层端到端")

    print()
    print("=" * 74)
    print(f"通过 {len(OK)} 项，失败 {len(BAD)} 项")
    if BAD:
        print("失败项:", BAD)
    print("=" * 74)


if __name__ == "__main__":
    main()
