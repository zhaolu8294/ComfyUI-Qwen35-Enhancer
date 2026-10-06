"""追查两件事：
  1) 裸算子测试里 fla fused_recurrent 为何出 NaN（是它的问题还是我构造输入的问题）
  2) 连续多步 decode 时 fla 与 torch 回退会不会漂移/爆掉
"""
import inspect
import torch

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\text_encoders\huihui-ai_Huihui-Qwen3.5-9B-abliterated"
DEV = "cuda"
DT = torch.bfloat16

import transformers.models.qwen3_5.modeling_qwen3_5 as Q5

print("=" * 74)
print("fla fused_recurrent_gated_delta_rule 签名")
print("=" * 74)
print(inspect.signature(Q5.fused_recurrent_gated_delta_rule))
doc = (inspect.getdoc(Q5.fused_recurrent_gated_delta_rule) or "")[:1400]
print(doc)
print()

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(MODEL_DIR)
tc = cfg.text_config
H = tc.linear_num_value_heads
K = tc.linear_key_head_dim
V = tc.linear_value_head_dim

# ---------------------------------------------------------------------------
# [1] 逐步排除 NaN 来源
# ---------------------------------------------------------------------------
print("=" * 74)
print("[1] 裸算子 NaN 排查")
print("=" * 74)
torch.manual_seed(0)


def mk(T, zero_state=True):
    return dict(
        q=torch.randn(1, T, H, K, device=DEV, dtype=DT),
        k=torch.randn(1, T, H, K, device=DEV, dtype=DT),
        v=torch.randn(1, T, H, V, device=DEV, dtype=DT),
        g=-torch.rand(1, T, H, device=DEV, dtype=DT) * 0.1,
        beta=torch.rand(1, T, H, device=DEV, dtype=DT),
        st=(torch.zeros if zero_state else torch.randn)(1, H, K, V, device=DEV, dtype=DT),
    )


cases = [
    ("T=1 零 state, l2norm=True,  out_state=True ", mk(1, True), True, True),
    ("T=1 零 state, l2norm=False, out_state=True ", mk(1, True), False, True),
    ("T=1 零 state, l2norm=True,  out_state=False", mk(1, True), True, False),
    ("T=1 非零state, l2norm=True,  out_state=True ", mk(1, False), True, True),
    ("T=8 零 state, l2norm=True,  out_state=True ", mk(8, True), True, True),
]
for label, a, l2, ofs in cases:
    try:
        with torch.no_grad():
            o, s = Q5.fused_recurrent_gated_delta_rule(
                a["q"], a["k"], a["v"], g=a["g"], beta=a["beta"],
                initial_state=a["st"].clone(), output_final_state=ofs,
                use_qk_l2norm_in_kernel=l2)
        fo = bool(torch.isfinite(o).all())
        fs = bool(torch.isfinite(s).all()) if s is not None else True
        print(f"  {label} -> o finite={fo}  s finite={fs}  "
              f"o.shape={tuple(o.shape)}"
              + (f"  s.shape={tuple(s.shape)}" if s is not None else ""))
        if not fo:
            bad = torch.isnan(o)
            print(f"      o 里 NaN {int(bad.sum())}/{bad.numel()} 个；"
                  f"o 的有限部分 max|·| = "
                  f"{float(o[~bad].abs().max()) if (~bad).any() else 'n/a'}")
    except Exception as e:
        print(f"  {label} -> 抛异常 {type(e).__name__}: {str(e)[:110]}")
print()

# ---------------------------------------------------------------------------
# [2] 真层连续多步 decode：漂移与爆值检查
# ---------------------------------------------------------------------------
print("=" * 74)
print("[2] 真层连续 24 步 decode：fla vs torch 回退的漂移")
print("=" * 74)
layer = Q5.Qwen3_5GatedDeltaNet(tc, 0).to(DEV, DT).eval()
torch.manual_seed(7)
x_pre = torch.randn(1, 96, tc.hidden_size, device=DEV, dtype=DT)
steps = [torch.randn(1, 1, tc.hidden_size, device=DEV, dtype=DT) for _ in range(24)]


def run(recurrent_impl, chunk_impl, tag):
    layer.recurrent_gated_delta_rule = recurrent_impl
    layer.chunk_gated_delta_rule = chunk_impl
    c = Q5.DynamicCache(config=cfg)
    outs = []
    with torch.no_grad():
        layer(x_pre, cache_params=c)
        for h in steps:
            outs.append(layer(h, cache_params=c).float().clone())
    return outs


outs_fla = run(Q5.fused_recurrent_gated_delta_rule, Q5.chunk_gated_delta_rule, "fla")
outs_ref = run(Q5.torch_recurrent_gated_delta_rule, Q5.torch_chunk_gated_delta_rule, "torch")

print("  step |  参考 max|out| |  fla max|out|  |  相对差  | fla finite")
worst = 0.0
for i, (a, b) in enumerate(zip(outs_ref, outs_fla), 1):
    ref = float(a.abs().max())
    got = float(b.abs().max())
    fin = bool(torch.isfinite(b).all())
    rel = float((a - b).abs().max() / max(ref, 1e-12))
    worst = max(worst, rel)
    mark = "" if i % 6 == 0 or i <= 2 else ""
    if i <= 3 or i % 6 == 0:
        print(f"  {i:4d} | {ref:14.4f} | {got:14.4f} | {rel:8.2e} | {fin}")
print(f"  >>> 24 步里最大的相对偏差 = {worst:.3e}")
print(f"  >>> {'未见漂移/爆值，可接受' if worst < 5e-2 else '偏差偏大，需谨慎'}")
print()
print("=" * 74)
