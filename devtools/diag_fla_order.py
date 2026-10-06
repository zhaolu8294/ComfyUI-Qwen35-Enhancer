"""锁定 fla fused_recurrent 的 NaN 触发条件：是否为调用顺序 / dtype 污染。

同一个进程里按顺序打点，观察哪一步开始出 NaN。
"""
import torch

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\text_encoders\huihui-ai_Huihui-Qwen3.5-9B-abliterated"
DEV = "cuda"
DT = torch.bfloat16

import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(MODEL_DIR)
tc = cfg.text_config
H, K, V = tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim

torch.manual_seed(0)
q = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
k = torch.randn(1, 1, H, K, device=DEV, dtype=DT)
v = torch.randn(1, 1, H, V, device=DEV, dtype=DT)
g = -torch.rand(1, 1, H, device=DEV, dtype=DT) * 0.1
beta = torch.rand(1, 1, H, device=DEV, dtype=DT)


def fla(st):
    with torch.no_grad():
        o, s = Q5.fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=st.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True)
    return o, s


def tor(st):
    with torch.no_grad():
        o, s = Q5.torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, st, True, use_qk_l2norm_in_kernel=True)
    return o, s


def rep(tag, o, s):
    fo, fs = bool(torch.isfinite(o).all()), bool(torch.isfinite(s).all())
    print(f"  {tag:52s} o_finite={fo!s:5s} s_finite={fs!s:5s} "
          f"o_max={float(o[torch.isfinite(o)].abs().max()) if fo else 'nan'}")
    return fo


zero32 = torch.zeros(1, H, K, V, device=DEV, dtype=torch.float32)
zero16 = torch.zeros(1, H, K, V, device=DEV, dtype=DT)
rnd16 = torch.randn(1, H, K, V, device=DEV, dtype=DT)

print("=" * 100)
print("按顺序打点（同一个进程）")
print("=" * 100)
o, s = fla(zero32); rep("A. fla, fp32 零 state（进程里第一次调用）", o, s)
o, s = tor(zero32); rep("B. torch, fp32 零 state", o, s)
o, s = fla(zero32); rep("C. fla, fp32 零 state（torch 之后）", o, s)
o, s = fla(zero16); rep("D. fla, bf16 零 state", o, s)
o, s = fla(zero32); rep("E. fla, fp32 零 state（bf16 之后）", o, s)
o, s = fla(rnd16);  rep("F. fla, bf16 随机 state", o, s)
o, s = tor(zero16); rep("G. torch, bf16 零 state", o, s)
o, s = fla(zero16); rep("H. fla, bf16 零 state（再跑一次）", o, s)
print("=" * 100)
