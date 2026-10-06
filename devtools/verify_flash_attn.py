# -*- coding: utf-8 -*-
"""flash-attn 安装后的完整验证。

分三层，每层失败都能独立定位：

  L1 包层   —— 装没装、版本号、ABI 对不对（import 会不会炸）
  L2 接口层 —— flash_attn_varlen_func 在不在；transformers 认不认
  L3 内核层 —— 真在 GPU 上跑一次 varlen 前向，跟参考实现比数值

L3 只用几十 MB 显存（序列长度 64、4 个 head、head_dim 64），
所以显卡被别的东西占着也能跑 —— 这点很重要，本机 GPU 常年满载。

用法：
    E:/AI/ComfyUI-aki-v3/python/python.exe verify_flash_attn.py
"""
import sys
import traceback

FAIL = []
WARN = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def warn(name, detail=""):
    print(f"  [WARN] {name}" + (f"  — {detail}" if detail else ""))
    WARN.append(name)


print("=" * 74)
print("L1  包层")
print("=" * 74)
try:
    import flash_attn
    ver = getattr(flash_attn, "__version__", "?")
    print(f"  flash_attn 包已导入，版本 {ver}")
    check("flash_attn 可导入", True)
    check("版本 >= 2.3.3（transformers 的硬要求）",
          ver != "?" and tuple(int(x) for x in ver.split(".")[:3]) >= (2, 3, 3), ver)
except Exception as e:
    check("flash_attn 可导入", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()
    print("\nL1 就失败了，后面两层无法继续。")
    sys.exit(1)

print()
print("=" * 74)
print("L2  接口层")
print("=" * 74)
varlen = None
try:
    from flash_attn import flash_attn_varlen_func
    varlen = flash_attn_varlen_func
    check("flash_attn_varlen_func 存在（Qwen3-VL 视觉塔就调它）", True)
except Exception as e:
    check("flash_attn_varlen_func 存在", False, f"{type(e).__name__}: {e}")

try:
    from flash_attn import flash_attn_func  # noqa: F401
    check("flash_attn_func 存在（LLM 路径用）", True)
except Exception as e:
    check("flash_attn_func 存在", False, f"{type(e).__name__}: {e}")

try:
    from flash_attn.bert_padding import pad_input, unpad_input  # noqa: F401
    check("bert_padding.pad_input / unpad_input 存在", True)
except Exception as e:
    warn("bert_padding 辅助函数不完整", f"{type(e).__name__}: {e}")

try:
    import transformers
    from transformers.utils import is_flash_attn_2_available
    ok_tf = is_flash_attn_2_available()
    check("transformers.is_flash_attn_2_available() == True", ok_tf,
          f"transformers {transformers.__version__}")
except Exception as e:
    check("transformers 认这个 flash-attn", False, f"{type(e).__name__}: {e}")

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as M
    check("Qwen3VLForConditionalGeneration._supports_flash_attn == True",
          bool(getattr(M, "_supports_flash_attn", False)))
except Exception as e:
    warn("读不到 Qwen3-VL 的 flash 支持声明", f"{type(e).__name__}: {e}")

print()
print("=" * 74)
print("L3  内核层（真机 GPU 冒烟测试）")
print("=" * 74)
if varlen is None:
    check("跳过 L3（L2 没拿到 varlen 接口）", False)
else:
    try:
        import torch

        if not torch.cuda.is_available():
            warn("没有可用 CUDA，跳过 L3")
        else:
            free, total = torch.cuda.mem_get_info()
            print(f"  当前可用显存 {free / 1024**3:.2f}GiB / {total / 1024**3:.2f}GiB")
            if free < 300 * 1024**2:
                warn("可用显存不足 300MiB，L3 可能失败", f"{free/1024**2:.0f}MiB")

            torch.manual_seed(0)
            dev, dt = "cuda", torch.bfloat16
            # 两条长度不等的序列，故意不对齐 —— 这正是 varlen 的用武之地
            lens = [40, 24]
            nh, hd = 4, 64
            q = torch.randn(sum(lens), nh, hd, device=dev, dtype=dt)
            k = torch.randn(sum(lens), nh, hd, device=dev, dtype=dt)
            v = torch.randn(sum(lens), nh, hd, device=dev, dtype=dt)
            cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                              device=dev, dtype=torch.int32)
            mx = max(lens)

            out = varlen(q, k, v, cu, cu, mx, mx)
            torch.cuda.synchronize()
            check("flash_attn_varlen_func 真机跑通", True,
                  f"输出 shape {tuple(out.shape)} dtype {out.dtype}")

            check("输出形状与输入一致", tuple(out.shape) == tuple(q.shape),
                  f"{tuple(out.shape)} vs {tuple(q.shape)}")
            check("输出没有 NaN/Inf",
                  bool(torch.isfinite(out).all()),
                  f"非有限元素 {int((~torch.isfinite(out)).sum())} 个")

            # 与 PyTorch 参考实现比数值（用 float32 复算，避免 bf16 噪声误判）
            ref_parts = []
            for i, L in enumerate(lens):
                s = int(cu[i])
                qq = q[s:s + L].transpose(0, 1).unsqueeze(0).float()   # (1, h, L, d)
                kk = k[s:s + L].transpose(0, 1).unsqueeze(0).float()
                vv = v[s:s + L].transpose(0, 1).unsqueeze(0).float()
                scale = hd ** -0.5
                att = (qq @ kk.transpose(-1, -2)) * scale
                att = torch.softmax(att, dim=-1)
                ref_parts.append((att @ vv).squeeze(0).transpose(0, 1))  # (L, h, d)
            ref = torch.cat(ref_parts, 0)
            diff = (out.float() - ref).abs()
            rel = (diff.max() / ref.abs().max()).item()
            print(f"  数值对照：最大绝对误差 {diff.max().item():.4f}，"
                  f"相对 {rel:.4f}（bf16 下 < 0.05 视为一致）")
            check("数值与参考实现一致", rel < 0.05, f"相对误差 {rel:.4f}")

            del q, k, v, out, ref
            torch.cuda.empty_cache()
    except Exception as e:
        check("L3 真机内核测试", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题：")
    for f in FAIL:
        print("   -", f)
    if WARN:
        print(f"另有 {len(WARN)} 项提醒：")
        for w in WARN:
            print("   -", w)
    sys.exit(1)
print("全部通过" + (f"（{len(WARN)} 项提醒）" if WARN else ""))
for w in WARN:
    print("   提醒:", w)
