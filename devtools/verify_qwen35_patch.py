# -*- coding: utf-8 -*-
"""验证 patch_embed 补丁对 Qwen3.5 是否真的生效（回归 Qwen3.5 prefill 75.61s）。

背景
----
补丁 v1 只按类名给 `Qwen3VLVisionPatchEmbed.forward` 打补丁。
换成 Qwen3.5 后模型用的是 `Qwen3_5VisionPatchEmbed` —— 类不同、模块不同，
补丁静默失效，但类级标志 `_PATCH_EMBED_PATCHED` 仍是 True，
于是日志照报「已换成 GEMM」，prefill 却退回 35.1 ms/视觉token。

本脚本分三层验证补丁 v2：
  A. 无 GPU：两个模型族的 patch_embed 结构一致、实例级扫描能认出来
  B. 无 GPU：模拟「只补了 Qwen3-VL 的类」的旧行为，证明 Qwen3.5 漏网；
     再走实例级替换，证明能救回来
  C. 有 GPU：用真实形状（8624 patch / 1536 -> 1152）对照 Conv3d 与 GEMM 的
     bf16 耗时 —— 每个 case 独立子进程 + 硬超时，否则一档卡死会拖垮整轮

用法:  python -u verify_qwen35_patch.py
"""
import os
import sys
import types
import subprocess
import textwrap

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
sys.path.insert(0, NODE_DIR)

# nodes.py 会 import folder_paths（ComfyUI 的内置模块）。这里不启 ComfyUI，
# 所以塞一个最小替身进去，只补它真正用到的属性。
_fp = types.ModuleType("folder_paths")
_fp.models_dir = os.path.join(CV, "models")
_fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = _fp

import torch as _t

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


PY = sys.executable

# 真实场景：2 张参考图 906x1280 + 896x1184，patch=16/temporal=2
#   906//16 * 1280//16 = 56*80  = 4480
#   896//16 * 1184//16 = 56*74  = 4144
#   合计 8624 patch；经 2x2 merge 得 8624/4 = 2156 视觉 token（与用户日志一致）
N_PATCH = 8624
IN_FLAT = 3 * 2 * 16 * 16          # 1536
EMBED = 1152

print("=" * 74)
print("A) 两个模型族的 patch_embed 结构是否一致")
print("=" * 74)
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
    HAVE_Q3 = True
except ImportError:
    HAVE_Q3 = False
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed
    HAVE_Q35 = True
except ImportError:
    HAVE_Q35 = False

print(f"  Qwen3-VL 实现可用 : {HAVE_Q3}")
print(f"  Qwen3.5  实现可用 : {HAVE_Q35}")
check("transformers 里 Qwen3.5 的实现存在", HAVE_Q35)

if HAVE_Q35:

    class _Cfg:
        patch_size = 16
        temporal_patch_size = 2
        in_channels = 3
        hidden_size = EMBED

    pe35 = Qwen3_5VisionPatchEmbed(_Cfg())
    pe35.eval()

    check("Qwen3.5 patch_embed 内部也是 Conv3d",
          isinstance(pe35.proj, _t.nn.Conv3d), type(pe35.proj).__name__)
    check("Qwen3.5 的 kernel 与 stride 相同（替换的前提）",
          tuple(pe35.proj.kernel_size) == tuple(pe35.proj.stride),
          f"{tuple(pe35.proj.kernel_size)} / {tuple(pe35.proj.stride)}")
    check("Qwen3.5 的 padding=0 / dilation=1 / groups=1（等价性的另外三个前提）",
          tuple(pe35.proj.padding) == (0, 0, 0)
          and tuple(pe35.proj.dilation) == (1, 1, 1)
          and pe35.proj.groups == 1)

    if HAVE_Q3:
        class _Cfg3(_Cfg):
            pass
        pe3 = Qwen3VLVisionPatchEmbed(_Cfg3())
        check("两个模型族的 Conv3d 参数形状完全一致",
              tuple(pe35.proj.weight.shape) == tuple(pe3.proj.weight.shape),
              f"{tuple(pe35.proj.weight.shape)}")
    check("两个类是不同对象（所以按类名补丁会漏）",
          (not HAVE_Q3) or (Qwen3_5VisionPatchEmbed is not Qwen3VLVisionPatchEmbed))

print()
print("=" * 74)
print("B) 复现旧行为：只补 Qwen3-VL 的类，Qwen3.5 是否漏网；实例级能否救回")
print("=" * 74)
import nodes as QM

ORIG35 = Qwen3_5VisionPatchEmbed.forward

# --- 模拟 v1 的旧行为：只给 Qwen3-VL 的类换 forward ---
if HAVE_Q3:
    Qwen3VLVisionPatchEmbed.forward = QM._gemm_patch_embed_forward
check("旧做法（只补 Qwen3-VL 类）后，Qwen3.5 的类仍未被打补丁",
      Qwen3_5VisionPatchEmbed.forward is ORIG35)
check("而类级标志会误报成功 —— 这正是日志说「已换 GEMM」却依然慢的原因",
      (not HAVE_Q3) or (Qwen3VLVisionPatchEmbed.forward is QM._gemm_patch_embed_forward))

# --- v2：实例级扫描 ---
hits = QM._patch_patch_embed_instances(pe35)
check("实例级扫描认出了 Qwen3.5 的 patch_embed", hits == 1, f"替换 {hits} 个")
check("实例 forward 确实被换掉",
      pe35.forward.__func__ is QM._gemm_patch_embed_forward)
check("打上标记，便于幂等", getattr(pe35, "_qwen35_gemm_patch_embed", False) is True)
check("幂等：再扫一次计数不翻倍（返回该模型实例总数）",
      QM._patch_patch_embed_instances(pe35) == 1)
check("幂等：forward 仍指向同一个替换实现",
      pe35.forward.__func__ is QM._gemm_patch_embed_forward)

# 换模型后计数必须刷新，不能带着上一个模型累加（否则日志会虚高，
# 将来若用它做「是否生效」判断，就会重演类级标志误报的老问题）
QM._PATCH_EMBED_PATCHED = False
QM._PATCH_EMBED_INSTANCES = 0
QM._patch_vision_patch_embed(pe35)                 # 模型 A：1 个
pe35b = Qwen3_5VisionPatchEmbed(_Cfg()).eval()
QM._patch_vision_patch_embed(pe35b)                # 模型 B：也是 1 个
check("换模型后实例计数被刷新而不是累加",
      QM._PATCH_EMBED_INSTANCES == 1, f"{QM._PATCH_EMBED_INSTANCES}")
QM._patch_vision_patch_embed(pe35b)                # 同一模型再调一次
check("同一个模型重复调用计数保持稳定",
      QM._PATCH_EMBED_INSTANCES == 1, f"{QM._PATCH_EMBED_INSTANCES}")

# --- 数值等价（CPU float32，安全）---
pe35c = pe35.to("cpu").float()
flat = _t.randn(N_PATCH, IN_FLAT)
with _t.no_grad():
    ref = ORIG35(pe35c, flat)                       # 原始 Conv3d
    got = pe35c(flat)                               # 已被替换成 GEMM
err = float((ref - got).abs().max())
print(f"  Conv3d vs GEMM 最大绝对误差 = {err:.3e} （真实形状 {N_PATCH}x{IN_FLAT}->{EMBED}）")
check("真实形状下数值等价（误差 < 1e-4）", err < 1e-4, f"{err:.3e}")
check("输出形状一致", tuple(got.shape) == tuple(ref.shape), str(tuple(got.shape)))
check("权重未被复制（仍是同一块内存）",
      pe35c.proj.weight.reshape(pe35.proj.out_channels, -1).data_ptr()
      == pe35c.proj.weight.data_ptr())

# --- 维度不匹配时必须报错，而不是静默算错 ---
try:
    with _t.no_grad():
        pe35c(_t.randn(4, IN_FLAT + 16))
    check("输入布局不对时报错而不是静默出错", False, "竟然没报错")
except RuntimeError as e:
    check("输入布局不对时报错而不是静默出错", "维度不匹配" in str(e))

# --- 结构守卫：不该被误替换的模块 ---
class _ConvHolder(_t.nn.Module):
    """类名不含 patchembed —— 用来验证「名字守卫」。"""
    def __init__(self, conv):
        super().__init__()
        self.proj = conv


class _PatchEmbedLike(_t.nn.Module):
    """类名含 patchembed 但卷积不够格 —— 用来验证「结构守卫」。"""
    def __init__(self, conv):
        super().__init__()
        self.proj = conv


_conv_nonstride = _t.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(1, 16, 16))
_conv_padded = _t.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16),
                            padding=(1, 0, 0))
_conv_grouped = _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(2, 16, 16),
                             groups=3)
_conv_ok = _t.nn.Conv3d(3, 8, kernel_size=(2, 16, 16), stride=(2, 16, 16))

check("kernel != stride 的 Conv3d 判为不等价", QM._is_gemm_equivalent_conv3d(_conv_nonstride) is False)
check("带 padding 的 Conv3d 判为不等价", QM._is_gemm_equivalent_conv3d(_conv_padded) is False)
check("groups != 1 的 Conv3d 判为不等价", QM._is_gemm_equivalent_conv3d(_conv_grouped) is False)
check("kernel==stride、padding=0、groups=1 的 Conv3d 判为等价",
      QM._is_gemm_equivalent_conv3d(_conv_ok) is True)
check("非 Conv3d 一律不等价", QM._is_gemm_equivalent_conv3d(_t.nn.Linear(4, 4)) is False)
check("结构守卫：名字像 patch_embed 但卷积不够格的模块不会被替换",
      QM._patch_patch_embed_instances(_PatchEmbedLike(_conv_nonstride)) == 0)
check("名字守卫：类名不含 patchembed 的模块一律不碰",
      QM._patch_patch_embed_instances(_ConvHolder(_conv_ok)) == 0)

print()
print("=" * 74)
print("C) GPU 实测：真实形状下 Conv3d bf16 vs GEMM bf16")
print("=" * 74)

_TEMPLATE = textwrap.dedent("""
    import sys, os, time, types, torch
    _cv = r"{cv}"
    _fp = types.ModuleType("folder_paths")
    _fp.models_dir = os.path.join(_cv, "models")
    _fp.get_filename_list = lambda *a, **k: []
    sys.modules["folder_paths"] = _fp
    sys.path.insert(0, os.path.join(_cv, "custom_nodes", "ComfyUI-Qwen35-Enhancer"))
    import nodes as QM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed
    class _Cfg:
        patch_size = 16; temporal_patch_size = 2; in_channels = 3; hidden_size = {embed}
    torch.manual_seed(0)
    pe = Qwen3_5VisionPatchEmbed(_Cfg()).to("cuda").to(torch.bfloat16).eval()
    x = torch.randn({n}, {flat}, device="cuda", dtype=torch.bfloat16)
    mode = sys.argv[1]
    if mode == "gemm":
        QM._patch_patch_embed_instances(pe)
    with torch.no_grad():
        for _ in range(2):                       # 预热一次
            pe(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            pe(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
    print(f"RESULT {{dt*1000:.3f}}")
""").format(cv=CV, embed=EMBED, n=N_PATCH, flat=IN_FLAT)

_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_patch_bench.py")
with open(_script, "w", encoding="utf-8") as f:
    f.write(_TEMPLATE)

TIMEOUT = 60
results = {}
for mode in ("conv", "gemm"):
    try:
        p = subprocess.run([PY, "-u", _script, mode], capture_output=True,
                           text=True, timeout=TIMEOUT)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if line:
            results[mode] = float(line[0].split()[1])
            print(f"  {mode:5s}: {results[mode]:>10.3f} ms")
        else:
            print(f"  {mode:5s}: 无输出 rc={p.returncode} {p.stderr.strip()[-200:]}")
    except subprocess.TimeoutExpired:
        results[mode] = None
        print(f"  {mode:5s}: 超过 {TIMEOUT}s 没返回（挂死）")

if "conv" in results and "gemm" in results:
    if results["conv"] is None and results["gemm"] is not None:
        check("Conv3d bf16 挂死而 GEMM 正常 —— 病态路径复现在 Qwen3.5 上",
              True, f"conv=挂死 gemm={results['gemm']:.2f}ms")
    elif results["conv"] is not None and results["gemm"] is not None:
        ratio = results["conv"] / max(results["gemm"], 1e-6)
        print(f"  倍数 = {ratio:.1f}x")
        check("GEMM 不慢于 Conv3d（本机预期差距极大或 conv 挂死）",
              results["gemm"] <= results["conv"], f"{ratio:.1f}x")
else:
    print("  （GPU 档未取到完整数据，可能显存被占；不影响 A/B 两层的结论）")

try:
    os.remove(_script)
except OSError:
    pass

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("全部通过")
