# -*- coding: utf-8 -*-
"""
ComfyUI-Qwen35-Enhancer 进度条改造校验（v4）

校验项：
  1. INPUT_TYPES 结构：18 个 widget、新参数在末位、hidden 声明了 unique_id
  2. enhance() 签名与 INPUT_TYPES 一致（hidden 参数单独放行）
  3. _ProgressReporter 真把进度推给了 comfy.utils.ProgressBar，且带正确 node_id
  4. 进度单调递增、终点为 100
  5. StoppingCriteria 逐 token 计数正确
  6. 中断标志被真正转成异常抛出
  7. 终端不支持方块字符时能回落 ASCII

全程用 stub 替换 comfy.*，不加载任何真实模型、不碰显卡。
"""
import os
import sys
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- stubs
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp

PBAR_CALLS = []


class FakeProgressBar:
    """记录每一次 update_absolute，用来证明进度真的推给了前端。"""

    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id
        PBAR_CALLS.append(("init", total, node_id))

    def update_absolute(self, value, total=None):
        PBAR_CALLS.append(("update", value, total))

    def update(self, value):
        PBAR_CALLS.append(("update_rel", value, None))


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = FakeProgressBar
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu

# 可翻转的中断标志，模拟前端点"取消"
INTERRUPT = {"flag": False}
mm = types.ModuleType("comfy.model_management")


class InterruptProcessingException(BaseException):
    pass


def _throw_if_interrupted():
    if INTERRUPT["flag"]:
        raise InterruptProcessingException("interrupted")


mm.InterruptProcessingException = InterruptProcessingException
mm.throw_exception_if_processing_interrupted = _throw_if_interrupted
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import nodes as QM  # noqa: E402

print("=" * 74)
print("1) INPUT_TYPES 结构")
print("=" * 74)

it = QM.Qwen35PromptEnhancer.INPUT_TYPES()
req = list(it["required"].keys())
opt = list(it["optional"].keys())
hidden = it.get("hidden", {})

LINK_ONLY = {"image", "image_2", "image_3", "image_4"}   # 纯连线输入，不占 widget
widgets = [k for k in req + opt if k not in LINK_ONLY]

print(f"  required ({len(req)}): {req}")
print(f"  optional ({len(opt)}): {opt}")
print(f"  hidden            : {hidden}")
print(f"  widgets ({len(widgets)}): {widgets}")
print()

check("widget 总数为 18", len(widgets) == 18, f"实际 {len(widgets)}")
check("末两位是 progress_interval / bilingual",
      widgets[-2:] == ["progress_interval", "bilingual"], str(widgets[-2:]))
check("原 15 项顺序未被改动",
      widgets[:15] == ["model_name", "system_prompt", "user_prompt", "quantization",
                       "attention", "enable_thinking", "mode", "max_images",
                       "keep_model_loaded", "unload_other_models", "temperature",
                       "max_new_tokens", "seed", "custom_model_path", "max_image_side"])
check("hidden 声明了 unique_id -> UNIQUE_ID", hidden.get("unique_id") == "UNIQUE_ID", str(hidden))
check("show_progress 默认 True", it["optional"]["show_progress"][1].get("default") is True)
check("progress_interval 默认 2.0", it["optional"]["progress_interval"][1].get("default") == 2.0)

print()
print("=" * 74)
print("2) enhance() 签名一致性")
print("=" * 74)

import inspect  # noqa: E402
sig = list(inspect.signature(QM.Qwen35PromptEnhancer().enhance).parameters)
# self 去掉；自带的默认参数
params = [p for p in sig if p != "self"]
declared = [p for p in req + opt]
missing = [k for k in declared if k not in params]
extra = [p for p in params if p not in declared and p not in hidden]
print(f"  enhance 参数 ({len(params)}): {params}")
check("INPUT_TYPES 声明的参数全部在函数签名里", not missing, str(missing))
check("未声明多余参数（hidden 除外）", not extra, str(extra))
check("unique_id 出现在签名里", "unique_id" in params)
check("unique_id 有默认值（不传也不报错）",
      inspect.signature(QM.Qwen35PromptEnhancer().enhance).parameters["unique_id"].default is None)

print()
print("=" * 74)
print("3) 进度真的推给了前端 ProgressBar")
print("=" * 74)

PBAR_CALLS.clear()
rep = QM._ProgressReporter(node_id="42", enabled=True, interval=999)
check("ProgressBar 被构造", PBAR_CALLS and PBAR_CALLS[0][0] == "init", str(PBAR_CALLS[:1]))
check("node_id 透传为 '42'", PBAR_CALLS[0][2] == "42", f"实际 {PBAR_CALLS[0][2]!r}")

rep.mark(QM._W_UNLOAD)
rep.mark(QM._W_UNLOAD + QM._W_LOAD)
rep.begin_stage(QM._W_UNLOAD + QM._W_LOAD + QM._W_PREP, QM._W_GEN)
for n in range(1, 51):
    rep.tick(n, 50)
rep.finish()

pushed = [c[1] for c in PBAR_CALLS if c[0] == "update"]
print(f"  推送次数: {len(pushed)}  序列: {pushed[:6]} ... {pushed[-4:]}")
check("推送了多次进度", len(pushed) >= 6, f"{len(pushed)} 次")
check("进度单调不减", all(b >= a for a, b in zip(pushed, pushed[1:])))
check("起点为卸载阶段的值", pushed[0] == int(QM._W_UNLOAD), str(pushed[0]))
check("终点为 100", pushed[-1] == 100, str(pushed[-1]))
check("最大值不超过 100", max(pushed) <= 100, str(max(pushed)))
check("total 恒为 100", all(c[2] == 100 for c in PBAR_CALLS if c[0] == "update"))

print()
print("=" * 74)
print("4) 关闭进度时零开销")
print("=" * 74)
PBAR_CALLS.clear()
rep_off = QM._ProgressReporter(node_id="7", enabled=False)
check("enabled=False 时不构造 ProgressBar", len(PBAR_CALLS) == 0, str(PBAR_CALLS))
rep_off.mark(50)
rep_off.tick(1, 10)
rep_off.message("不应出现")
rep_off.finish()
check("enabled=False 时全程不推送", len(PBAR_CALLS) == 0, str(PBAR_CALLS))
check("enabled=False 时仍不报错", True)

print()
print("=" * 74)
print("5) StoppingCriteria 逐 token 计数")
print("=" * 74)
PBAR_CALLS.clear()
rep2 = QM._ProgressReporter(node_id="1", enabled=True, interval=999)
crit = QM._make_progress_criteria(rep2, 64)
check("criteria 构造成功", crit is not None)
if crit is not None:
    for _ in range(10):
        r = crit(input_ids=None, scores=None)
    check("10 次回调计数为 10", crit.n == 10, f"实际 {crit.n}")
    check("回调返回 False（不提前停止）", r is False, repr(r))
    pushed2 = [c[1] for c in PBAR_CALLS if c[0] == "update"]
    check("每次回调都推进了进度", len(pushed2) == 10, f"{len(pushed2)} 次")

print()
print("=" * 74)
print("6) 中断标志能真正打断生成")
print("=" * 74)
rep3 = QM._ProgressReporter(node_id="1", enabled=False)
crit3 = QM._make_progress_criteria(rep3, 64)
if crit3 is not None:
    INTERRUPT["flag"] = False
    ok_no_raise = crit3(input_ids=None, scores=None) is False
    check("未中断时不抛异常", ok_no_raise)
    INTERRUPT["flag"] = True
    raised = False
    try:
        crit3(input_ids=None, scores=None)
    except InterruptProcessingException:
        raised = True
    except BaseException as e:
        raised = f"抛了别的异常 {type(e).__name__}"
    check("中断时抛出 InterruptProcessingException", raised is True, str(raised))
    INTERRUPT["flag"] = False

print()
print("=" * 74)
print("7) 控制台进度渲染 + ASCII 回落")
print("=" * 74)
r = QM._ProgressReporter(node_id=None, enabled=True, interval=999)
line = r._render(128, 512, "输入 1148 tok")
print("  unicode:", line)
check("渲染包含百分比", "%" in line)
check("渲染包含已生成/总量 token", "128/512" in line)
check("渲染包含速度", "tok/s" in line)
check("渲染包含备注", "输入 1148 tok" in line)

QM._UNICODE_BAR = False
ascii_line = r._render(256, 512)
print("  ascii  :", ascii_line)
check("ASCII 回落不包含方块字符", "█" not in ascii_line and "#" in ascii_line, ascii_line)
QM._UNICODE_BAR = True

bar_full = QM._ProgressReporter._bar(1.0)
bar_empty = QM._ProgressReporter._bar(0.0)
check("0% 全为空槽", bar_empty.count("░") == 22, str(bar_empty))
check("100% 全为实心", bar_full.count("█") == 22, str(bar_full))
check("越界值被裁剪", QM._ProgressReporter._bar(5.0).count("█") == 22)

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("全部校验通过")
