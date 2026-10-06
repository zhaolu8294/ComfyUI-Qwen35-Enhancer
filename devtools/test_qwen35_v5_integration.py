# -*- coding: utf-8 -*-
"""
ComfyUI-Qwen35-Enhancer 进度条集成测试（v5）

用「假模型 / 假 processor」把 enhance() 完整跑一遍，验证进度接线：
  - 四个阶段都设了进度点，且进度单调递增到 100
  - 生成阶段的回调次数 == max_new_tokens
  - 阶段提示日志按顺序出现
  - 生成中途触发中断时，异常能正确抛出且进度条被收尾
全程不加载模型、不碰显卡。
"""
import os
import sys
import types

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")

FAIL = []
LOGS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- stubs
fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(CV, "models")
fp.get_filename_list = lambda *a, **k: []
sys.modules["folder_paths"] = fp

PUSH = []


class FakeProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id

    def update_absolute(self, value, total=None):
        PUSH.append((value, total))


cu = types.ModuleType("comfy.utils")
cu.ProgressBar = FakeProgressBar
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu


class InterruptProcessingException(BaseException):
    pass


INTERRUPT = {"flag": False, "after": 10**9, "seen": 0}
mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda *a, **k: None


def _throw_if_interrupted():
    if INTERRUPT["flag"]:
        raise InterruptProcessingException()


mm.throw_exception_if_processing_interrupted = _throw_if_interrupted
mm.InterruptProcessingException = InterruptProcessingException
comfy.model_management = mm
sys.modules["comfy.model_management"] = mm

sys.path.insert(0, NODE_DIR)
import nodes as QM  # noqa: E402

# 收集日志，用来断言阶段提示顺序
import logging  # noqa: E402


class Cap(logging.Handler):
    def emit(self, record):
        LOGS.append(record.getMessage())


QM.logger.addHandler(Cap())
QM.logger.setLevel(logging.INFO)


# ---------------------------------------------------------------- 假模型
class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def apply_chat_template(self, messages, **kw):
        return FakeInputs({"input_ids": _Tiny((1, 12))})

    def decode(self, ids, skip_special_tokens=True):
        return ("integrated_multimodal_description: A woman walks.\n\n"
                "overall_soundscape: Footsteps.\n\n"
                "non_diegetic_music: N/A")


class _Tiny:
    def __init__(self, shape):
        self.shape = shape


class FakeModel:
    device = "cpu"

    def __init__(self, steps, interrupt_after=10**9, stop_at=None):
        self.steps = steps
        self.interrupt_after = interrupt_after
        self.stop_at = stop_at

    def generate(self, input_ids=None, stopping_criteria=None, max_new_tokens=1, **kw):
        criteria = (stopping_criteria or [None])[0]
        n = 0
        limit = max_new_tokens if self.stop_at is None else min(max_new_tokens, self.stop_at)
        for i in range(limit):
            n += 1
            if i + 1 == self.interrupt_after:
                INTERRUPT["flag"] = True
            if criteria is not None:
                if criteria(input_ids=_Tiny((1, 12)), scores=None):
                    break
        import torch
        return torch.zeros((1, 12 + n), dtype=torch.long)


def run(max_new_tokens=32, interrupt_after=10**9, show_progress=True, stop_at=None):
    PUSH.clear()
    LOGS.clear()
    INTERRUPT.update({"flag": False, "after": interrupt_after})
    node = QM.Qwen35PromptEnhancer()
    node._resolve_path = lambda *a, **k: r"E:\fake\model"
    node._load = lambda *a, **k: (FakeModel(max_new_tokens, interrupt_after, stop_at),
                                  FakeProcessor(), True)
    return node.enhance(
        model_name="FakeModel", system_prompt="sys", user_prompt="user",
        quantization="8bit", attention="sdpa",
        max_new_tokens=max_new_tokens, unique_id="99",
        show_progress=show_progress, progress_interval=999,
        bilingual="off",   # 本文件测单语进度接线；双语场景见 v6
    )


print("=" * 74)
print("A) 正常生成：进度接线")
print("=" * 74)
text, _ = run(max_new_tokens=32)
vals = [v for v, _ in PUSH]
print(f"  进度推送 {len(vals)} 次")
print(f"  序列: {vals[:8]} ... {vals[-5:]}")
check("返回值是三段式文本", "integrated_multimodal_description" in text)
check("进度推送了多次", len(vals) >= 8, f"{len(vals)} 次")
check("进度单调不减", all(b >= a for a, b in zip(vals, vals[1:])))
check("起点 = 卸载阶段权重 2", vals[0] == int(QM._W_UNLOAD), str(vals[0]))
check("终点 = 100", vals[-1] == 100, str(vals[-1]))
check("上限不超过 100", max(vals) <= 100)

# 生成阶段应该有正好 max_new_tokens 次 tick
gen_base = QM._W_UNLOAD + QM._W_LOAD + QM._W_PREP
gen_ticks = [v for v in vals if v > gen_base]
print(f"  生成阶段推送: {len(gen_ticks)} 次（32 次 tick + 1 次收尾）")
check("生成阶段推送 = tick 32 + 收尾 1", len(gen_ticks) == 33, f"{len(gen_ticks)} 次")
check("首个生成进度 = 38 + 62/32 ≈ 40", gen_ticks[0] == 40, str(gen_ticks[0]))

# 阶段提示顺序
msgs = [m for m in LOGS if m.startswith("[Qwen35]")]
print("  阶段日志:")
for m in msgs:
    print("    ", m[:78])
check("有『释放显存』提示", any("释放显存" in m for m in msgs))
check("有『正在加载模型』提示", any("正在加载模型" in m for m in msgs))
check("有『模型加载完成』提示", any("模型加载完成" in m for m in msgs))
check("有『开始生成』提示", any("开始生成" in m for m in msgs))
check("加载提示里带量化信息", any("量化=8bit" in m for m in msgs))
check("有耗时分解输出", any("耗时分解" in m for m in msgs))

print()
print("=" * 74)
print("B) 生成中途中断")
print("=" * 74)
raised = None
try:
    run(max_new_tokens=64, interrupt_after=10)
except InterruptProcessingException:
    raised = "InterruptProcessingException"
except BaseException as e:
    raised = f"{type(e).__name__}: {e}"
check("中断异常被正确抛出", raised == "InterruptProcessingException", str(raised))
vals2 = [v for v, _ in PUSH]
print(f"  中断前进度: {vals2[-4:]}")
check("中断前进度仍在推进", len(vals2) >= 4, f"{len(vals2)} 次")
check("中断时进度被收尾到 100", vals2 and vals2[-1] == 100, str(vals2[-1:] if vals2 else []))

print()
print("=" * 74)
print("C) show_progress=False")
print("=" * 74)
text3, _ = run(max_new_tokens=16, show_progress=False)
vals3 = [v for v, _ in PUSH]
print(f"  进度推送: {len(vals3)} 次")
check("关闭进度后零推送", len(vals3) == 0, f"{len(vals3)} 次")
check("关闭进度后仍能正常出结果", "integrated_multimodal_description" in text3)
check("关闭进度后阶段日志也不再刷",
      not any("开始生成" in m for m in LOGS if m.startswith("[Qwen35]")))

print()
print("=" * 74)
print("D) 生成提前结束（EOS 早停）时进度条能补满")
print("=" * 74)
text4, _ = run(max_new_tokens=32, stop_at=8)   # 只跑 8 步就停
vals4 = [v for v, _ in PUSH]
print(f"  进度序列尾部: {vals4[-5:]}")
check("早停时进度末值为 100", vals4 and vals4[-1] == 100, str(vals4[-1:] if vals4 else []))
check("早停时进度仍单调不减", all(b >= a for a, b in zip(vals4, vals4[1:])))
check("早停时输出仍为三段式", "integrated_multimodal_description" in text4)

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("全部通过")
