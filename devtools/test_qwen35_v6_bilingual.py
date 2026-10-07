# -*- coding: utf-8 -*-
"""
ComfyUI-Qwen35-Enhancer 双语预览集成测试（v6）

用「假模型 / 假 processor」把 enhance() 完整跑一遍，验证两阶段接线：
  - bilingual=en_then_zh 时确实产生两次 generate，第二次是纯文本（不带图）
  - 翻译阶段温度被压到 0.3
  - 两个阶段各自都有 token 级进度，整体单调递增到 100
  - bilingual=off 时只跑一次，第二个输出为空串
  - 中断在「翻译阶段」同样生效
  - 中文标签被翻错时能还原成英文
  - widgets 顺序与 enhance 签名一致（bilingual 在末位，旧工作流不会错位）
全程不加载模型、不碰显卡。
"""
import inspect
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


INTERRUPT = {"flag": False}
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

import logging  # noqa: E402


class Cap(logging.Handler):
    def emit(self, record):
        LOGS.append(record.getMessage())


QM.logger.addHandler(Cap())
QM.logger.setLevel(logging.INFO)


# ---------------------------------------------------------------- 假件
class _Tiny:
    def __init__(self, shape):
        self.shape = shape


class FakeInputs(dict):
    def to(self, device):
        return self


EN_TEXT = ("integrated_multimodal_description: [Shot 1] Cinematic, a medium-wide shot frames "
           "a woman walking through rain.\n\n"
           "overall_soundscape: Rain hits the pavement and her footsteps splash.\n\n"
           "non_diegetic_music: N/A")

# 故意用中文标签 —— 用来验证 _restore_zh_labels 的还原能力
ZH_RAW = ("综合多模态描述（integrated_multimodal_description）：[Shot 1] 电影感画面，"
          "中景镜头框住一位在雨中行走的女子。\n\n"
          "整体声音环境：雨水打在路面上，脚步声溅起水花。\n\n"
          "非叙事音乐：N/A")


class FakeProcessor:
    def __init__(self):
        self.tmpl_calls = 0
        self.decode_calls = 0
        self.messages_seen = []

    def apply_chat_template(self, messages, **kw):
        self.tmpl_calls += 1
        self.messages_seen.append(messages)
        return FakeInputs({"input_ids": _Tiny((1, 20))})

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls += 1
        return EN_TEXT if self.decode_calls == 1 else ZH_RAW


class FakeModel:
    device = "cpu"

    def __init__(self, steps, interrupt_at=None, stop_at=None):
        self.steps = steps
        self.interrupt_at = interrupt_at      # (第几次 generate, 第几步)
        self.stop_at = stop_at
        self.calls = 0
        self.seen = []

    def generate(self, input_ids=None, stopping_criteria=None, max_new_tokens=1, **kw):
        self.calls += 1
        self.seen.append({
            "max_new_tokens": max_new_tokens,
            "temperature": kw.get("temperature"),
            "do_sample": kw.get("do_sample"),
            "eos_token_id": kw.get("eos_token_id"),
            "has_criteria": stopping_criteria is not None,
        })
        criteria = (stopping_criteria or [None])[0]
        n = 0
        limit = max_new_tokens if self.stop_at is None else min(max_new_tokens, self.stop_at)
        for i in range(limit):
            n += 1
            if self.interrupt_at and self.interrupt_at[0] == self.calls \
                    and i + 1 == self.interrupt_at[1]:
                INTERRUPT["flag"] = True
            if criteria is not None:
                if criteria(input_ids=_Tiny((1, 20)), scores=None):
                    break
        import torch
        return torch.zeros((1, 20 + n), dtype=torch.long)


def run(**over):
    PUSH.clear()
    LOGS.clear()
    INTERRUPT.update({"flag": False})
    node = QM.Qwen35PromptEnhancer()
    node._resolve_path = lambda *a, **k: r"E:\fake\model"
    model = FakeModel(over.get("max_new_tokens", 32),
                      over.get("interrupt_at"), over.get("stop_at"))
    proc = FakeProcessor()
    node._load = lambda *a, **k: (model, proc, True)
    out = node.enhance(
        model_name="FakeModel", system_prompt="sys", user_prompt="user",
        quantization="8bit", attention="sdpa",
        max_new_tokens=over.get("max_new_tokens", 32), unique_id="99",
        show_progress=over.get("show_progress", True), progress_interval=999,
        bilingual=over.get("bilingual", "en_then_zh"),
    )
    return out, model, proc


BASE_GEN = QM._W_UNLOAD + QM._W_LOAD + QM._W_PREP     # 38
PRIOR_SPAN = QM._W_GEN * QM._EN_SPAN_PRIOR            # 21.7 英文段先验宽度
# 英文段跑完后 pbar.rescale() 按实测重锚定；本例英 32 tok、中文预算也是 32 tok
# → 占比 0.5 → 分界点 = 38 + 62*0.5
BOUNDARY = BASE_GEN + QM._W_GEN * 0.5                 # 69.0

print("=" * 74)
print("A) bilingual=en_then_zh：两阶段接线")
print("=" * 74)
out, model, proc = run(max_new_tokens=32)
print(f"  generate 调用次数={model.calls}  apply_chat_template 次数={proc.tmpl_calls}")
print(f"  返回类型={type(out).__name__}  长度={len(out)}")
check("返回两个输出", isinstance(out, tuple) and len(out) == 2, str(len(out)))
en, zh = out
check("英文输出是三段式", "integrated_multimodal_description" in en)
check("英文输出无中文", not any("\u4e00" <= c <= "\u9fff" for c in en))
check("中文输出非空", len(zh) > 0, f"{len(zh)} 字符")
check("中文输出含中文", any("\u4e00" <= c <= "\u9fff" for c in zh))
check("跑到两次 generate", model.calls == 2, str(model.calls))
check("模板被调用两次", proc.tmpl_calls == 2, str(proc.tmpl_calls))
check("两次都有进度回调", all(s["has_criteria"] for s in model.seen),
      str([s["has_criteria"] for s in model.seen]))
check("英文轮保持采样解码", model.seen[0]["do_sample"] is True,
      str(model.seen[0]["do_sample"]))
check("英文轮温度保持 0.4", model.seen[0]["temperature"] == 0.4,
      str(model.seen[0]["temperature"]))
check("翻译轮改为贪心解码", model.seen[1]["do_sample"] is False,
      str(model.seen[1]["do_sample"]))
check("翻译轮不再传 temperature", model.seen[1]["temperature"] is None,
      str(model.seen[1]["temperature"]))
# 英文出了 32 tok → 预算 = min(max_new_tokens, max(96, 32*1.6+64)) = 32
check("翻译轮 max_new_tokens 受预算约束（本例 32）",
      model.seen[1]["max_new_tokens"] == 32, str(model.seen[1]["max_new_tokens"]))

# 第二次请求必须是纯文本
second = proc.messages_seen[1]
has_image = False
for m in second:
    c = m.get("content")
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image":
                has_image = True
check("翻译阶段不带图片", not has_image)
check("翻译阶段带 system prompt",
      any(m.get("role") == "system" for m in second))
check("翻译 system prompt 是预览专用",
      any("PREVIEW TRANSLATION" in str(m.get("content", "")) for m in second))
check("用户消息里嵌了英文提示词原文",
      any("integrated_multimodal_description" in str(m.get("content", "")) for m in second))

# 中文标签被还原
check("中文标签被还原成英文", zh.startswith("integrated_multimodal_description:"),
      zh[:46].replace("\n", " "))
check("第二段标签也被还原", "overall_soundscape:" in zh)
check("第三段标签也被还原", "non_diegetic_music:" in zh)

print()
vals = [v for v, _ in PUSH]
print(f"  进度推送 {len(vals)} 次")
print(f"  序列头部: {[round(v, 1) for v in vals[:6]]}")
print(f"  序列尾部: {[round(v, 1) for v in vals[-4:]]}")
en_ticks = [v for v in vals if BASE_GEN < v < BOUNDARY - 1e-9]
zh_ticks = [v for v in vals if v > BOUNDARY + 1e-9]
reanchors = [v for v in vals if abs(v - BOUNDARY) < 1e-9]
check("进度单调不减", all(b >= a for a, b in zip(vals, vals[1:])))
check("终点 = 100", vals[-1] == 100, str(vals[-1]))
check("上限不超过 100", max(vals) <= 100)
check("英文段先验宽度生效（tick = 32）", len(en_ticks) == 32, str(len(en_ticks)))
check("英文段只占先验宽度，不到分界点", en_ticks[-1] < BOUNDARY,
      f"{round(en_ticks[-1], 1)} < {round(BOUNDARY, 1)}")
check("跑完英文后按实测重锚定一次", len(reanchors) == 1,
      str([round(v, 1) for v in reanchors]))
# 中文阶段是最后一轮，末尾多一次 finish() 收尾推送 —— 生成提前 EOS 时靠它把条补到 100
check("中文阶段 tick = 32 + 收尾 1", len(zh_ticks) == 33, str(len(zh_ticks)))
check("两阶段间有分界（英文末值 < 中文首值）",
      en_ticks[-1] < zh_ticks[0], f"{round(en_ticks[-1],1)} < {round(zh_ticks[0],1)}")

msgs = [m for m in LOGS if m.startswith("[Qwen35]")]
print("  阶段日志:")
for m in msgs[:12]:
    print("    ", m[:74])
check("有『开始生成英文提示词』", any("开始生成英文提示词" in m for m in msgs))
check("有『英文提示词完成』", any("英文提示词完成" in m for m in msgs))
check("有『开始翻译中文预览』", any("开始翻译中文预览" in m for m in msgs))
check("有『中文预览完成』", any("中文预览完成" in m for m in msgs))
check("耗时分解含翻译行", any("翻译中文预览" in m for m in msgs))
check("耗时分解含英文行", any("生成英文提示词" in m for m in msgs))

print()
print("=" * 74)
print("B) bilingual=off：退回单语，行为不变")
print("=" * 74)
out2, model2, proc2 = run(max_new_tokens=16, bilingual="off")
print(f"  generate 调用次数={model2.calls}")
check("只跑一次 generate", model2.calls == 1, str(model2.calls))
check("模板只调一次", proc2.tmpl_calls == 1, str(proc2.tmpl_calls))
check("第二个输出为空串", out2[1] == "", repr(out2[1]))
check("英文输出正常", "integrated_multimodal_description" in out2[0])
vals2 = [v for v, _ in PUSH]
check("单语时进度仍到 100", vals2 and vals2[-1] == 100)
check("单语时生成段独占 62 点（38 -> 100）",
      vals2[0] == int(QM._W_UNLOAD) and len([v for v in vals2 if v > BASE_GEN]) == 17,
      f"生成段 tick {len([v for v in vals2 if v > BASE_GEN])} 次")
check("单语时不打翻译日志",
      not any("翻译中文预览" in m for m in LOGS if m.startswith("[Qwen35]")))
check("单语时耗时分解标注 bilingual=off",
      any("bilingual=off" in m for m in LOGS))

print()
print("=" * 74)
print("C) 中断发生在翻译阶段")
print("=" * 74)
raised = None
try:
    run(max_new_tokens=32, interrupt_at=(2, 10))   # 第 2 次 generate 的第 10 步
except InterruptProcessingException:
    raised = "InterruptProcessingException"
except BaseException as e:
    raised = f"{type(e).__name__}: {e}"
check("翻译阶段中断能抛出", raised == "InterruptProcessingException", str(raised))
vals3 = [v for v, _ in PUSH]
print(f"  中断前进度尾部: {[round(v,1) for v in vals3[-4:]]}")
check("中断时进度被收尾到 100", vals3 and vals3[-1] == 100,
      str(vals3[-1:] if vals3 else []))
check("中断发生在中文段（进度 > 分界点）", any(v > BOUNDARY for v in vals3),
      f"最大 {round(max(vals3),1) if vals3 else 0}")

print()
print("=" * 74)
print("D) _restore_zh_labels 标签还原")
print("=" * 74)
cases = [
    ("综合多模态描述：内容", "integrated_multimodal_description:内容"),
    ("综合多模态描述（integrated_multimodal_description）：内容",
     "integrated_multimodal_description:内容"),
    ("整体声音环境: 内容", "overall_soundscape: 内容"),
    ("非叙事音乐：N/A", "non_diegetic_music:N/A"),
    ("画外音乐：钢琴独奏", "non_diegetic_music:钢琴独奏"),
]
for src, want in cases:
    got = QM._restore_zh_labels(src)
    check(f"还原 {src[:12]}…", got == want, f"得到 {got!r}")
untouched = "integrated_multimodal_description: [Shot 1] A woman walks."
check("英文原文不受影响", QM._restore_zh_labels(untouched) == untouched)

print()
print("=" * 74)
print("E) widgets 顺序（防止旧工作流错位）")
print("=" * 74)
it = QM.Qwen35PromptEnhancer.INPUT_TYPES()
# backend 是**连线输入**（类型 QWEN35_BACKEND，由 Qwen35GGUFServer 节点连线提供），
# 它不占 widgets_values 的位置 —— 与打标节点同一约定（见 test_qwen35_tagger.py
# 的「连线输入不占 widgets_values 的位置」与「widgets 总数 = 32」两条）。
# 这里必须把它排除：否则会把"连线输入"误判成"新控件插在了 required 中间"，
# 得出"顺序错位"的错误结论（2026-10-07 踩到，backend 是当天新增的）。
_line_inputs = [k for k, v in it["optional"].items()
                if isinstance(v[0], str) and v[0] == "QWEN35_BACKEND"]
order = list(it["required"]) + [
    k for k in it["optional"]
    if k not in _line_inputs and not k.startswith("image")]
print("  widgets 顺序:")
for i, k in enumerate(order):
    print(f"    [{i:>2}] {k}")
check("连线输入只有 backend（不占 widgets 槽位）",
      _line_inputs == ["backend"], str(_line_inputs))
check("widgets 总数 = 18（backend 是连线输入，不计数）",
      len(order) == 18, str(len(order)))
check("bilingual 在末位", order[-1] == "bilingual", order[-1])
check("prompt_zh 是新输出且不挤占旧端口",
      QM.Qwen35PromptEnhancer.RETURN_NAMES == ("prompt", "prompt_zh"),
      str(QM.Qwen35PromptEnhancer.RETURN_NAMES))

sig = list(inspect.signature(QM.Qwen35PromptEnhancer.enhance).parameters.keys())
sig = [s for s in sig if s not in ("self", "unique_id")
       and not s.startswith("image") and s not in _line_inputs]
check("签名与 widgets 顺序一致", sig == order,
      f"差异 {set(sig) ^ set(order)}" if sig != order else "")
# 回退后双语默认关闭：默认路径下第二段根本不执行，速度与单语版一致
check("bilingual 的 INPUT_TYPES 默认 = off",
      it["optional"]["bilingual"][1]["default"] == "off",
      str(it["optional"]["bilingual"][1]["default"]))
check("enhance 签名默认 = off",
      inspect.signature(QM.Qwen35PromptEnhancer.enhance)
      .parameters["bilingual"].default == "off")
check("bilingual 的候选值仍是 off / en_then_zh",
      tuple(it["optional"]["bilingual"][0]) == ("off", "en_then_zh"),
      str(it["optional"]["bilingual"][0]))

print()
print("=" * 74)
print("F) 中文预算换算 _zh_budget（拦住『跑满上限』的空转）")
print("=" * 74)
b_cases = [
    (0, 1024, 1024, "英文为空 → 退回用户上限"),
    (32, 32, 32, "用户上限更小 → 绝不越界"),
    (10, 1024, 96, "极短英文 → 兜到最小预算 96"),
    (32, 1024, 115, "32*1.6+64 = 115"),
    (500, 1024, 864, "500*1.6+64 = 864"),
    (1024, 1024, 1024, "英文已触上限 → 中文不再放大"),
]
for en_tok, cap, want, note in b_cases:
    got = QM._zh_budget(en_tok, cap)
    check(f"_zh_budget({en_tok},{cap}) = {want}  [{note}]", got == want, f"得到 {got}")
check("预算永不超用户上限",
      all(QM._zh_budget(e, 300) <= 300 for e in (0, 50, 200, 1000)))
check("预算随英文长度单调不减",
      all(QM._zh_budget(a, 4096) <= QM._zh_budget(b, 4096)
          for a, b in ((10, 60), (60, 200), (200, 900))))

print()
print("=" * 74)
print("G) 长英文：预算放大 + 分界点按实测重算")
print("=" * 74)
out4, model4, _ = run(max_new_tokens=1024, stop_at=64)
print(f"  英文 out_len=64 -> 翻译轮 max_new_tokens={model4.seen[1]['max_new_tokens']}")
check("英文 64 tok 时中文预算 = 166（64*1.6+64）",
      model4.seen[1]["max_new_tokens"] == 166, str(model4.seen[1]["max_new_tokens"]))
check("中文预算大于英文 token（中文更吃 token）",
      model4.seen[1]["max_new_tokens"] > 64)
vals4 = [v for v, _ in PUSH]
check("长英文轮进度仍单调到 100",
      all(b >= a for a, b in zip(vals4, vals4[1:])) and vals4[-1] == 100,
      f"len={len(vals4)} 尾={round(vals4[-1], 1)}")

rep = QM._ProgressReporter(enabled=False)
rep.begin_stage(38.0, 21.7)
rep.tick(21, 21)
before = rep.value
rep.rescale(30.0, 60.0)          # 故意给一个比当前值小的分界
check("rescale 估错也不会让进度条回退", rep.value >= before, f"{round(before,2)} -> {round(rep.value,2)}")
rep.rescale(80.0, 20.0)
check("rescale 前进时正常跳转", rep.value == 80.0, str(rep.value))

print()
print("=" * 74)
print("H) 英文回声清理 _drop_english_echo")
print("=" * 74)
echo = QM._restore_zh_labels(EN_TEXT + "\n\n" + ZH_RAW)
got = QM._drop_english_echo(echo)
print(f"  回声长度 {len(echo)} -> 清理后 {len(got)}")
check("砍掉抄回来的英文，只剩中文段",
      got.startswith("integrated_multimodal_description:") and QM._has_cjk(got),
      got[:44].replace("\n", " "))
check("清理后不含英文原文", "walking through rain" not in got)
nature = "integrated_multimodal_description: 电影感画面\n\noverall_soundscape: 雨声"
check("正常输出（标签在 0 位）原样不动", QM._drop_english_echo(nature) == nature)
pure_en = "integrated_multimodal_description: A woman walks."
check("整段没中文时不猜（原样返回）", QM._drop_english_echo(pure_en) == pure_en)
check("回声清理在 enhance 主链路里生效（中文输出不含英文原文）",
      "walking through rain" not in zh)

print()
print("=" * 74)
print("I) EOS 补全 _eos_token_ids")
print("=" * 74)


class FakeTok:
    chat_template = None

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(t, 0)

    def convert_ids_to_tokens(self, i):
        return {151645: "<|im_end|>", 151643: "<|endoftext|>"}.get(i, "<unk>")


class FakeGenCfg:
    eos_token_id = [151645]


got_ids = QM._eos_token_ids(type("P", (), {"tokenizer": FakeTok()})(),
                            type("M", (), {"generation_config": FakeGenCfg})())
print(f"  并集得到的 EOS ids = {got_ids}")
check("并入 config 的 EOS 与分词器结束符", got_ids == [151645, 151643], str(got_ids))
check("未知 token 不会被误当 EOS（<|eot_id|> → id 0）", 0 not in got_ids, str(got_ids))
check("拿不到 EOS 时返回空列表（该参数不传）",
      QM._eos_token_ids(FakeProcessor(), FakeModel(1)) == [])
check("假件场景下 generate 没收到 eos_token_id（不引入噪音）",
      model.seen[0]["eos_token_id"] is None, str(model.seen[0]["eos_token_id"]))

node_e = QM.Qwen35PromptEnhancer()
node_e._resolve_path = lambda *a, **k: r"E:\fake\model"
m_e = FakeModel(16)
m_e.generation_config = FakeGenCfg()
p_e = FakeProcessor()
p_e.tokenizer = FakeTok()
node_e._load = lambda *a, **k: (m_e, p_e, True)
PUSH.clear()
LOGS.clear()
INTERRUPT.update({"flag": False})
node_e.enhance(model_name="F", system_prompt="sys", user_prompt="u", quantization="8bit",
               attention="sdpa", max_new_tokens=16, unique_id="9", show_progress=True,
               progress_interval=999, bilingual="off")
check("EOS 真的被传进 generate",
      m_e.seen[0]["eos_token_id"] == [151645, 151643], str(m_e.seen[0]["eos_token_id"]))
check("日志打印了 EOS ids",
      any("EOS ids" in m for m in LOGS if m.startswith("[Qwen35]")))

print()
print("=" * 74)
print("J) 提速诊断（显存/设备/速度体检）")
print("=" * 74)
dms = QM._device_map_summary(type("M", (), {
    "hf_device_map": {"model.embed": 0, "model.layers.0": 0, "model.layers.1": "cpu",
                      "visual": "cpu"}})())
print(f"  device_map 汇总 = {dms}")
check("device_map 汇总按设备计数", dms == {"cuda:0": 2, "cpu": 2}, str(dms))
check("没有 hf_device_map 时返回 None",
      QM._device_map_summary(type("M", (), {})()) is None)


class _P:
    def __init__(self, dev):
        self._d = dev

    def __iter__(self):
        return iter(())


class _VisHolder:
    def __init__(self, dev):
        import torch
        self.visual = torch.nn.Linear(2, 2).to(dev)


import torch as _t  # noqa: E402
check("能找出视觉塔在 cpu",
      QM._vision_device(_VisHolder("cpu")) == "cpu", str(QM._vision_device(_VisHolder("cpu"))))
check("找不到视觉塔时返回 None",
      QM._vision_device(object()) is None)

_no_mm = QM._probe_memory(None)
print(f"  mm=None 时探针返回 {tuple(round(x / 1024**3, 2) for x in _no_mm)} GiB")
check("拿不到 model_management 时可用显存/内存为 0",
      _no_mm[0] == 0 and _no_mm[2] == 0, str(_no_mm))
check("显存总量独立于 mm（torch 直接读得到）", _no_mm[1] >= 0, str(_no_mm[1]))


class _FakeMM:
    @staticmethod
    def get_free_memory(dev=None, torch_free_too=False):
        return ([7 * 1024 ** 3, 1 * 1024 ** 3] if torch_free_too
                else 7 * 1024 ** 3)


check("探针能从 get_free_memory 取到数值（int 与 tuple 两种返回都吃）",
      QM._probe_memory(_FakeMM())[0] == 7 * 1024 ** 3, str(QM._probe_memory(_FakeMM())[0]))
check("_fmt_gib 格式正确", QM._fmt_gib(2 * 1024 ** 3) == "2.0GiB", QM._fmt_gib(2 * 1024 ** 3))

# 速度体检：三层归因（CPU 摊派 → 量化档 → 原因未知），且正常与样本过小都不告警。
# 第 3 层是刻意留的：宁可承认不知道，也不能给一个听起来合理却指错方向的说法 ——
# 旧版把所有慢都归到「层被摊到 CPU」，而实测 8bit 慢是因为量化档本身，层全在 GPU 上。
check("正常速度（50 tok/s）不告警",
      QM._decode_speed_warning(500, 10.0) == "")
check("token 太少（<16）不下结论", QM._decode_speed_warning(5, 10.0) == "")
check("零耗时不下结论", QM._decode_speed_warning(100, 0.0) == "")
check("体检线设在合理区间（bf16 正常约 18~22 tok/s，不能高于它）",
      10.0 <= QM._SLOW_DECODE_TOK_S <= 16.0, str(QM._SLOW_DECODE_TOK_S))

_off_bak, _q_bak = QM._CPU_OFFLOAD_GIB, QM._CUR_QUANT
try:
    # 1) 真有层落在 CPU → 硬故障，优先修它
    QM._CPU_OFFLOAD_GIB, QM._CUR_QUANT = 3.2, "none"
    _w = QM._decode_speed_warning(171, 206.41)
    check("层落在 CPU → 归因到 CPU 摊派", "摊到 CPU" in _w, _w[:46])
    check("  并报出实际落在 CPU 的 GiB 数", "3.2GiB" in _w, _w[:58])

    # 2) 层全在 GPU 但用了 8bit → 归因到量化档（本机实测的主因）
    QM._CPU_OFFLOAD_GIB = 0.0
    _w = QM._decode_speed_warning(171, 206.41, "8bit")
    check("层全在 GPU + 8bit → 归因到量化档", "quantization=8bit" in _w, _w[:46])
    check("  给出 none/4bit/8bit 三档实测对照",
          "18.5~21.6" in _w and "17.3~19.7" in _w and "12.4~13.1" in _w, _w[:70])
    check("  指出 8bit 的另一半代价在「加载」而不是解码",
          "加载" in _w and "33.90s" in _w and "19.60s" in _w, _w[:90])
    check("  建议改用 4bit（受控实测 4bit 才划算）", "quantization=4bit" in _w, _w[:70])
    check("  4bit 建议带真机整轮数字（54.06 / 51.85）",
          "54.06s" in _w and "51.85s" in _w, _w[:90])
    check("  显存够则直接 none", "直接 none" in _w, _w[:70])

    # 2b) 4bit 也慢 → 不能拿 4bit 当解释（实测 4bit 的 MLP 比 bf16 还快）
    QM._CPU_OFFLOAD_GIB = 0.0
    _w = QM._decode_speed_warning(171, 206.41, "4bit")
    check("层全在 GPU + 4bit → 明说 4bit 解释不了", "解释不了" in _w, _w[:46])

    # 3) 两者都不是 → 如实说不知道，不硬编原因、也不让人去调无关参数
    QM._CUR_QUANT = "none"
    _w = QM._decode_speed_warning(171, 206.41)
    check("两者都不是 → 承认原因不在已排查范围内", "不在已排查的这几处" in _w, _w[:46])
    check("  并叫停「调 max_image_side」这个错动作",
          "max_image_side" in _w and "无关" in _w, _w[:70])

    # 显式传参优先于模块级状态（缓存切换模型时前者更可靠）
    check("显式 quantization 覆盖模块级 _CUR_QUANT",
          "quantization=4bit" in QM._decode_speed_warning(171, 206.41, "4bit"))
finally:
    QM._CPU_OFFLOAD_GIB, QM._CUR_QUANT = _off_bak, _q_bak

# 字节格式化自适应单位
check("_fmt_gib: GiB 档", QM._fmt_gib(2 * 1024 ** 3) == "2.0GiB", QM._fmt_gib(2 * 1024 ** 3))
check("_fmt_gib: MiB 档", QM._fmt_gib(5 * 1024 ** 2) == "5MiB", QM._fmt_gib(5 * 1024 ** 2))
check("_fmt_gib: KiB 档", QM._fmt_gib(2048) == "2KiB", QM._fmt_gib(2048))

# 参数级设备/精度统计 —— hf_device_map 为空时它是唯一的判定依据
_net = _t.nn.Module()
_net.w = _t.nn.Parameter(_t.zeros(1000, dtype=_t.float32))
_dev_b, _dt_b = QM._param_device_stats(_net)
check("参数统计: 设备分布", _dev_b == {"cpu": 4000}, str(_dev_b))
check("参数统计: 精度分布", _dt_b == {"float32": 4000}, str(_dt_b))
check("参数统计: 无参数返回 None", QM._param_device_stats(_t.nn.Module()) == (None, None))
check("参数统计: 可只统计子模块",
      QM._param_device_stats(_net, _net) == ({"cpu": 4000}, {"float32": 4000}))

# 视觉塔定位（设备体检依赖它）
_vis = _t.nn.Module()
_vis.p = _t.nn.Parameter(_t.zeros(10))
_holder = _t.nn.Module()
_holder.visual = _vis
check("_vision_module 找到 visual", QM._vision_module(_holder) is _vis)
check("_vision_module 找不到返回 None", QM._vision_module(_t.nn.Module()) is None)
check("_vision_device 读到 cpu", QM._vision_device(_holder) == "cpu",
      str(QM._vision_device(_holder)))

# _fmt_stat 按体量倒序，大块在前
_stat = QM._fmt_stat({"cpu": 1024, "cuda:0": 2 * 1024 ** 3})
check("_fmt_stat 大体量排前", _stat.startswith("cuda:0") and "cpu" in _stat, _stat)

# 量化跳过：视觉塔必须被排除（4bit 下逐层反量化会把 prefill 拖到几十秒）
check("_QUANT_SKIP_MODULES 含视觉塔前缀",
      "model.visual" in QM._QUANT_SKIP_MODULES, str(QM._QUANT_SKIP_MODULES))
check("_QUANT_SKIP_MODULES 含 lm_head（补回被丢的默认保护）",
      "lm_head" in QM._QUANT_SKIP_MODULES, str(QM._QUANT_SKIP_MODULES))

import re as _re


def _should_convert(full_name, patterns):
    """复刻 transformers/quantizers/quantizers_utils.py:37 的判定规则。"""
    skip = any(
        _re.match(f"{k}\\.", full_name) or _re.match(f"{k}", full_name) or full_name.endswith(k)
        for k in patterns
    )
    return not skip


check("视觉塔层不会被量化",
      not _should_convert("model.visual.blocks.0.attn.qkv", QM._QUANT_SKIP_MODULES))
check("视觉塔 merger 不会被量化",
      not _should_convert("model.visual.merger.mlp.0", QM._QUANT_SKIP_MODULES))
check("lm_head 不会被量化",
      not _should_convert("lm_head", QM._QUANT_SKIP_MODULES))
check("LLM 层仍然会被量化（跳过规则没有误伤）",
      _should_convert("model.language_model.layers.0.self_attn.q_proj", QM._QUANT_SKIP_MODULES),
      "language_model 层被误跳过")

# 视觉 token 计数：把 prefill 秒数摊到 token 上的前提
_g = _t.tensor([[1, 40, 58], [1, 30, 35]])          # (t, h, w)，单位是 16px 的 patch 数
check("视觉 token 计数正确",
      QM._count_visual_tokens({"image_grid_thw": _g}) == (40 * 58 + 30 * 35) // 4,
      str(QM._count_visual_tokens({"image_grid_thw": _g})))
check("无 image_grid_thw 时返回 0", QM._count_visual_tokens({}) == 0)
check("inputs 为 None 时返回 0（不抛异常）", QM._count_visual_tokens(None) == 0)

# prefill / 解码拆分进入日志
msgs_j = [m for m in LOGS if m.startswith("[Qwen35]")]
check("耗时分解含 prefill 行", any("其中 prefill" in m for m in msgs_j), "")
check("耗时分解含解码行", any("其中 解码" in m for m in msgs_j), "")
check("完成日志里带了 prefill/解码拆分",
      any("prefill" in m and "tok/s" in m and "英文提示词完成" in m for m in msgs_j))

print()
print("=" * 74)
print("K) 注意力后端解析（flash-attn 探测）")
print("=" * 74)
check("选项含 auto / flash_attention_2 / sdpa / eager",
      set(QM._ATTENTION_CHOICES) == {"auto", "flash_attention_2", "sdpa", "eager"},
      str(QM._ATTENTION_CHOICES))
check("attention 默认值是 auto（装了 flash-attn 就自动走 flash 路径）",
      QM.Qwen35PromptEnhancer.INPUT_TYPES()["required"]["attention"][1].get("default") == "auto",
      str(QM.Qwen35PromptEnhancer.INPUT_TYPES()["required"]["attention"]))
check("最低版本要求是 2.3.3（与 transformers 一致）",
      QM._FA_MIN_VERSION == (2, 3, 3), str(QM._FA_MIN_VERSION))

_real_ver = QM._flash_attn_version()
print(f"  当前环境 flash-attn = {_real_ver}")
check("_flash_attn_version 返回 None 或元组",
      _real_ver is None or isinstance(_real_ver, tuple), repr(_real_ver))
check("_flash_attn_available 与版本一致",
      QM._flash_attn_available() == (_real_ver is not None and _real_ver[:3] >= (2, 3, 3)),
      f"{_real_ver} / {QM._flash_attn_available()}")

# 用桩函数覆盖三种环境，验证 auto 的解析结果
_orig_fn = QM._flash_attn_version
try:
    QM._flash_attn_version = lambda: None
    _impl, _note = QM._resolve_attention("auto")
    check("未装 flash-attn 时 auto → sdpa", _impl == "sdpa", _impl)
    check("未装时给出说明", bool(_note), _note)

    QM._flash_attn_version = lambda: (2, 8, 3)
    _impl, _note = QM._resolve_attention("auto")
    check("装了 flash-attn 2.8.3 时 auto → flash_attention_2",
          _impl == "flash_attention_2", _impl)
    check("说明里带上版本号", "2.8.3" in _note, _note)

    QM._flash_attn_version = lambda: (2, 1, 0)
    _impl, _note = QM._resolve_attention("auto")
    check("flash-attn 版本低于 2.3.3 时退回 sdpa", _impl == "sdpa", _impl)
    check("版本过低时说明里点名 2.3.3", "2.3.3" in _note, _note)
finally:
    QM._flash_attn_version = _orig_fn

# 显式指定的值必须原样透传，不能被 auto 逻辑吞掉
for _want in ("flash_attention_2", "sdpa", "eager"):
    _impl, _note = QM._resolve_attention(_want)
    check(f"显式 {_want} 原样透传", _impl == _want and _note == "", f"{_impl} / {_note!r}")
check("空串按 auto 处理",
      QM._resolve_attention("")[0] == QM._resolve_attention("auto")[0])
check("None 按 auto 处理",
      QM._resolve_attention(None)[0] == QM._resolve_attention("auto")[0])

# 缓存键用解析后的后端：否则装了 flash-attn 还会一直复用 sdpa 的旧模型
import inspect as _ins
_src = _ins.getsource(QM.Qwen35PromptEnhancer._load)
check("缓存键用解析后的后端而不是原始选项", "key = (path, quantization, attn_impl)" in _src)
check("flash 初始化失败时退回 sdpa（不整个跑挂）",
      "本次退回 sdpa" in _src)

print()
print("=" * 74)
print("L) patch_embed 的 Conv3d -> GEMM 等价替换")
print("=" * 74)
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    class _VisCfg:
        patch_size = 16
        temporal_patch_size = 2
        in_channels = 3
        hidden_size = 64

    _orig_cls_forward = Qwen3VLVisionPatchEmbed.forward
    _pe = Qwen3VLVisionPatchEmbed(_VisCfg())
    _pe.eval()
    check("patch_embed 内部确实是 Conv3d",
          isinstance(_pe.proj, _t.nn.Conv3d),
          type(_pe.proj).__name__)
    check("Conv3d 的 kernel 与 stride 相同（是输出的前提）",
          tuple(_pe.proj.kernel_size) == tuple(_pe.proj.stride),
          f"{_pe.proj.kernel_size} / {_pe.proj.stride}")

    # 2 张图、每张 2x2 个 patch（grid_thw 的 h*w），共 8 个 patch
    _flat = 3 * 2 * 16 * 16
    _x2d = _t.randn(8, _flat)
    with _t.no_grad():
        _ref = _orig_cls_forward(_pe, _x2d)          # 原始 Conv3d 实现
        _x5d = _x2d.view(8, 3, 2, 16, 16)
        _ref5 = _orig_cls_forward(_pe, _x5d.view(8, _flat))

    check("原始实现输出形状 (8, 64)", tuple(_ref.shape) == (8, 64), str(tuple(_ref.shape)))

    # 打补丁
    _first = QM._patch_vision_patch_embed()
    check("补丁已生效（首次返回 True）", _first is True)
    check("补丁幂等（第二次返回 False）", QM._patch_vision_patch_embed() is False)
    check("类 forward 已被替换",
          Qwen3VLVisionPatchEmbed.forward is QM._gemm_patch_embed_forward)

    with _t.no_grad():
        _new = _pe(_x2d)
        _new5 = _pe(_x5d)
    _err = float((_new - _ref).abs().max())
    print(f"  Conv3d vs GEMM 最大绝对误差 = {_err:.3e}")
    check("2D 输入下与原始 Conv3d 数值一致（误差 < 1e-5）", _err < 1e-5, f"{_err:.3e}")
    check("5D 输入下也一致",
          float((_new5 - _ref5).abs().max()) < 1e-5,
          f"{float((_new5 - _ref5).abs().max()):.3e}")
    check("输出形状与原始一致", tuple(_new.shape) == tuple(_ref.shape))
    check("权重没有被复制成新张量（仍是同一块内存的视图）",
          _pe.proj.weight.reshape(_pe.proj.out_channels, -1).data_ptr()
          == _pe.proj.weight.data_ptr())

    # 参数名 / state_dict 键名不能变，否则会跟官方权重对不上
    _keys = set(_pe.state_dict().keys())
    check("state_dict 键名不变（仍含 proj.weight / proj.bias）",
          _keys == {"proj.weight", "proj.bias"}, str(sorted(_keys)))

    # 环境变量逃生门
    import os as _os
    _os.environ["QWEN35_NO_PATCH_EMBED_FIX"] = "1"
    QM._PATCH_EMBED_PATCHED = False
    check("QWEN35_NO_PATCH_EMBED_FIX=1 时跳过补丁",
          QM._patch_vision_patch_embed() is False)
    _os.environ.pop("QWEN35_NO_PATCH_EMBED_FIX", None)
    QM._PATCH_EMBED_PATCHED = True

    check("体检线常量存在", QM._SLOW_VISION_PER_TOKEN_MS == 5.0)
except ImportError as e:
    print(f"  跳过（transformers 里没有 Qwen3VL 实现）: {e}")

print()
print("=" * 74)
print("M) 补丁 v2 —— 按结构识别 + 打到模型实例上（Qwen3.5 回归）")
print("=" * 74)
# 回归背景：补丁 v1 只按类名补 Qwen3VLVisionPatchEmbed.forward。
# 换成 Qwen3.5（类名 Qwen3_5VisionPatchEmbed）后补丁静默失效，
# 但类级标志仍是 True → 日志照报「已换成 GEMM」，prefill 却从 0.69s 退回 75.61s。
# 详见 h3_workflow/verify_qwen35_patch.py 与 诊断报告_扩写慢.md。
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed

    class _Cfg35:
        patch_size = 16
        temporal_patch_size = 2
        in_channels = 3
        hidden_size = 48

    # --- 结构守卫：四个前提缺一不可 ---
    _ok = _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    check("kernel==stride / padding=0 / groups=1 判为等价",
          QM._is_gemm_equivalent_conv3d(_ok) is True)
    check("kernel != stride 判为不等价",
          QM._is_gemm_equivalent_conv3d(
              _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(1, 16, 16))) is False)
    check("带 padding 判为不等价",
          QM._is_gemm_equivalent_conv3d(
              _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(2, 16, 16),
                            padding=(1, 0, 0))) is False)
    check("groups != 1 判为不等价",
          QM._is_gemm_equivalent_conv3d(
              _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(2, 16, 16),
                            groups=3)) is False)
    check("非 Conv3d 判为不等价",
          QM._is_gemm_equivalent_conv3d(_t.nn.Linear(4, 4)) is False)

    # --- 实例级替换：认得出 Qwen3.5，且幂等 ---
    _pe35 = Qwen3_5VisionPatchEmbed(_Cfg35()).eval()
    _orig35 = Qwen3_5VisionPatchEmbed.forward
    _n1 = QM._patch_patch_embed_instances(_pe35)
    check("实例级扫描认出 Qwen3.5 的 patch_embed", _n1 == 1, f"{_n1} 个")
    check("实例 forward 被换成 GEMM",
          _pe35.forward.__func__ is QM._gemm_patch_embed_forward)
    check("幂等：重复扫描计数不翻倍（返回的是该模型实例总数）",
          QM._patch_patch_embed_instances(_pe35) == 1)
    check("幂等：forward 仍指向同一个替换实现",
          _pe35.forward.__func__ is QM._gemm_patch_embed_forward)

    # 类型变了（不同类）但结构一样 -> 数值应当仍然等价
    _flat35 = 3 * 2 * 16 * 16
    _x35 = _t.randn(5, _flat35)
    with _t.no_grad():
        _r35 = _orig35(_pe35, _x35)
        _g35 = _pe35(_x35)
    _e35 = float((_r35 - _g35).abs().max())
    check("Qwen3.5 实例替换后数值仍等价（误差 < 1e-5）", _e35 < 1e-5, f"{_e35:.3e}")

    # 维度不匹配要报错，不能静默算错
    try:
        with _t.no_grad():
            _pe35(_t.randn(3, _flat35 + 16))
        _raised = False
    except RuntimeError:
        _raised = True
    check("输入布局不对时报错而不是静默出错", _raised)

    # --- 名字守卫：类名不含 patchembed 的模块不碰 ---
    class _ConvHolder(_t.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _t.nn.Conv3d(3, 6, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    check("类名不含 patchembed 的模块不会被替换（名字守卫）",
          QM._patch_patch_embed_instances(_ConvHolder()) == 0)

    # --- _patch_vision_patch_embed(model=...) 返回语义与实例计数 ---
    QM._PATCH_EMBED_PATCHED = False
    QM._PATCH_EMBED_INSTANCES = 0
    _pe35b = Qwen3_5VisionPatchEmbed(_Cfg35()).eval()
    check("传 model 时该实例被计入替换数",
          QM._patch_vision_patch_embed(_pe35b) is True and QM._PATCH_EMBED_INSTANCES == 1,
          f"instances={QM._PATCH_EMBED_INSTANCES}")
    check("再传一次返回 False（已打过）",
          QM._patch_vision_patch_embed(_pe35b) is False)

    check("类级目标清单覆盖 Qwen3-VL 与 Qwen3.5 两个族",
          any("qwen3_vl.modeling" in m for m, _c in QM._PATCH_EMBED_CLASS_TARGETS)
          and any("qwen3_5.modeling" in m for m, _c in QM._PATCH_EMBED_CLASS_TARGETS))
except ImportError as e:
    print(f"  跳过（transformers 里没有 Qwen3.5 实现）: {e}")

print()
print("=" * 74)
print("N) 线性注意力快路径体检（Qwen3.5 混合架构 / torch 回退）")
print("=" * 74)
# 背景：Qwen3.5 是 24/32 层 Gated DeltaNet 的混合架构。fla 没装时
# decode 落到 transformers 自带的 torch_recurrent_gated_delta_rule
# （fp32 + 逐 token + 小张量串行），实测占解码约一半。
# 判定必须看「已加载模型实例上挂的函数来自哪个模块」——
# 与 M 段「补丁打在别的类上却报成功」是同一类教训。
try:
    def _fake_fla(*a, **k):
        return None

    def _fake_torch_impl(*a, **k):
        return None

    _fake_fla.__module__ = "fla.ops.gated_delta_rule.fused_recurrent"
    _fake_torch_impl.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"

    class _FakeDeltaNet(_t.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.recurrent_gated_delta_rule = fn

    class _Plain(_t.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = _t.nn.Linear(2, 2)

    check("体检线常量存在",
          isinstance(QM._LINEAR_ATTN_SLOW_DECODE_TOK_S, float),
          str(QM._LINEAR_ATTN_SLOW_DECODE_TOK_S))
    # 这条线必须落在 bf16 正常值（本机 18.5~21.1 tok/s）**之下**，
    # 否则正常跑完也会刷这条提示，变成噪音。
    check("触发线低于 bf16 正常值（12 < line < 20）",
          12.0 < QM._LINEAR_ATTN_SLOW_DECODE_TOK_S < 20.0,
          str(QM._LINEAR_ATTN_SLOW_DECODE_TOK_S))

    # --- 扫描：按实例上挂的函数所属模块判定 ---
    _m = _t.nn.ModuleList([_FakeDeltaNet(_fake_fla), _FakeDeltaNet(_fake_fla),
                           _FakeDeltaNet(_fake_torch_impl), _Plain()])
    check("扫描只统计含 recurrent_gated_delta_rule 的模块",
          QM._scan_linear_attn_fast_path(_m) == (2, 3),
          str(QM._scan_linear_attn_fast_path(_m)))
    check("扫描结果写回模块级变量",
          QM._LIN_ATTN_FUSED == 2 and QM._LIN_ATTN_TOTAL == 3)
    check("非混合架构模型扫出 (0, 0)",
          QM._scan_linear_attn_fast_path(_t.nn.ModuleList([_Plain()])) == (0, 0))

    # --- 加载期提示 ---
    check("不是混合架构 -> 加载期无提示", QM._linear_attn_load_note(0, 0) is None)
    _all_fused = QM._linear_attn_load_note(24, 24)
    check("全部走融合内核 -> 提示里说 fla 已生效",
          _all_fused and "fla" in _all_fused, str(_all_fused)[:60])
    _part = QM._linear_attn_load_note(0, 24)
    check("全部走回退 -> 提示里给出层数与 fla 安装指引",
          _part and "24" in _part and "flash-linear-attention" in _part)
    _part2 = QM._linear_attn_load_note(12, 24)
    check("部分走回退 -> 提示里报出回退层数",
          _part2 and "12 层走 torch 回退" in _part2, str(_part2)[:70])

    # --- 耗时分解里的解码侧提示 ---
    # 注意这里用 263tok/43.95s ≈ 6.0 tok/s（8bit 那次的真实速率）才该触发。
    # 263tok/12.61s ≈ 20.9 tok/s 是 bf16 的**正常**值，不该刷这条提示 ——
    # 旧断言把正常值当成了「偏慢」，等于把提示线划在正常值之上，会一直误报。
    QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL = 0, 24
    check("bf16 正常速率（20.9 tok/s）不该提示",
          QM._linear_attn_decode_note(263, 12.61) == "")
    check("解码偏慢（6.0 tok/s）且全走回退 -> 给出定量提示",
          bool(QM._linear_attn_decode_note(263, 43.95)))
    _note = QM._linear_attn_decode_note(263, 43.95)
    check("提示里含 3365 kernel/token 的实测口径",
          "3365" in _note, _note[:60])
    check("提示里含「不要装 flash-linear-attention」的说明",
          "flash-linear-attention" in _note, _note[:60])
    # 2026-10-07：解码慢的实测定因是「线程被调度到 E-core」，不是线性注意力。
    # 提示必须把用户引向 P-core 亲和性，而不是引向装 fla（后者已被证伪且会让模型加载失败）。
    check("提示把「慢」指向 CPU 的 P-core/E-core，并给出实测数字",
          "P-core" in _note and "E-core" in _note
          and "18.8" in _note and "2.66" in _note, _note[-90:])
    check("提示给出可执行的解法（钉 P-core 的工具名）",
          "pin_comfyui_pcores" in _note, _note[-90:])
    check("解码够快时不提示", QM._linear_attn_decode_note(600, 12.0) == "")
    check("样本太小不提示", QM._linear_attn_decode_note(8, 5.0) == "")
    QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL = 24, 24
    check("全部融合时不提示", QM._linear_attn_decode_note(263, 43.95) == "")
    QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL = 0, 0
    check("非混合架构不提示", QM._linear_attn_decode_note(263, 43.95) == "")
except Exception as e:
    print("  跳过:", type(e).__name__, e)

print()
print("=" * 74)
print("O) bnb 量化档：int8 阈值开关 + skip 列表（2026-10-07 受控对照结论）")
print("=" * 74)
# 背景：8bit 在这台机器上是**净负收益**（真机 A/B：71.01s vs 4bit 54.06s / none 51.85s）。
# 受控对照（bench_qwen35_proj_ab.py，bf16 与候选背靠背交替、各取最小值）定位到两处可改：
#   ① threshold>0 走 int8_mixed_scaled_mm；改成 0.0 走 int8_scaled_mm，快 1.7~1.9x
#   ② 调用次数多、矩阵小的那几组，bnb 固定开销主导 → 把小投影排除出量化
# 并确认「要省显存该用 4bit 而不是 8bit」这条结论确实写进了提示。
try:
    check("_INT8_THRESHOLD 已设为 0.0", QM._INT8_THRESHOLD == 0.0, str(QM._INT8_THRESHOLD))
    check("0.0 会落到 bnb 的 else 分支（纯 int8_scaled_mm）",
          not (QM._INT8_THRESHOLD > 0.0))
    for _k in ("model.visual", "visual", "lm_head", "in_proj_a", "in_proj_b"):
        check(f"skip 列表含 {_k}", _k in QM._QUANT_SKIP_MODULES)

    from transformers import BitsAndBytesConfig
    _cfg8 = BitsAndBytesConfig(load_in_8bit=True,
                               llm_int8_threshold=QM._INT8_THRESHOLD,
                               llm_int8_skip_modules=list(QM._QUANT_SKIP_MODULES))
    check("transformers 接受 llm_int8_threshold=0.0 并原样保留",
          float(_cfg8.llm_int8_threshold) == 0.0, str(_cfg8.llm_int8_threshold))
    check("量化方式识别为 llm_int8", _cfg8.quantization_method() == "llm_int8")

    # skip 列表按 **后缀** 匹配（transformers/quantizers/quantizers_utils.py:37）
    from transformers.quantizers.quantizers_utils import should_convert_module as _scm
    check("线性注意力 in_proj_a 被跳过（后缀匹配真的生效）",
          _scm("model.language_model.layers.3.linear_attn.in_proj_a",
               list(QM._QUANT_SKIP_MODULES)) is False)
    check("线性注意力 out_proj 仍被量化（没误伤）",
          _scm("model.language_model.layers.3.linear_attn.out_proj",
               list(QM._QUANT_SKIP_MODULES)) is True)
    check("MLP 投影仍被量化",
          _scm("model.language_model.layers.7.mlp.gate_proj",
               list(QM._QUANT_SKIP_MODULES)) is True)
    check("视觉塔被跳过",
          _scm("model.visual.patch_embed.proj",
               list(QM._QUANT_SKIP_MODULES)) is False)
except ImportError as e:
    print("  跳过（没有 transformers）:", e)
except Exception as e:
    print("  跳过:", type(e).__name__, e)

# threshold=0.0 必须真的算对，不能只是"被选中"就跑
try:
    import bitsandbytes as _bnb
    if _t.cuda.is_available():
        _t.manual_seed(0)
        _ref_lin = _t.nn.Linear(256, 512, bias=False, device="cuda", dtype=_t.bfloat16)
        with _t.no_grad():
            _ref_lin.weight.mul_(0.05)
        _xi = _t.randn(4, 256, device="cuda", dtype=_t.bfloat16)
        with _t.no_grad():
            _want = _ref_lin(_xi)
        _q8 = _bnb.nn.Linear8bitLt(256, 512, bias=False, has_fp16_weights=False,
                                   threshold=QM._INT8_THRESHOLD)
        _q8.weight = _bnb.nn.Int8Params(_ref_lin.weight.data.clone(),
                                        requires_grad=False, has_fp16_weights=False)
        _q8 = _q8.to("cuda").eval()
        check("threshold=0.0 已挂到 state 上", float(_q8.state.threshold) == 0.0,
              str(_q8.state.threshold))
        with _t.no_grad():
            _got8 = _q8(_xi)
        check("threshold=0.0 输出是有限值", bool(_t.isfinite(_got8).all()))
        _den = float(_want.float().abs().max())
        _rel8 = float((_got8.float() - _want.float()).abs().max()) / max(_den, 1e-9)
        check("threshold=0.0 数值与 bf16 接近（相对误差 < 5e-2）", _rel8 < 5e-2,
              f"{_rel8:.3e}")
        _q8.state.threshold = 6.0
        with _t.no_grad():
            _got6 = _q8(_xi)
        _rel6 = float((_got6.float() - _want.float()).abs().max()) / max(_den, 1e-9)
        check("threshold=6.0 同样正确（两种分支都可用）", _rel6 < 5e-2, f"{_rel6:.3e}")
        del _q8, _ref_lin, _xi, _want, _got8, _got6
        _t.cuda.empty_cache()
    else:
        print("  跳过 GPU 数值验证（无 CUDA）")
except ImportError as e:
    print("  跳过（没有 bitsandbytes）:", e)
except Exception as e:
    print("  跳过:", type(e).__name__, e)

# 解码体检文案：8bit 要指向 4bit；4bit 慢时不能拿 4bit 当解释
try:
    QM._CPU_OFFLOAD_GIB = 0.0
    _w8 = QM._decode_speed_warning(263, 43.95, quantization="8bit")
    check("8bit 提示里给了 4bit 的实测结论", "4bit" in _w8, _w8[:50])
    check("8bit 提示里提到 threshold 已调", "llm_int8_threshold" in _w8)
    check("8bit 提示里含 bf16 带宽实测数字", "754" in _w8 or "855" in _w8)
    check("8bit 提示里含 int8 内核实测带宽", "44" in _w8 or "117" in _w8)
    _w4 = QM._decode_speed_warning(263, 43.95, quantization="4bit")
    check("4bit 慢时明确说 4bit 解释不了", "解释不了" in _w4, _w4[:50])
    check("正常速度不提示（8bit）",
          QM._decode_speed_warning(600, 12.0, quantization="8bit") == "")
    check("正常速度不提示（4bit）",
          QM._decode_speed_warning(600, 12.0, quantization="4bit") == "")
except Exception as e:
    print("  跳过:", type(e).__name__, e)

print()
print("=" * 74)
print("P) 真机 A/B 定案：4bit 54.06s / 8bit 71.01s / none 51.85s —— 差在加载，不在解码")
print("=" * 74)
# 用户实跑（comfyui.log，2026-10-07 06:03 与 06:09 两次），同一工作流、同一组参考图：
#   档位    加载      EN 解码       ZH 解码       整轮     驻留
#   none   23.05s  12.61s(20.9)  12.95s(21.6)  51.85s  17.5GiB
#   4bit   19.60s  16.51s(17.3)  14.59s(19.7)  54.06s   7.9GiB
#   8bit   33.90s  15.72s(12.4)  15.83s(13.1)  71.01s  11.1GiB
# 关键：8bit 多花的 16.95s 里 14.30s（84%）是**加载**；解码墙钟几乎持平
# （8bit 那次生成的 token 更少），按每 token 才算慢 1.4~1.6 倍。
# 由此产生两处改动：体检线 12.0→14.0；判为 8bit 主因时压掉 fla 建议。
_i_bak = QM._CPU_OFFLOAD_GIB
try:
    import inspect as _inspect
    QM._CPU_OFFLOAD_GIB = 0.0

    check("体检线高于 8bit 实测上限 13.1（否则 8bit 触发不了体检）",
          QM._SLOW_DECODE_TOK_S > 13.1, str(QM._SLOW_DECODE_TOK_S))
    check("体检线低于 4bit 实测下限 17.3（否则会误报 4bit）",
          QM._SLOW_DECODE_TOK_S < 17.3, str(QM._SLOW_DECODE_TOK_S))
    check("真机 8bit（195 tok / 15.72s）现在会被判为慢",
          QM._decode_speed_warning(195, 15.72, "8bit") != "")
    check("真机 4bit（286 tok / 16.51s）不会被误报",
          QM._decode_speed_warning(286, 16.51, "4bit") == "")

    _clssrc = _inspect.getsource(QM.Qwen35PromptEnhancer)
    check("加载期 8bit 提示带真机整轮对照（71.01 / 54.06）",
          "71.01" in _clssrc and "54.06" in _clssrc)
    check("加载期提示点明 14.3s 是加载开销", "14.3s" in _clssrc)
    check("加载期提示仍保留 llm_int8_threshold 与 4bit 指向",
          "llm_int8_threshold" in _clssrc and "**4bit**" in _clssrc)
    check("模块 docstring 已换成真机三档对照",
          "54.06s" in (QM.__doc__ or "") and "71.01s" in (QM.__doc__ or ""))
    check("「8bit 解码慢 3~8 倍」这个旧说法已从源码里清掉",
          "3~8 倍" not in (QM.__doc__ or "")
          and "3~8 倍" not in _clssrc)

    # 互斥规则：判为 8bit 主因时不再提 fla（否则把人往次优解带）
    _f_bak, _t_bak = QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL
    try:
        QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL = 0, 24
        _n8 = QM._decode_notes(195, 15.72, "8bit")
        check("8bit 主因命中 → 只给 1 条（压掉 fla 建议）", len(_n8) == 1, str(len(_n8)))
        check("  留下的是量化归因那条",
              bool(_n8) and "quantization=8bit" in _n8[0])
        check("  且不含「装 fla」的建议",
              all("fla" not in _x for _x in _n8), str(_n8)[:80])
        _n4 = QM._decode_notes(195, 15.72, "4bit")
        check("4bit 慢时两条都给（fla 建议不被压）", len(_n4) == 2, str(len(_n4)))
    finally:
        QM._LIN_ATTN_FUSED, QM._LIN_ATTN_TOTAL = _f_bak, _t_bak
except Exception as e:
    print("  跳过:", type(e).__name__, e)
finally:
    QM._CPU_OFFLOAD_GIB = _i_bak

print()
print("=" * 74)
if FAIL:
    print(f"存在 {len(FAIL)} 项问题:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("全部通过")
