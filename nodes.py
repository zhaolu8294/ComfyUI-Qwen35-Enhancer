# -*- coding: utf-8 -*-
"""
Qwen3.5 / Qwen3-VL nodes for ComfyUI

本文件提供两个节点：
  1. Qwen35PromptEnhancer    —— 把简单提示词（+ 可选参考图）用本地 Qwen 多模态
                                模型扩写成 MiniMax H3 官方规范的三段式视频提示词。
  2. Qwen35BatchImageTagger  —— 给一个文件夹批量打标，标签写到与图片同目录、
                                同名的 .txt；系统提示词可改。
两者共用同一条加载链路（含全部速度优化）与同一份常驻模型缓存 ——
详见文件末尾「批量打标节点」的总说明。

特点
----
* 架构自动适配: 用 AutoModelForImageTextToText，同一份代码同时支持
  Qwen3_5ForConditionalGeneration (Qwen3.5-9B) 与
  Qwen3VLForConditionalGeneration (Qwen3-VL-8B/4B)，加新模型无需改代码。
* 批量打标: Qwen35BatchImageTagger —— 一次加载打完整个文件夹（走扩写节点
  循环 N 次的话每张都要重载）；默认 skip 绝不覆盖已有 txt；单张失败不中断
  整批，但 ComfyUI 的「取消」会立刻中止并保留已完成的部分。
* 模型自动发现: 扫描 models/text_encoders 与 models/prompt_generator 下
  所有含 config.json 的完整 HF 文件夹。
* 图片可选: 不接 image 时走纯文本改写；接 image 时把图一起喂给模型。
* 多图参考: image / image_2 / image_3 / image_4 最多 4 路输入（各自可以是 batch），
  按连线顺序编号，正好对应 H3 的 <Picture 1> .. <Picture N>。
* 模式感知: mode 决定往 system prompt 注入哪套 H3 规则
  (text2video / image2video 首帧 / reference 参考图引用)。
* 思考模式剥离: Qwen3.5 的 chat template 默认注入  thinking，
  输出会自动去掉思考块，只保留最终提示词。
* 显存可控: none / 8bit / 4bit 量化 + 跑完可释放。**默认 none。**
  bnb 的量化档是「省显存、费速度」，8bit 尤其反直觉。2026-10-07 真机 A/B
  （Qwen3.5-9B，同一工作流、同一组参考图，详见诊断报告第十三章）：
      none  51.85s（加载 23.05s｜解码 20.9/21.6 tok/s｜17.5GiB）
      4bit  54.06s（加载 19.60s｜解码 17.3/19.7 tok/s｜ 7.9GiB）
      8bit  71.01s（加载 33.90s｜解码 12.4/13.1 tok/s｜11.1GiB）
  → 8bit 多花的 17s 里 **14.3s（84%）花在加载**上，按每 token 算解码只慢 1.4~1.6 倍；
    而 4bit 整轮只比 none 慢 4%，显存却只剩 45%。**要省显存请用 4bit。**
  （详见文件头部 _INT8_THRESHOLD 后的注释。）
  量化时会排除视觉塔（_QUANT_SKIP_MODULES）：它有约 430M 参数，
  4bit 化后每层前向都要实时反量化，实测把图片 prefill 从几秒拖到 57~86 秒；
  同时排除 Qwen3.5 线性注意力里两个 Linear(4096,32) 的小投影。
* 视觉塔入口换 GEMM: Qwen3-VL 与 Qwen3.5 的 patch_embed 都是
  Conv3d(kernel=stride=(2,16,16))，输出恒为 1x1x1 —— 数学上就是个 GEMM。
  本机（torch 2.9.1+cu130 / cuDNN 9.12 / Windows）实测 Conv3d 在
  **fp32 下 6.5ms，在 fp16/bf16 下首次调用超过 45 秒都没返回**。
  视觉塔整段 29 秒里约 96% 卡在这个卷积上，是 prefill 的头号瓶颈。
  节点加载后会把它的 forward 换成 F.linear（权重只 reshape、不复制，数值等价，
  实测最大绝对误差 2e-06），绕开那条病态路径。
  **按「结构」替换而不是按「类名」**：两个模型族的类名不同
  （Qwen3VLVisionPatchEmbed / Qwen3_5VisionPatchEmbed），只按类名打补丁
  会在换模型后静默失效（实测 prefill 0.69s → 75.61s）。
  现在是遍历加载好的模型，凡类名含 patchembed 且 .proj 是 kernel==stride 的
  Conv3d 就换 forward，换模型族不会再漏。
  可用 QWEN35_NO_PATCH_EMBED_FIX=1 关闭。
* 注意力后端可选: attention = auto / flash_attention_2 / sdpa / eager。
  auto 在装了 flash-attn（>=2.3.3）时自动用 flash_attention_2，否则退回 sdpa，
  所以卸载 flash-attn 也不会把工作流跑挂。切后端时 PretrainedConfig 的 setter
  会把值递归写到 vision_config，视觉塔会自动跟着切。
  **但要如实说明**：实测同一份权重、同一份输入下，eager / sdpa / flash_attention_2
  的视觉塔耗时是 29.16 / 28.91 / 29.30 秒 —— 三种后端**毫无差别**。
  「缺 flash-attn 导致视觉塔逐图切块变慢」这个早先的判断已被推翻，
  flash-attn 不是这里的分水岭（它仍值得装：省显存、LLM 侧长上下文更快）。
* 进度可见: 生成阶段逐 token 上报
    - ComfyUI 前端会在这个节点上画出进度条（走 comfy.utils.ProgressBar）
    - 同时往控制台/日志按节流打进度行；真实终端下还会原地刷新一条实时条
    - 顺带接上 ComfyUI 的中断标志，前端点"取消"能真正打断长生成
* 双语成本可控（**默认关闭**）: 中文段只供人工核对、不进 H3，它的一切开销
  都是可省的额外成本，所以默认 bilingual=off，第二段根本不执行；
  需要在界面上核对时才把下拉切到 en_then_zh。开启后代码也会替你把成本压住
    - 预算按英文实际输出长度换算（_zh_budget），不再照抄 max_new_tokens 跑满
    - 贪心解码 + 显式 EOS，并在 prompt 里明令禁止抄写原文
    - 回声清理 _drop_english_echo 兜底（"先抄英文再翻译"会让耗时翻到 3 倍）
    - 日志打印两段 token 比并自动告警（实测正常约 1.1 倍）
* 提速诊断: device_map="auto" 在显存不够时**不会报错**，而是静默把一部分层
  （经常正好是视觉塔）放到 CPU 上，于是 4bit 8B 在 4090D 上从正常的
  40~80 tok/s 掉到 1 tok/s，日志里只看到"慢"看不到原因。所以节点会
    - 加载前打印「可用显存 x → y，本档量化约需 z」并在不足时告警
    - 加载后打印「模型放置：cuda:0×32 …」与视觉塔所在设备，有 CPU 层就告警
    - 耗时分解把生成拆成 prefill / 解码两行（两者变慢的原因完全不同）
    - 解码低于 5 tok/s 时直接提示"这是 CPU 摊派/换页，不是节点算法问题"
"""

import os
import gc
import sys
import json
import re
import time
import types
import logging
import importlib

import torch

import folder_paths

logger = logging.getLogger("Qwen35Enhancer")

# ---------------------------------------------------------------------------
# 搜索路径：ComfyUI 默认只注册了 text_encoders，prompt_generator 要自己拼
# ---------------------------------------------------------------------------
MODELS_DIR = getattr(folder_paths, "models_dir", None) or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)

SEARCH_DIRS = [
    os.path.join(MODELS_DIR, "text_encoders"),
    os.path.join(MODELS_DIR, "prompt_generator"),
    os.path.join(MODELS_DIR, "LLM"),
]

# 过滤掉的"文本编码器"目录名（这些是给 diffusion 用的，没有 lm_head / 不是对话模型）
SKIP_DIR_HINTS = (
    "minimax_h3",
    "qwen_image",
    "Qwen_image",
    "flux",
    "clip",
    "t5",
    "llava",
)

# 这些架构不是对话/改写模型，不列进下拉
SKIP_ARCH_HINTS = ("Florence", "Whisper", "CLIP", "T5", "Siglip", "Bert")


def discover_local_models():
    """扫描所有含 config.json 的完整 HF 模型文件夹，返回 {显示名: 绝对路径}"""
    found = {}
    for base in SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            full = os.path.join(base, name)
            if not os.path.isdir(full):
                continue
            cfg_path = os.path.join(full, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            # 过滤明显不是对话模型的目录
            lowered = name.lower()
            if any(h.lower() in lowered for h in SKIP_DIR_HINTS):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                arch = (cfg.get("architectures") or ["?"])[0]
                mtype = cfg.get("model_type", "?")
            except Exception:
                arch, mtype = "?", "?"
            # 过滤 caption / 编码器类架构（不是对话模型）
            if any(h.lower() in arch.lower() for h in SKIP_ARCH_HINTS):
                continue
            label = f"{name}  [{arch}]"
            found[label] = full
    return found


def _model_choices():
    models = discover_local_models()
    if not models:
        return ["<no local HF model found>"]
    return list(models.keys())


# ---------------------------------------------------------------------------
# thinking 块剥离
# ---------------------------------------------------------------------------
_THINK_PATTERNS = [
    # 成对闭合块：<think ...> ... </think ...>
    re.compile(r"<think\b[^>]*>.*?</think\b[^>]*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"</think\b[^>]*>", re.IGNORECASE),
    re.compile(r"<\|end_of_thinking\|>", re.IGNORECASE),
]
# 未闭合的 <think ...> 直到结尾，整段砍掉
_THINK_TAIL = re.compile(r"<think\b[^>]*>.*$", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """去掉 Qwen3.5 / Qwen3-VL-Thinking 产生的思考块，只留最终答案。"""
    if not text:
        return ""
    out = text
    for pat in _THINK_PATTERNS:
        out = pat.sub("", out)
    out = _THINK_TAIL.sub("", out)
    return out.strip()


# ---------------------------------------------------------------------------
# H3 三段式规范化
# ---------------------------------------------------------------------------
_SEC_1 = re.compile(r"^\s*integrated_multimodal_description\s*:", re.MULTILINE | re.IGNORECASE)
_SEC_2 = re.compile(r"^\s*overall_soundscape\s*:", re.MULTILINE | re.IGNORECASE)


def normalize_h3_sections(text: str) -> str:
    """
    保证输出符合 H3 三段式：
      integrated_multimodal_description: ...
      overall_soundscape: ...
      non_diegetic_music: ...
    模型经常漏掉第一段的标签、或把三段挤在一起，这里统一修正。
    """
    if not text:
        return text
    t = text.strip()

    # 1) 让三个标签各自单独起段（前后空行）
    for label in ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"):
        t = re.sub(rf"[ \t]*\n*[ \t]*{label}\s*:", f"\n\n{label}:", t, flags=re.IGNORECASE)
    t = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", t).strip()

    # 2) 补齐缺失的第一段标签
    if not re.search(r"^integrated_multimodal_description\s*:", t, re.MULTILINE | re.IGNORECASE):
        m = _SEC_2.search(t)
        if m:
            head = t[:m.start()].strip()
            tail = t[m.start():].strip()
            t = (f"integrated_multimodal_description: {head}\n\n{tail}" if head
                 else f"integrated_multimodal_description: N/A\n\n{tail}")
        else:
            t = f"integrated_multimodal_description: {t}"

    return t


# ---------------------------------------------------------------------------
# 中文预览：把模型偶尔翻掉的三段标签还原成英文
#
# 中文版只给人看，但标签名必须保持英文才能和英文版逐段对照。模型多数情况下
# 会照做，偶尔会把 "overall_soundscape" 之类也翻掉，这里做一次无害的还原。
# ---------------------------------------------------------------------------
_ZH_LABEL_MAP = {
    "integrated_multimodal_description": (
        "综合多模态描述", "多模态综合描述", "集成多模态描述", "综合多模态说明",
    ),
    "overall_soundscape": (
        "整体声音环境", "整体声场", "整体音景", "声音环境", "环境音景",
    ),
    "non_diegetic_music": (
        "非叙事音乐", "非画内音乐", "画外音乐", "非剧情音乐",
    ),
}


def _restore_zh_labels(text: str) -> str:
    """还原被翻成中文的三段标签，兼容「中文（english）：」与「中文：」两种写法。"""
    if not text:
        return text
    t = text
    for en, zh_variants in _ZH_LABEL_MAP.items():
        for zh in zh_variants:
            t = re.sub(
                rf"(?m)^[ \t]*{zh}[ \t]*[（(][ \t]*{en}[ \t]*[)）][ \t]*[：:]",
                f"{en}:", t,
            )
            t = re.sub(rf"(?m)^[ \t]*{zh}[ \t]*[：:]", f"{en}:", t)
    return t


# ---------------------------------------------------------------------------
# 翻译阶段的两个后处理：砍英文回声 / 约束预算
#
# 实测最贵的失败模式不是"翻得不好"，而是模型先把英文原文整段抄一遍再翻译。
# 那样第二段就要解码约 2 倍 token，整轮耗时直接翻到 3 倍。这里做一道兜底：
# 只要在"第一个中文字符之前"出现了三段标签，就认为那是抄回来的英文，砍掉。
# ---------------------------------------------------------------------------
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ANY_LABEL = re.compile(
    r"(?m)^[ \t]*(integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*[：:]"
)


def _has_cjk(text: str) -> bool:
    return bool(text) and _CJK.search(text) is not None


def _drop_english_echo(text: str) -> str:
    """砍掉翻译结果开头被抄了一遍的英文原文。

    判定：在第一个中文字符之前出现的最后一个三段标签，就是中文版真正的起点。
    标签已经在 0 位置（正常情况）或整段没有中文时，原样返回不猜。
    砍完还必须留着三段标签，否则宁可不动 —— 绝不把有效内容切没。
    """
    if not text:
        return text
    first_cjk = _CJK.search(text)
    if first_cjk is None:
        return text                     # 整段没中文，交给下游报错，不在这里猜
    cut = 0
    for lab in _ANY_LABEL.finditer(text):
        if lab.start() < first_cjk.start():
            cut = lab.start()
    if cut <= 0:
        return text
    stripped = text[cut:].lstrip()
    if not _ANY_LABEL.search(stripped):
        return text
    return stripped


# ---------------------------------------------------------------------------
# 模式相关的追加规则（注入到 system prompt）
# ---------------------------------------------------------------------------
MODE_HINTS = {
    "text2video": "",

    "image2video": (
        "[IMAGE MODE — the attached image is the FIRST FRAME]\n"
        "The attached image is the video's first frame, NOT a style reference and NOT a <Picture>.\n"
        "Never emit <Picture i> tags in this mode.\n"
        "Open the description with the exact content of that frame — subject, wardrobe, pose, "
        "lighting, background — then describe how the motion develops forward from it."
    ),

    "reference": (
        "[REFERENCE MODE — the attached images are <Picture 1> .. <Picture N>]\n"
        "Images are numbered strictly in the order they are attached, starting at <Picture 1>.\n"
        "The main description MUST explicitly cite at least one of them, e.g. "
        "\"<Picture 1> defines the lead character's face and outfit; <Picture 2> defines the rooftop set.\"\n"
        "Use the tags verbatim inside the description. Never renumber an image and never invent "
        "a <Picture i> tag for an image that was not attached.\n"
        "Decide which reference governs identity, which governs wardrobe and which governs "
        "environment, and state that mapping explicitly before describing the shot."
    ),
}


def _inject_mode_hint(system_prompt: str, mode: str) -> str:
    """把 mode 对应的规则插到 system prompt 的收尾句之前（没有收尾句就追加尾部）。"""
    hint = MODE_HINTS.get(mode, "")
    if not hint:
        return system_prompt
    marker = "Rewrite the user input now:"
    if marker in system_prompt:
        return system_prompt.replace(marker, f"{hint}\n\n{marker}")
    return f"{system_prompt}\n\n{hint}"


# ---------------------------------------------------------------------------
# chat template 能力嗅探
#
# enable_thinking 是 Qwen3.5 模板才有的变量（传 False 会让模板注入一个空
# <think> 块来关闭思考）。Qwen3-VL 的模板里根本没有这个变量，传进去会被
# transformers 静默忽略，只在控制台留一条警告。所以先看模板里有没有它。
# 无论哪种情况，strip_thinking() 都是最后一道保障。
# ---------------------------------------------------------------------------
def _supports_enable_thinking(processor) -> bool:
    try:
        for obj in (processor, getattr(processor, "tokenizer", None)):
            tmpl = getattr(obj, "chat_template", None)
            if tmpl:
                return "enable_thinking" in tmpl
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# EOS 补全
#
# 慢的另一个隐藏原因：如果 generation_config 里没登记对话结束符
# （Qwen 的 <|im_end|>），generate 就永远不会停，一路解码到 max_new_tokens
# 上限。两个阶段都受影响，第二段尤其致命。
# 这里把 config 里的 EOS 和分词器里真实存在的结束符并起来一起传给 generate。
# 用 convert_ids_to_tokens 往返校验，防止把 unk id 当成 EOS 传进去。
# ---------------------------------------------------------------------------
_EOS_CANDIDATES = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")


def _eos_token_ids(processor, model):
    ids = []
    try:
        v = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(v, int):
            ids.append(int(v))
        elif isinstance(v, (list, tuple)):
            ids.extend(int(x) for x in v if isinstance(x, int))
    except Exception:
        pass
    try:
        tok = getattr(processor, "tokenizer", processor)
        for name in _EOS_CANDIDATES:
            tid = tok.convert_tokens_to_ids(name)
            if not isinstance(tid, int):
                continue
            # 往返校验：只有真的能还原成这个名字，才是有效特殊 token
            if tok.convert_ids_to_tokens(tid) == name:
                ids.append(int(tid))
    except Exception:
        pass
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


# ---------------------------------------------------------------------------
# 显存 / 设备探针
#
# 这是本项目最贵的坑：显卡空闲显存不够时 device_map="auto" 不报错，而是
# 静默把一部分层（常常正好是视觉塔）放到 CPU 上。于是 4bit 8B 在 4090D 上
# 从正常的 40~80 tok/s 掉到 1 tok/s —— 日志里只看到"慢"，看不到原因。
# 所以加载前后各探测一次，并把设备分布写进日志。
# ---------------------------------------------------------------------------

# 各量化档位大致需要的显存（含视觉塔与 KV 余量），单位 MiB
_VRAM_NEED_MIB = {"none": 19000, "8bit": 11000, "4bit": 7000}

# 量化时必须「排除」的模块。
#
# 1) 视觉塔一定要排除。它每层前向都要实时反量化 4bit 权重，实测把图片 prefill
#    从几秒拖到 57~86 秒（占整轮 79%），而它本身只有约 430M 参数，
#    bf16 也不过 0.9GiB —— 为省这点显存付出的代价完全不值。
# 2) lm_head 是 transformers 原本的默认保护项。这里必须手动带上：
#    一旦传了 llm_int8_skip_modules，get_modules_to_not_convert() 就走 else
#    分支、**不再**自动追加默认跳过项，默认的 lm_head 保护会丢。
#    匹配规则见 transformers/quantizers/quantizers_utils.py:37
#    （前缀加点 / 正则前缀 / endswith 三种），用 "model.visual" 前缀最可靠。
# 3) embed_tokens 是 nn.Embedding，本就不在被替换之列，无需列出。
# 4) in_proj_a / in_proj_b 是 Qwen3.5 线性注意力层里的 **Linear(4096, 32)** ——
#    每个只有 131k 参数（int8 后 131KB），但 24 层合计每 token 要调用 48 次。
#    受控对照实测（bench_qwen35_proj_ab.py：同一批形状，bf16 与候选背靠背交替、
#    各取最小值）：**调用次数多、单个矩阵小的那一组，bnb 的固定单次开销占绝对主导**
#    —— 全注意力投影组只有 0.34B 参数，int8 却慢 7.21x（batch=1）。
#    排除这两个小投影只多占约 12.6MB 显存（48×131k×2B），换来去掉 48 次纯开销调用。
#    收益数字属按上述对照外推，未单独实测。
_QUANT_SKIP_MODULES = ("model.visual", "visual", "lm_head", "in_proj_a", "in_proj_b")

# ---------------------------------------------------------------------------
# bnb int8 的关键开关：llm_int8_threshold
#
# 读 bnb 0.49.2 源码（bitsandbytes/autograd/_functions.py:MatMul8bitLt.forward）
# 可见这是一个**二选一的算子分派**，不是数值参数：
#     if state.threshold > 0.0:
#         output, subA = int8_mixed_scaled_mm(A, CA, state.CB, SCA, state.SCB, outlier_cols, bias)
#     else:
#         output = int8_scaled_mm.default(CA, state.CB, SCA, state.SCB, bias=bias, dtype=A.dtype)
# transformers 默认给 6.0 → 永远走上面那条（混合内核）。
#
# 受控对照实测（相对 bf16，batch=1 / batch=256 两个比值）：
#     组                        int8 t=6.0     int8 t=0.0
#     线性注意力投影×24(120次)    5.62x/3.45x    3.23x/1.82x
#     全注意力投影×8 (32次)      7.21x/4.30x    4.18x/2.16x
# → 改成 0.0 稳定快 1.7~1.9 倍。而且是**固定的算子选择收益**，不是数据相关：
#   即使一列离群都挑不出来，mixed 内核本身就更贵。
#
# 代价：0.0 之后离群列不再单独用 fp16 计算，改由「按列 absmax 缩放」承担，
# 该列其余数值精度下降 —— 也就是主动放弃了 LLM.int8() 论文里做无损的那部分。
# 对提示词扩写可接受；要保守可给个较大的正数（如 20.0）。
# transformers 只校验它是 float（quantization_config.py:530），0.0 合法；
# 传递位点在 integrations/bitsandbytes.py:196。
_INT8_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# 要省显存就用 4bit，不要用 8bit（2026-10-07 受控对照结论）
#
# 同一批形状、bf16 与候选变体背靠背交替计时、各取最小值：
#
#   MLP×32（96 次调用，4.83B 参数，占参数大头）
#       bf16     10.52 ms (855 GB/s)   ← 已经在带宽极限附近
#       4bit nf4  5.15 ms (437 GB/s)   → **0.49x，比 bf16 快一倍**
#     权重字节只剩 1/4，即便内核效率更低，净值仍然赢。
#     int8 做不到：t=0.0 约 0.95x、t=6.0 约 1.64x（同法测得）。
#
#   小矩阵那两组（线性注意力投影、全注意力投影）4bit 是 1.95~2.39x，
#   int8 是 3.23~7.21x —— 都是「调用次数多、矩阵小」，固定开销主导。
#
# 结论：显存不够时 4bit（实测整轮 54.06s vs bf16 51.85s = 1.04x、驻留 7.9GiB
# vs 17.5GiB）是明显更优的档位 —— 这条外推已在 2026-10-07 被真机 A/B 证实
# （第十三章）。8bit 只在已经选了它、又不想重载模型时，才值得把 threshold 调到 0。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 解码速度体检阈值 —— 以及「8bit 反而更慢」这个反直觉的事实
#
# 本机（4090D 24GB）实测，同一模型、同一参数，只改量化档：
#     Qwen3.5-9B   none 18.5~21.6 tok/s | 4bit 17.3~19.7 | 8bit 2.3~13.1
#     Qwen3-VL-8B  none 22.6~23.4 tok/s | 8bit 7.0~7.1
#
# 8bit 那个 2.3~13.1 横跨 5.7 倍：低端是 threshold=6.0 时代的旧值，改成 0.0 之后
# （本节点已生效）稳定在 12.4~13.1 tok/s —— 改善明显，但仍追不上 4bit/bf16。
# 更要紧的是整轮 A/B 的结论：8bit 相对 4bit 多花的 17s 里 **84% 是加载**
# （33.90s vs 19.60s），解码只占小头。所以「8bit 慢」慢在两处，别只盯解码。
# 也就是说 bnb 的 LLM.int8() 在这台机器上仍是**净负收益**：省下 6.4GiB 显存，
# 代价是加载慢 73% + 解码慢 1.4~1.6 倍。
#
# 根因（2026-10-07 受控对照实测，bench_qwen35_proj_ab.py）三条，按重要性：
#   ① **bf16 已经没有余量可省**：batch=1 时 bf16 的 Linear 实测 754~855 GB/s，
#      4090D 峰值约 1008 GB/s —— 权重流本来就接近极限。int8 砍一半字节，
#      理论上限也就 2x，而实际内核效率只有 44~117 GB/s，反而更差。
#   ② **调用次数太多、矩阵太小**：Qwen3.5-9B 每 token 约 248 次 Linear 调用
#      （24 层线性注意力 ×5 + 8 层全注意力 ×4 + 32 层 MLP ×3），
#      bnb 每次都有固定开销。最典型的是全注意力投影组：只有 0.34B 参数，
#      int8 却慢 7.21x —— 慢的全是开销，不是数据量。
#   ③ **混合内核额外更贵**：threshold>0 时走 int8_mixed_scaled_mm，
#      改成 0.0 后稳定快 1.7~1.9 倍（已在本节点生效）。
#      （注意"离群列是否真的触发"不是主因：阈值 6.0 是绝对值，
#       post-RMSNorm 激活 RMS≈1，batch=1 时几乎挑不出离群列，内核仍贵。）
# 还错判过一次：拿「带宽利用率只有 37%」推出「8bit 应该更快」，方向正好反了
#    —— 利用率低恰恰说明瓶颈不在带宽。
#
# 所以默认档位是 none。要省显存请用 **4bit**（真机 A/B：整轮 54.06s vs bf16
# 51.85s = 1.04x，驻留 7.9GiB vs 17.5GiB），不要用 8bit。
#
# 体检线：9B/8B 的正常值约 18~22 tok/s，4bit 实测 17.3/19.7 —— 留余量设 14.0。
# 2026-10-07 从 12.0 上调：threshold=0.0 之后 8bit 实测 12.4 tok/s，正好卡在旧线
# 之上、体检直接哑火；14.0 落在「4bit 最低 17.3」与「8bit 最高 13.1」之间的空档，
# 既不误报 4bit、又能让 8bit 每次都触发。
_SLOW_DECODE_TOK_S = 14.0
_SLOW_PREFILL_S = 30.0

# 本次加载的实测状态，供解码体检做分层归因（每次 _load 刷新）
_CPU_OFFLOAD_GIB = 0.0     # 有多少权重落在 CPU；>0 就是硬故障，优先修它
_CUR_QUANT = "none"        # 本次实际用的量化档

# ---------------------------------------------------------------------------
# 注意力后端：**不是** prefill 慢的原因（此处曾误判，2026-10-07 已推翻）
#
# 早先的判断是：Qwen3-VL 的视觉塔只有走 flash 路径时才用 cu_seqlens 一次性
# 变长处理，否则退化成「逐图切块 + Python 循环」（modeling_qwen3_vl.py:223 的
# else 分支），实测每视觉 token 约 33ms —— 由此推断缺 flash-attn 是 prefill
# 慢的根因。
#
# **这个结论被受控实验推翻了。** 同一份权重、同一份输入，只换视觉塔注意力后端：
#     eager 29.16s / sdpa 28.91s / flash_attention_2 29.30s —— 三者毫无差别。
# 真正的瓶颈是视觉塔入口 patch_embed 的 Conv3d 在 fp16/bf16 下病态
# （fp32 6.5ms，half 超过 45s 不返回，等价 GEMM 0.1ms）。修掉之后 prefill
# 从 75.6s 降到 1.59s，与注意力后端无关。
#
# 那这个开关还留着干什么：
#   * flash-attn 本身对 LLM 长上下文省显存、也略快，装了就用（默认 auto）；
#   * auto 在未装 flash-attn 时退回 sdpa，与手选 sdpa 等价，不会有副作用。
# 但**不要**指望它改善图片 prefill。
#
# 设置方式：attn_implementation 置成 flash_attention_2 时，PretrainedConfig
# 的 setter 会**自动递归**写到 vision_config / text_config
# （configuration_utils.py:379-387），视觉塔跟着切，不需要手工传播。
# ---------------------------------------------------------------------------
_FA_MIN_VERSION = (2, 3, 3)          # transformers 只认 >= 2.3.3
_ATTENTION_CHOICES = ("auto", "flash_attention_2", "sdpa", "eager")

# 视觉塔的「每视觉 token 毫秒」体检线。本机实测：
#   入口健康（patch_embed 走 GEMM）时，每 token 应在 1ms 量级；
#   入口走原始 Conv3d 时是 30~40 ms/token。
_SLOW_VISION_PER_TOKEN_MS = 5.0

# ---------------------------------------------------------------------------
# 线性注意力的融合内核体检线（2026-10-07 新增）
#
# Qwen3.5 是**混合架构**：32 层里 24 层是 Gated DeltaNet 线性注意力，
# 只有 8 层是全注意力（full_attention_interval=4）。
#
# transformers 5.9.0 的 modeling_qwen3_5.py 里，每个线性注意力层实例上挂的是：
#     self.chunk_gated_delta_rule     = chunk_gated_delta_rule     or torch_chunk_...
#     self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_...
# 前两个来自 fla（flash-linear-attention）。fla 没装时四个符号全为 None，
# decode 就落到 torch_recurrent_gated_delta_rule —— 实现是：
#     [x.transpose(1,2).contiguous().to(torch.float32) for x in (...)]
#     for i in range(sequence_length): ...
# 即 fp32 + 逐个 token + 一串小张量算子。
#
# 本机 4090D 实测（同权重同形状，seq=1 decode）：
#     torch 回退 : 1.009 ms/层  × 24 = 24.21 ms/token   → 占解码 50.6%
#     fla 融合   : 0.659 ms/层  × 24 = 15.81 ms/token
# 单看 gated delta rule 本身（裸算子）：0.321 ms → 0.056 ms，5.8×。
# 等效带宽只有峰值 1008GB/s 的 1.2% —— 完全是"小算子太多"的延迟问题，
# 不是算力也不是带宽问题。所以**调 max_image_side / 换量化都救不了它**。
#
# 注意 is_fast_path_available 只控制 transformers 自己那条 warning，
# 不决定用哪份实现；判定必须看**已加载模型实例上挂的函数来自哪个模块**。
# ---------------------------------------------------------------------------
# 低于这个速率才顺带提线性注意力。9B bf16 在 4090D 上正常 18.5~21.6 tok/s，
# 所以线要划在正常值**之下**（15），否则正常跑完也会刷这条提示，变成噪音。
# 另有一条互斥规则见 _decode_notes：判为 8bit 主因时这条不再出现。
_LINEAR_ATTN_SLOW_DECODE_TOK_S = 15.0
_LIN_ATTN_FUSED = None                  # 当前模型里走融合内核的层数
_LIN_ATTN_TOTAL = None                  # 当前模型里线性注意力层总数


def _flash_attn_version():
    """返回已安装 flash-attn 的版本元组；没装或版本串读不出来时返回 None。"""
    try:
        import flash_attn
    except Exception:
        return None
    raw = str(getattr(flash_attn, "__version__", "") or "")
    nums = []
    for part in raw.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        nums.append(int(digits))
    return tuple(nums) if len(nums) >= 2 else None


def _flash_attn_available():
    """装没装可用的 flash-attn（版本要 >= 2.3.3，否则 transformers 不认）。"""
    v = _flash_attn_version()
    return v is not None and tuple(v[:3]) >= _FA_MIN_VERSION


def _resolve_attention(requested):
    """把界面上选的后端解析成真正传给 transformers 的值。

    返回 (后端名, 说明)。"auto" 在装了合规 flash-attn 时选 flash_attention_2，
    否则退回 sdpa —— 这样卸载 flash-attn 也不会把已有工作流跑挂。
    """
    req = str(requested or "auto").strip().lower()
    if req != "auto":
        return req, ""
    v = _flash_attn_version()
    if v is not None and tuple(v[:3]) >= _FA_MIN_VERSION:
        return "flash_attention_2", f"auto → flash_attention_2（flash-attn {'.'.join(map(str, v))}）"
    if v is None:
        return "sdpa", "auto → sdpa（未装 flash-attn）"
    return "sdpa", f"auto → sdpa（flash-attn {'.'.join(map(str, v))} 低于要求的 2.3.3）"


# ---------------------------------------------------------------------------
# 视觉塔入口的 Conv3d：本机实测的头号瓶颈（比注意力重要得多）
#
# [Qwen3-VL / Qwen3.5] 的 patch_embed 都是 Conv3d(3 -> 1152, kernel=stride=(2,16,16))，
# 输出空间尺寸恒为 1x1x1 —— 数学上就等于一个 (N,1536)@(1536,1152) 的 GEMM，
# 理论约 12 GFLOP，4090D 上应该是 10ms 量级。
#
# 但在本机（torch 2.9.1+cu130 / cuDNN 9.12 / Windows）实测：
#     Conv3d fp32 : 首次 0.06s，稳态 6.5ms      正常
#     Conv3d fp16 : 首次调用超过 45s 都没返回    病态
#     Conv3d bf16 : 同上
# 而视觉塔（27 层、几千个 patch token）整段 29s，其中约 96% 卡在这个卷积上。
# 同一份权重、同一份输入下换 eager / sdpa / flash_attention_2 三种注意力后端，
# 耗时分别是 29.16 / 28.91 / 29.30 s —— 完全没差别，所以「缺 flash-attn 导致
# 视觉塔逐图切块」这个更早的判断是错的，flash-attn 解决不了这里的慢。
#
# 因此把 patch_embed 的前向换成 F.linear：权重只做 reshape 视图、不复制，
# 数学完全等价（kernel==stride => 输出 1x1x1 => 就是 GEMM），只是绕开 cuDNN。
#
# 【2026-10-07 补丁 v2】必须打到「已加载模型的模块实例」上，不能只按类名打到类上。
#   Qwen3-VL  的类叫 Qwen3VLVisionPatchEmbed（transformers.models.qwen3_vl）
#   Qwen3.5   的类叫 Qwen3_5VisionPatchEmbed（transformers.models.qwen3_5）
#   两者源码逐字相同（depth 27 / hidden 1152 / patch 16 / temporal 2，Conv3d
#   kernel==stride、padding 0、groups 1），但**类与模块都不同**。
#   只补 Qwen3-VL 那个类，一旦换成 Qwen3.5 补丁就静默失效：日志照报「已换成 GEMM」，
#   prefill 却从 0.69s 退回 75.61s（35.1 ms/视觉token，正是没打补丁时的水平）。
#   所以现在按「结构」认，而不是按类名认：
#       遍历模型里每个模块 -> 类名含 patchembed -> .proj 是 kernel==stride 的 Conv3d
#   换模型族（Qwen3.5 / MoE / 以后的新版）都不会再漏。
# ---------------------------------------------------------------------------
_PATCH_EMBED_PATCHED = False      # 是否已成功给某个 patch_embed 类打过补丁
_PATCH_EMBED_INSTANCES = 0        # **当前模型**里已替换的 patch_embed 实例数（每次加载刷新）

# 已知的 patch_embed 类。用途是覆盖「还没建出实例」的场景（单元测试、提前打补丁），
# 以及给实例级探测做交叉印证。**不是**唯一的覆盖手段 —— 实例级扫描才是主力。
_PATCH_EMBED_CLASS_TARGETS = (
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5VisionPatchEmbed"),
    ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeVisionPatchEmbed"),
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeVisionPatchEmbed"),
)


def _patch_embed_disabled():
    """QWEN35_NO_PATCH_EMBED_FIX 非空且非 0 时跳过补丁（排查用）。"""
    return os.environ.get("QWEN35_NO_PATCH_EMBED_FIX", "").strip() not in ("", "0")


def _is_gemm_equivalent_conv3d(proj):
    """判断这个 Conv3d 是否数学等价于一次 GEMM（输出恒为 1x1x1）。

    kernel==stride 且无补零、无孔洞、单组时，滑窗只会正好覆盖一次，
    输出尺寸 = 1x1x1，卷积退化成一次矩阵乘。四个条件缺一不可，
    否则替换后数值就对不上了。
    """
    if not isinstance(proj, torch.nn.Conv3d):
        return False
    return (
        tuple(proj.kernel_size) == tuple(proj.stride)
        and tuple(proj.dilation) == (1, 1, 1)
        and tuple(proj.padding) == (0, 0, 0)
        and proj.groups == 1
    )


def _gemm_patch_embed_forward(self, hidden_states):
    """等价替换 *VisionPatchEmbed.forward（Conv3d -> GEMM）。

    原实现：
        hidden_states.view(-1, C, T, P, P) -> Conv3d -> .view(-1, embed_dim)
    kernel == stride 时输出恒为 1x1x1，展开到矩阵形式就是
        out = x_flat @ W.reshape(out_dim, -1).T + b
    权重与输入都是同一块内存的视图，没有复制、没有改数值。
    入参无论是 (N, C*T*P*P) 还是 (N, C, T, P, P)，reshape(n, -1) 都成立。
    """
    proj = self.proj
    target_dtype = proj.weight.dtype
    n = hidden_states.shape[0]
    x = hidden_states.reshape(n, -1)
    w = proj.weight.reshape(proj.out_channels, -1)
    # 展平后维度必须对得上，否则说明这个模块的输入布局不是 (...,C,T,P,P)。
    # 与其静默算出错误结果，不如直接报出来 —— 实例级扫描是按结构认的，
    # 万一撞上同名同构但布局不同的模块，这里会第一时间暴露。
    if x.shape[1] != w.shape[1]:
        raise RuntimeError(
            f"[Qwen35] patch_embed GEMM 维度不匹配：输入展平 {x.shape[1]} vs "
            f"权重展平 {w.shape[1]}，该模块输入布局不是 (..., C, T, P, P)。"
            "请设 QWEN35_NO_PATCH_EMBED_FIX=1 跳过此优化并反馈。"
        )
    if x.dtype != target_dtype:
        x = x.to(dtype=target_dtype)
    b = proj.bias
    if b is not None and b.dtype != target_dtype:
        b = b.to(dtype=target_dtype)
    return torch.nn.functional.linear(x, w, b)


def _patch_patch_embed_classes():
    """按已知类名给「类」打补丁，返回成功个数。"""
    hit = 0
    for mod_name, cls_name in _PATCH_EMBED_CLASS_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue                      # 该模型族这个 transformers 版本里不存在，正常
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        try:
            cls.forward = _gemm_patch_embed_forward
            hit += 1
        except Exception as e:
            logger.warning(f"[Qwen35] {cls_name} 换 GEMM 失败，将走原始 Conv3d: {e}")
    return hit


def _patch_patch_embed_instances(root):
    """遍历已加载模型，把每个 patch_embed 实例的 forward 换成 GEMM。

    按结构识别而不是按类名：换模型族时类名和模块路径都会变，硬编码必漏。

    返回**该模型里符合条件的实例总数**，而不是"本次新替换了几个"——
    这样同一个模型被幂等重入时也能拿到正确计数，不会因为累加而虚高。
    """
    hit = 0
    for mod in root.modules():
        if "patchembed" not in type(mod).__name__.lower():
            continue
        if not _is_gemm_equivalent_conv3d(getattr(mod, "proj", None)):
            continue
        hit += 1
        if getattr(mod, "_qwen35_gemm_patch_embed", False):
            continue                       # 已经打过了，只计数不重复替换
        mod.forward = types.MethodType(_gemm_patch_embed_forward, mod)
        mod._qwen35_gemm_patch_embed = True
    return hit


def _patch_vision_patch_embed(model=None):
    """把视觉塔 patch_embed 换成等价 GEMM；返回是否为「本次新打上」。

    两层保险：
      * 类级 —— 覆盖已知的 Qwen3-VL / Qwen3.5 / MoE patch_embed 类；
      * 实例级 —— 遍历已加载模型按结构识别（**主力**，换模型族也不会漏）。
    只替换 forward，参数名与 state_dict 键名都不动，与官方权重完全兼容。
    设 QWEN35_NO_PATCH_EMBED_FIX=1 可跳过（排查用）。
    """
    global _PATCH_EMBED_PATCHED, _PATCH_EMBED_INSTANCES
    if _patch_embed_disabled():
        logger.info("[Qwen35] QWEN35_NO_PATCH_EMBED_FIX 已设，保留原始 Conv3d patch_embed")
        return False

    first = not _PATCH_EMBED_PATCHED
    n_cls = _patch_patch_embed_classes()
    if n_cls:
        _PATCH_EMBED_PATCHED = True

    n_inst = 0
    if model is not None:
        n_inst = _patch_patch_embed_instances(model)
        # 赋值而不是累加 —— 这个计数代表「当前加载的这个模型」，
        # 换模型后必须刷新，否则会带着上一个模型的数字虚高。
        _PATCH_EMBED_INSTANCES = n_inst

    if _PATCH_EMBED_PATCHED:
        logger.info(
            f"[Qwen35] patch_embed 换 GEMM：类级 {n_cls} 个 / 本模型实例 {n_inst} 个"
            "（原始 Conv3d 在本机 fp16/bf16 下首次调用超过 45s，fp32 只要 6.5ms）"
        )
    return first

# 本进程内每种输入组合跑了第几次。首次含 kernel 编译 / CUDA 预热，
# prefill 会明显偏慢，日志里要区分开，免得把「首次」误判成「有病」。
_RUN_COUNT = {}

# 视觉塔的属性路径（不同 transformers 版本命名不一，逐个试）
_VISION_PATHS = ("visual", "model.visual", "vision_model", "model.vision_tower")


def _fmt_gib(n_bytes):
    """字节数转成可读单位：GiB / MiB / KiB（自适应）。"""
    try:
        n = float(n_bytes)
    except Exception:
        return "?"
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f}GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f}MiB"
    return f"{n / 1024:.0f}KiB"


def _first_int(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]) if v else 0
    if isinstance(v, (int, float)):
        return int(v)
    return 0


def _probe_memory(mm_mod):
    """返回 (可用显存, 总显存, 可用内存) 三个字节数；拿不到的项为 0。

    可用内存取的是 ComfyUI 的 virtual_memory_available（含页面文件），
    因为真正致慢的往往就是内存换页而不仅是物理内存不足。
    """
    free_vram = total_vram = free_ram = 0
    try:
        if mm_mod is not None and hasattr(mm_mod, "get_free_memory"):
            free_vram = _first_int(mm_mod.get_free_memory(torch_free_too=True))
            free_ram = _first_int(
                mm_mod.get_free_memory(torch.device("cpu"), torch_free_too=True)
            )
    except Exception:
        pass
    try:
        # 用 is_initialized 而不是 is_available：后者会顺带把 CUDA 上下文建起来，
        # 在只跑单元测试（假模型）的进程里不该占显存。
        if torch.cuda.is_initialized():
            _, total_vram = torch.cuda.mem_get_info()
    except Exception:
        pass
    return free_vram, total_vram, free_ram


def _device_map_summary(model):
    """把 model.hf_device_map 汇总成 {'cuda:0': 32, 'cpu': 12} 这样的计数。"""
    dm = getattr(model, "hf_device_map", None)
    if not isinstance(dm, dict) or not dm:
        return None
    counts = {}
    for dev in dm.values():
        if isinstance(dev, int):
            dev = f"cuda:{dev}" if dev >= 0 else "cpu"
        else:
            dev = str(dev)
        counts[dev] = counts.get(dev, 0) + 1
    return counts


def _param_device_stats(model, module=None):
    """按「参数量」统计 device / dtype 分布，返回 (device字节, dtype字节)。

    为什么不能只看 hf_device_map: 新版 transformers 在 4bit 量化 + 单卡时
    往往**根本不写** hf_device_map，于是只看它的探针会整段跳过 —— 实测就
    是这样，日志里「模型放置」那行从来没出现过，只剩视觉塔的单点读数。
    直接遍历参数最可靠：只要 CPU 上出现非零字节，就是真的有层没上卡。
    """
    from collections import defaultdict
    dev_b, dt_b = defaultdict(int), defaultdict(int)
    try:
        params = list((module if module is not None else model).parameters())
    except Exception:
        return None, None
    total = 0
    for p in params:
        try:
            nbytes = int(p.numel()) * max(int(p.element_size()), 1)
            dev_b[str(p.device)] += nbytes
            dt_b[str(p.dtype).replace("torch.", "")] += nbytes
            total += nbytes
        except Exception:
            continue
    if not total:
        return None, None
    return dict(dev_b), dict(dt_b)


def _fmt_stat(d):
    """把 {'cuda:0': 12345, 'cpu': 67} 排成 'cuda:0 6.42GiB  cpu 67KiB'。"""
    if not d:
        return ""
    return "  ".join(f"{k} {_fmt_gib(v)}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def _vision_module(model):
    """返回视觉塔模块对象；找不到返回 None。"""
    for path in _VISION_PATHS:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _count_visual_tokens(inputs):
    """从 processor 输出里数视觉 token 数。

    Qwen3-VL: 图像按 32x32 像素出 1 个 token，即 prod(grid_thw) / spatial_merge²
    （见 modeling_qwen3_vl.py 里 split_sizes 的算法）。拿不到就返回 0。
    """
    try:
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return 0
        return int((grid.prod(-1) // 4).sum())
    except Exception:
        return 0


def _decode_speed_warning(out_tok, dec_s, quantization=None):
    """解码速度异常慢时返回一句诊断；正常或样本太小时返回空串。

    归因按「最该先修」分三层，不再把所有慢都算到 CPU 摊派头上：
      1) 真有层落在 CPU（_CPU_OFFLOAD_GIB > 0）→ 硬故障，先腾显存；
      2) 层全在 GPU 但用了 bnb 量化 → 本机实测这个档位自身就是主因；
      3) 两者都不是 → 如实说「原因不在这两处」，不瞎猜。
    第 3 条是刻意留的：宁可承认不知道，也不给一个听起来合理但指错方向的说法。
    """
    if out_tok < 16 or dec_s <= 0:
        return ""
    rate = float(out_tok) / float(dec_s)
    if rate >= _SLOW_DECODE_TOK_S:
        return ""
    q = str(quantization if quantization is not None else _CUR_QUANT)
    if _CPU_OFFLOAD_GIB > 0:
        return (
            f"  ⚠ 解码只有 {rate:.1f} tok/s：本模型有 {_CPU_OFFLOAD_GIB:.1f}GiB 权重"
            f"被摊到 CPU 上（见上面「权重分布」行）。这是硬故障 —— 先腾出显存再跑。"
        )
    if q == "8bit":
        return (
            f"  ⚠ 解码只有 {rate:.1f} tok/s：层都装进显卡了、不是显存问题，"
            f"主因就是 quantization=8bit 本身（本机实测：none 18.5~21.6 tok/s、"
            f"4bit 17.3~19.7 tok/s、8bit 12.4~13.1 tok/s）。\n"
            f"     根因是 bnb 的 int8 路径在这台机器上效率太低：batch=1 时 bf16 的 "
            f"Linear 已经跑到 754~855 GB/s（4090D 峰值约 1008 GB/s），权重流早就接近极限，"
            f"int8 省下的字节无处兑现；而 bnb 的 int8 内核实测只有 44~117 GB/s，"
            f"模型每 token 还有约 248 次小矩阵 Linear 调用，固定开销占绝对主导。\n"
            f"     本节点已把 llm_int8_threshold 设为 {_INT8_THRESHOLD}（走纯 int8_scaled_mm "
            f"而非混合内核），能让 int8 内部快 1.7~1.9 倍，但仍追不上 bf16。\n"
            f"     ⚠ 还有一半代价不在解码、在**加载**（真机 A/B：8bit 33.90s vs "
            f"4bit 19.60s）—— 只优化解码是治不好的。\n"
            f"     → 要省显存请改用 quantization=4bit（真机 A/B：整轮 54.06s vs "
            f"bf16 51.85s、只慢 4%，驻留 7.9GiB vs bf16 17.5GiB）；显存够就直接 none。"
        )
    if q == "4bit":
        return (
            f"  ⚠ 解码只有 {rate:.1f} tok/s：quantization=4bit 被选中，但 4bit 在同机真机 A/B 里"
            f"整轮只比 bf16 慢 4%（54.06s vs 51.85s）、解码实测仍有 17.3~19.7 tok/s —— "
            f"所以 4bit 本身解释不了这么慢，原因不在这处。请对照上面「权重分布」行："
            f"若全部落在 cuda 上，需要进一步实测定位。"
        )
    return (
        f"  ⚠ 解码只有 {rate:.1f} tok/s：层全在 GPU 上、也没用量化，"
        f"说明慢的原因不在已排查的这几处（CPU 摊派 / 量化档 / 线性注意力回退）。"
        f"需要进一步实测定位；请不要据此去调 max_image_side —— 那与本项无关。"
    )


def _scan_linear_attn_fast_path(model):
    """按**已加载模型的实例**判定线性注意力走的是融合内核还是 torch 回退。

    为什么不能只看 transformers 的 import 开关：
        modeling_qwen3_5.py 里 `is_fast_path_available` 只控制它自己那条
        warning，真正决定用哪份实现的是每个层实例上的
            self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_...
        所以必须看实例上挂的函数来自哪个模块 —— 与 patch_embed 那次
        「补丁打在了别的类上却报成功」是同一类教训。

    返回 (走融合内核的层数, 线性注意力层总数)；都不是线性注意力层时返回 (0, 0)。
    """
    global _LIN_ATTN_FUSED, _LIN_ATTN_TOTAL
    fused = total = 0
    try:
        for mod in model.modules():
            fn = getattr(mod, "recurrent_gated_delta_rule", None)
            if fn is None:
                continue
            total += 1
            if str(getattr(fn, "__module__", "")).split(".")[0] == "fla":
                fused += 1
    except Exception:
        return 0, 0
    _LIN_ATTN_FUSED, _LIN_ATTN_TOTAL = fused, total
    return fused, total


def _linear_attn_decode_note(out_tok, dec_s):
    """解码偏慢、且线性注意力走 torch 回退时给出定量提示；否则返回空串。"""
    if out_tok < 16 or dec_s <= 0 or not _LIN_ATTN_TOTAL:
        return ""
    if _LIN_ATTN_FUSED == _LIN_ATTN_TOTAL:
        return ""
    rate = float(out_tok) / float(dec_s)
    if rate >= _LINEAR_ATTN_SLOW_DECODE_TOK_S:
        return ""
    return (
        f"  ℹ 解码 {rate:.1f} tok/s：本模型有 {_LIN_ATTN_TOTAL} 层线性注意力"
        f"（Gated DeltaNet），其中 {_LIN_ATTN_TOTAL - (_LIN_ATTN_FUSED or 0)} 层走的是 "
        f"transformers 自带的 torch 回退（fp32 + 逐个 token + 小张量串行）。"
        f"本机实测回退 1.0 ms/层、融合内核 0.66 ms/层，24 层即占解码约一半；"
        f"装 fla（flash-linear-attention，纯 Python wheel、无需编译）即换成融合 Triton "
        f"内核，约省 1/3 解码时间。"
    )


def _decode_notes(out_tok, dec_s, quantization=None):
    """解码后的两条体检，带一条互斥规则，返回要追加到耗时分解里的行。

    两条各自独立、平时互不干扰：
      * `_decode_speed_warning`   —— 按「CPU 摊派 → 量化档 → 原因未知」归因
      * `_linear_attn_decode_note`—— 线性注意力走 torch 回退

    互斥规则（2026-10-07 真机踩到）：8bit 实测 12.4 tok/s 会**同时**低于两条线
    （速度线 14.0 / 线性注意力线 15.0），于是日志会一边说"主因是 8bit"、
    一边建议"装 fla 省 1/3 解码"。后者不算错，但收益量级差太远
    （fla ≈ +2 tok/s，换 4bit ≈ +5 tok/s 且显存顺带砍一半），
    在已经点明主因时再给这条建议，等于把人往次优解上带。所以主因是 8bit 时压掉它。
    """
    warn = _decode_speed_warning(out_tok, dec_s, quantization)
    notes = [warn] if warn else []
    if not (warn and str(quantization) == "8bit"):
        note = _linear_attn_decode_note(out_tok, dec_s)
        if note:
            notes.append(note)
    return notes


def _linear_attn_load_note(fused, total):
    """加载完成时报告线性注意力快路径状态；不是混合架构模型时返回 None。"""
    if not total:
        return None
    if fused == total:
        return (
            f"[Qwen35] 线性注意力：{total} 层全部走融合内核（fla 已生效）"
        )
    extra = f"（{fused} 层已用融合内核）" if fused else ""
    return (
        f"[Qwen35] 线性注意力：{total} 层里 {total - fused} 层走 torch 回退{extra} —— "
        f"这些层在解码时逐 token 跑 fp32 小算子，实测 1.0ms/层（融合内核 0.66ms/层），"
        f"24 层即占解码约一半。装 fla 可换掉：pip install flash-linear-attention"
        f"（纯 Python wheel，无需编译；装完重启 ComfyUI 即生效）"
    )


def _vision_device(model):
    """找出视觉塔所在的设备字符串；找不到返回 None。"""
    obj = _vision_module(model)
    if obj is None:
        return None
    try:
        return str(next(obj.parameters()).device)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 进度上报
#
# 慢的根因是"加载模型"和"逐 token 生成"，所以进度要同时给两条出口：
#   1) 前端 —— comfy.utils.ProgressBar(node_id=...) -> main.py 的 PROGRESS_BAR_HOOK
#      -> websocket "progress" 事件 -> 前端在这个节点上画进度条。
#   2) 控制台 —— 阶段切换即时打一行；生成阶段按 interval 秒打一行。
#      如果 stdout 是真实终端，还会用 \r 原地刷新一条实时进度条。
#
# 全程静默降级：任何一步失败都只是没有进度条，绝不影响出结果。
# ---------------------------------------------------------------------------

# 各阶段在整个进度条上的权重（合计 100）。
# 加载占大头是故意的 —— 用户反馈的时间大头就在加载，进度条必须如实反映。
_W_UNLOAD, _W_LOAD, _W_PREP = 2.0, 33.0, 3.0
_W_GEN = 100.0 - _W_UNLOAD - _W_LOAD - _W_PREP   # = 62.0

# 双语时"英文生成"段的先验宽度（只作为下限锚点）。
# 中文 token 数其实比英文多（同样的内容约 1.6 倍 token），所以真正的分界点
# 必须等第一段跑完、拿到英文实际输出长度后按实测重算（见 enhance() 里的
# pbar.rescale）。这个先验取得偏小，保证重算时进度只会往前跳、不会回退。
_EN_SPAN_PRIOR = 0.35

# ---------------------------------------------------------------------------
# 中文预览的 token 预算
#
# 实测（h3_workflow/measure_zh_token_ratio.py，本机 Qwen3-VL 分词器 + 三组
# 真实 H3 提示词）：忠实的中文译文只要英文的 1.07~1.12 倍 token。
# 中文字符数虽然只有英文的 0.41 倍，但每个中文 token 只装 1.53 个字，
# 而英文每个 token 能装 4.14 个字符，两者大致抵消。
#
# 所以双语正常只该多花约 20%，不是 3 倍。预算取 1.6 倍 + 64 是刻意留 ~45%
# 余量：正常翻译永远摸不到它（因此不产生任何额外成本），但一旦模型开始
# "先把英文原文抄一遍再翻译"——那种情况第二段要解码约 2 倍 token，正是
# 耗时翻到 3 倍的原因——预算会在 1.6 倍处截停，而不是一路跑满 max_new_tokens。
# ---------------------------------------------------------------------------
_ZH_TOK_RATIO = 1.6
_ZH_BUDGET_EXTRA = 64    # 余量，容纳对白释义等额外内容
_ZH_BUDGET_MIN = 96      # 英文极短时也要给足预算


def _zh_budget(en_tokens, hard_cap):
    """按英文实际输出长度换算中文段的 token 预算。

    永远不超过用户设的 max_new_tokens（hard_cap）：这是用户明确表达的上限，
    不能被内部估算顶掉。英文还没出结果（en_tokens<=0）时才退回 hard_cap。
    """
    cap = max(1, int(hard_cap))
    if int(en_tokens) <= 0:
        return cap
    want = max(_ZH_BUDGET_MIN, int(int(en_tokens) * _ZH_TOK_RATIO) + _ZH_BUDGET_EXTRA)
    return max(1, min(cap, want))

_UNICODE_BAR = True   # 控制台是否允许用方块字符画条（GBK 终端会失败并自动回落 ASCII）


class _ProgressReporter:
    """把一个长任务（卸载/加载/预处理/生成）映射成 0~100 的进度。"""

    TOTAL = 100.0

    def __init__(self, node_id=None, enabled=True, interval=2.0):
        self.enabled = bool(enabled)
        self.interval = max(0.2, float(interval or 2.0))
        self.node_id = node_id
        self.value = 0.0
        self.base = 0.0          # 当前阶段在 0~100 上的起点
        self.span = 0.0          # 当前阶段占的宽度
        self.pbar = None
        self._t0 = time.perf_counter()
        self._stage_t0 = self._t0        # 当前生成阶段的起点
        self._first_tick_at = None       # 本阶段第一个 token 到达的时刻
        self._last_log = 0.0
        self._last_render = 0.0
        try:
            self._live = bool(enabled) and bool(sys.stderr.isatty())
        except Exception:
            self._live = False

        if self.enabled:
            try:
                import comfy.utils
                self.pbar = comfy.utils.ProgressBar(self.TOTAL, node_id=node_id)
            except Exception as e:      # 不在 ComfyUI 里跑（例如单测）时会走到这里
                logger.debug(f"[Qwen35] 前端进度条不可用，仅用控制台: {e}")
                self.pbar = None

    # ---------------------------------------------------------------- 内部
    def _push(self):
        """推一次给前端。ProgressBar 内部已有 100ms / 0.5% 节流，这里不用再判。"""
        if self.pbar is None:
            return
        try:
            self.pbar.update_absolute(int(round(self.value)), int(round(self.TOTAL)))
        except Exception:
            self.pbar = None

    @staticmethod
    def _write_live(text):
        global _UNICODE_BAR
        try:
            sys.stderr.write("\r" + text)
            sys.stderr.flush()
        except UnicodeEncodeError:
            # GBK 之类的控制台画不出方块，回落 ASCII 后重试一次
            _UNICODE_BAR = False
            try:
                sys.stderr.write("\r" + text.encode("ascii", "replace").decode("ascii"))
                sys.stderr.flush()
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _clear_live():
        try:
            if sys.stderr.isatty():
                sys.stderr.write("\r" + " " * 120 + "\r")
                sys.stderr.flush()
        except Exception:
            pass

    @staticmethod
    def _bar(frac, width=22):
        fill = int(round(max(0.0, min(1.0, frac)) * width))
        if _UNICODE_BAR:
            return "█" * fill + "░" * (width - fill)
        return "#" * fill + "-" * (width - fill)

    def _render(self, done, total, note="", unit="tok"):
        frac = 0.0 if not total else max(0.0, min(1.0, float(done) / float(total)))
        elapsed = time.perf_counter() - self._t0
        # 速度列始终带单位：起步阶段（elapsed 太短）或 0 token 时也要能看懂
        spd = f"{done / elapsed:5.1f} {unit}/s" if done and elapsed > 0.05 else f"  --  {unit}/s"
        eta = ""
        if done and elapsed > 0.05 and total > done:
            eta = f"  剩余 ~{(elapsed / done) * (total - done):5.1f}s"
        tail = f"  ({note})" if note else ""
        return (f"生成 {self._bar(frac)} {frac * 100:5.1f}%  "
                f"{int(done)}/{int(total)} {unit}  {spd}{eta}{tail}")

    # ---------------------------------------------------------------- 对外
    def message(self, text):
        """阶段提示，立即输出一行日志（生成中的实时条先擦掉，避免串行）。"""
        if not self.enabled:
            return
        self._clear_live()
        logger.info(f"[Qwen35] {text}")

    def mark(self, value):
        """设置绝对进度，用于不可细分的阶段（卸载 / 加载 / 预处理）。"""
        self.value = float(value)
        self._push()

    def begin_stage(self, base, span):
        """进入可逐单位推进的阶段，之后用 tick() 报进度。"""
        self.base, self.span = float(base), float(span)
        self._stage_t0 = time.perf_counter()
        self._first_tick_at = None

    def rescale(self, boundary, span):
        """阶段切换时按实测数据重画剩余宽度。

        值只前进不后退（取 max），所以调用方即使估错也不会让进度条回跳。
        """
        self.value = max(self.value, float(boundary))
        self.base, self.span = float(boundary), float(span)
        self._stage_t0 = time.perf_counter()
        self._first_tick_at = None
        self._push()

    @property
    def prefill_s(self):
        """本阶段从开始到第一个 token 的等待时间（prefill）。没出过 token 返回 0。"""
        if self._first_tick_at is None:
            return 0.0
        return max(0.0, self._first_tick_at - self._stage_t0)

    def tick(self, done, total, note="", unit="tok"):
        """逐单位推进：更新前端进度 + 按节流输出控制台进度。

        unit 默认 "tok"（生成阶段）。批量打标节点按「张」推进，传 unit="张"
        即可复用同一个渲染器 —— 否则日志会写成 "37/200 tok"，读起来是错的。
        """
        if self._first_tick_at is None:
            self._first_tick_at = time.perf_counter()
        total = float(total) or 1.0
        self.value = self.base + self.span * max(0.0, min(1.0, float(done) / total))
        self._push()
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._live and now - self._last_render >= 0.15:
            self._last_render = now
            self._write_live(self._render(done, total, note, unit))
        if now - self._last_log >= self.interval or done >= total:
            self._last_log = now
            self._clear_live()
            logger.info(f"[Qwen35] {self._render(done, total, note, unit)}")

    def finish(self):
        self.value = self.TOTAL
        self._push()
        self._clear_live()


def _make_progress_criteria(reporter, max_new_tokens, allow_interrupt=True, unit="tok"):
    """
    构造一个 StoppingCriteria：每生成一个 token 被回调一次，用来驱动进度。

    额外好处：每步检查 ComfyUI 的中断标志，于是前端点"取消"能真正打断
    一次可能长达数十秒的生成（否则要等 max_new_tokens 跑完）。

    reporter=None 时只做中断检查、不推进度 —— 批量打标节点要的是这个：
    它的进度条按「第几张图」推进，若在这里再按 token 推进，两者会互相覆盖。

    构造失败返回 None，此时 generate 不传 stopping_criteria，功能照旧。
    """
    try:
        from transformers import StoppingCriteria
    except Exception as e:
        logger.debug(f"[Qwen35] 拿不到 StoppingCriteria，跳过 token 级进度: {e}")
        return None

    class _TokenProgress(StoppingCriteria):
        def __init__(self):
            super().__init__()
            self.n = 0
            self.t0 = time.perf_counter()
            self.first_at = None      # 第一个 token 到达的时刻

        @property
        def prefill_s(self):
            """从开始到第一个 token 的等待时间。

            批量打标节点用 reporter=None，拿不到 reporter.prefill_s，
            所以这里自己记一份 —— prefill 与解码变慢的原因完全不同，
            日志必须能分开看（这是本项目反复强调的那条）。
            """
            if self.first_at is None:
                return 0.0
            return max(0.0, self.first_at - self.t0)

        def __call__(self, input_ids=None, scores=None, **kwargs):
            self.n += 1
            if self.first_at is None:
                self.first_at = time.perf_counter()
            if allow_interrupt:
                try:
                    import comfy.model_management as mm
                    mm.throw_exception_if_processing_interrupted()
                except ImportError:
                    pass
                except AttributeError:
                    pass
            if reporter is not None:
                reporter.tick(self.n, max_new_tokens, unit=unit)
            return False

    try:
        return _TokenProgress()
    except Exception as e:
        logger.debug(f"[Qwen35] 创建进度回调失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
class Qwen35PromptEnhancer:
    """用本地 Qwen 模型把简单提示词扩写成 MiniMax H3 规范提示词（图片可选）。"""

    _cache = {
        "model": None,
        "processor": None,
        "key": None,      # (model_path, quantization, attention)
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_choices(),),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT,
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只橘猫在清晨的咖啡馆窗边打哈欠，阳光斜射进来，背景是巴黎街景。",
                }),
                # 默认 none。真机 A/B（9B，同工作流）整轮：none 51.85s /
                # 4bit 54.06s / 8bit 71.01s（8bit 多花的 17s 里 14.3s 是加载）。
                # 显存够就别开；24GB 卡跑 9B bf16 只占 17.5GiB。
                "quantization": (["none", "8bit", "4bit"], {"default": "none"}),
                # auto = 装了 flash-attn 就用 flash_attention_2，否则退回 sdpa。
                # 注意：注意力后端**不是**图片 prefill 快慢的原因（实测三种后端
                # 耗时 29.16/28.91/29.30s，毫无差别）；prefill 的真凶是视觉塔
                # 入口那个 Conv3d，已由 _patch_vision_patch_embed 自动修掉。
                # 这项只影响 LLM 侧长上下文与显存占用。
                "attention": (list(_ATTENTION_CHOICES), {"default": "auto"}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "mode": (["text2video", "image2video", "reference"], {"default": "text2video"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "max_images": ("INT", {"default": 4, "min": 1, "max": 9, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "unload_other_models": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "custom_model_path": ("STRING", {"default": ""}),
                "max_image_side": ("INT", {"default": 1280, "min": 0, "max": 4096, "step": 128}),
                # ↓ 以下两项追加在最后：旧工作流 widgets 数量不足时自动取默认值，不会错位
                "show_progress": ("BOOLEAN", {"default": True}),
                "progress_interval": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                # 双语预览：off = 只出英文（默认，第二段根本不会执行，零额外耗时）；
                # en_then_zh = 额外翻一份中文到 prompt_zh 端点，供人工核对
                "bilingual": (["off", "en_then_zh"], {"default": "off"}),
            },
            "hidden": {
                # 节点 id，用来把进度条挂到正确的节点上
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "prompt_zh")
    FUNCTION = "enhance"
    CATEGORY = "Qwen35/Prompt"
    OUTPUT_NODE = True

    # ----------------------------------------------------------------
    def _resolve_path(self, model_name, custom_model_path):
        if custom_model_path and custom_model_path.strip():
            p = custom_model_path.strip().strip('"')
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                return p
        mapping = discover_local_models()
        if model_name in mapping:
            return mapping[model_name]
        # 兜底：名字前缀匹配
        for k, v in mapping.items():
            if k.startswith(model_name):
                return v
        raise RuntimeError(
            f"找不到模型 '{model_name}'。请确认它是含 config.json 的完整 HF 文件夹，"
            f"或在上面的 custom_model_path 里填绝对路径。"
        )

    # ----------------------------------------------------------------
    def _free_vram(self, quantization, unload_other_models, pbar, weight=0.0):
        """卸载其他模型 + 尽量把显存还回来，返回耗时秒数。

        扩写节点与批量打标节点共用。weight 只决定进度条 mark 到哪 ——
        两个节点的进度分配不同（批量场景下加载要摊到 N 张图上，占比小得多），
        所以由调用方传进来，不在方法里写死。

        ComfyUI 的 async-offload 会 pin 住大量内存，unload_all_models() 不一定
        真的把显存还回来；后面再补一刀 free_memory(按需驱逐) + soft_empty_cache。
        这三步都在 try 里，任何一步失败都不影响出结果。
        """
        t0 = time.perf_counter()
        mm_mod = None
        try:
            import comfy.model_management as mm_mod
        except Exception as e:
            logger.debug(f"[Qwen35] 拿不到 comfy.model_management: {e}")

        need_mib = _VRAM_NEED_MIB.get(str(quantization), 9000)
        if unload_other_models:
            free_before, _, ram_before = _probe_memory(mm_mod)
            if pbar is not None:
                pbar.message("释放显存：卸载其他模型…")
            if mm_mod is not None:
                try:
                    mm_mod.unload_all_models()
                except Exception as e:
                    logger.warning(f"[Qwen35] unload_all_models 失败: {e}")
                try:
                    target = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                    mm_mod.free_memory(need_mib * 1024 * 1024, target)
                except Exception as e:
                    logger.debug(f"[Qwen35] free_memory 不可用或失败: {e}")
                try:
                    mm_mod.soft_empty_cache(force=True)
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            free_after, total_vram, ram_after = _probe_memory(mm_mod)
            if free_after or free_before:
                line = (f"显存 {_fmt_gib(free_before)} → {_fmt_gib(free_after)} 可用")
                if total_vram:
                    line += f" / 共 {_fmt_gib(total_vram)}"
                line += f"，本档量化约需 {need_mib / 1024:.1f}GiB"
                if ram_after:
                    line += f"；内存可用 {_fmt_gib(ram_after)}"
                logger.info(f"[Qwen35] {line}")
                if free_after < need_mib * 1024 * 1024:
                    logger.warning(
                        f"[Qwen35] ⚠ 空闲显存 {_fmt_gib(free_after)} 低于本档量化约需的 "
                        f"{need_mib / 1024:.1f}GiB，模型很可能被摊到 CPU 上运行（速度会掉 1~2 个数量级）。"
                        f"建议先关掉占显存的程序，或把 quantization 调到 4bit / 调小 max_image_side。"
                    )
        if pbar is not None:
            pbar.mark(weight)
        return time.perf_counter() - t0

    # ----------------------------------------------------------------
    def _load(self, path, quantization, attention):
        global _CPU_OFFLOAD_GIB, _CUR_QUANT
        # 记录本次实际用的档位与 CPU 摊派量，供解码体检分层归因。
        # 每次都重置：两个都是「当前模型实例」的属性，不是累计量。
        _CUR_QUANT = str(quantization)
        _CPU_OFFLOAD_GIB = 0.0
        # 先把 "auto" 解析成真实后端，缓存键用解析后的值 ——
        # 否则「装了 flash-attn 但进程里还复用着 sdpa 的老模型」会一直存在。
        attn_impl, attn_note = _resolve_attention(attention)
        key = (path, quantization, attn_impl)
        if self._cache["key"] == key and self._cache["model"] is not None:
            return self._cache["model"], self._cache["processor"], False

        # 换了模型 → 先把旧的清掉
        self._release(force=True)

        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        logger.info(
            f"[Qwen35] loading: {path}  quant={quantization}  attn={attn_impl}"
            + (f"  [{attn_note}]" if attn_note else "")
        )

        quant_cfg = None
        if quantization == "4bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=list(_QUANT_SKIP_MODULES),
            )
        elif quantization == "8bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
                # 关键：默认 6.0 会永远走 int8_mixed_scaled_mm（离群列混合内核），
                # 实测比重 1.7~1.9 倍。0.0 → 走纯 int8_scaled_mm。详见文件头部常量注释。
                llm_int8_threshold=_INT8_THRESHOLD,
                llm_int8_skip_modules=list(_QUANT_SKIP_MODULES),
            )

        kwargs = {"device_map": "auto", "attn_implementation": attn_impl}
        if quant_cfg is not None:
            kwargs["quantization_config"] = quant_cfg
            kwargs["dtype"] = "auto"
        else:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            kwargs["dtype"] = dtype

        def _from_pretrained(impl):
            kw = dict(kwargs)
            kw["attn_implementation"] = impl
            try:
                return AutoModelForImageTextToText.from_pretrained(path, **kw)
            except TypeError:
                # 老版本 transformers 用 torch_dtype
                kw["torch_dtype"] = kw.pop("dtype")
                return AutoModelForImageTextToText.from_pretrained(path, **kw)

        try:
            model = _from_pretrained(attn_impl)
        except Exception as e:
            # flash 路径建不起来（包不完整 / 版本不合 / 有层落在 CPU 上）时
            # 退回 sdpa —— 慢总比整个工作流跑挂好。
            if attn_impl.startswith("flash"):
                logger.warning(
                    f"[Qwen35] ⚠ 以 {attn_impl} 初始化失败（{type(e).__name__}: {e}），"
                    f"本次退回 sdpa。视觉塔会走逐图切块路径，图片 prefill 会明显变慢。"
                )
                attn_impl = "sdpa"
                attn_note = ""
                key = (path, quantization, attn_impl)
                model = _from_pretrained("sdpa")
            else:
                raise

        model.eval()
        processor = AutoProcessor.from_pretrained(path)

        # ---- 修掉视觉塔入口的 Conv3d 病态路径（本机实测的头号瓶颈）----
        _patch_vision_patch_embed(model)

        # ---- 设备分布体检 ----
        # 空闲显存不够时 device_map="auto" 会静默把一部分层放到 CPU 上，速度
        # 直接掉两个数量级。这里把分布写进日志，越界就报警。
        counts = _device_map_summary(model)
        vis_dev = _vision_device(model)
        vis_mod = _vision_module(model)
        dev_b, dt_b = _param_device_stats(model)
        vis_b, vis_dt = _param_device_stats(model, vis_mod)

        # 主模型权重分布 —— 以参数量为准（hf_device_map 在量化+单卡时常常是空的）
        if dev_b:
            logger.info(f"[Qwen35] 权重分布：{_fmt_stat(dev_b)}   |   精度：{_fmt_stat(dt_b)}")
        if vis_mod is not None and vis_b:
            logger.info(f"[Qwen35] 视觉塔  ：{_fmt_stat(vis_b)}   |   精度：{_fmt_stat(vis_dt)}")
        elif vis_dev:
            logger.info(f"[Qwen35] 视觉塔在 {vis_dev}")
        if counts:
            desc = "  ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
            logger.info(f"[Qwen35] 设备映射：{desc}")

        # 视觉塔入口的路径 —— prefill 快慢的关键，比注意力后端重要得多。
        # 必须按「本模型实例」报，不能按「类补过没有」报：Qwen3.5 用的
        # Qwen3_5VisionPatchEmbed 和 Qwen3-VL 那个类同名不同物，
        # 只看类级标志会把「补到了别的类、当前模型其实没补」误报成成功。
        if _PATCH_EMBED_INSTANCES > 0:
            logger.info(
                f"[Qwen35] 视觉塔入口：patch_embed 走 等价 GEMM"
                f"（本模型实例已替换 {_PATCH_EMBED_INSTANCES} 个，已绕开 Conv3d 慢路径）"
            )
        else:
            logger.warning(
                "[Qwen35] 视觉塔入口：⚠ 没在本模型里找到可替换的 patch_embed —— "
                "prefill 可能仍走 Conv3d 慢路径（本机 fp16/bf16 下可达 35ms/视觉token）。"
                "若「耗时分解」里 prefill 明显偏高，请反馈模型名"
            )

        # 注意力后端体检。
        # 注意：「缺 flash-attn 导致视觉塔逐图切块」这个早先的判断**已被实测推翻**。
        # 同权重同输入下 eager / sdpa / flash_attention_2 分别是 29.16 / 28.91 /
        # 29.30 秒，三种后端毫无差别 —— 说明 prefill 的瓶颈根本不在注意力。
        # 真凶是视觉塔入口的 Conv3d（见 _patch_vision_patch_embed），
        # 本机 fp16/bf16 下首次调用超过 45s，而 fp32 只要 6.5ms。
        # 所以这一行只作「后端是否按预期生效」的记录，不再据此告警。
        try:
            top_attn = getattr(getattr(model, "config", None), "_attn_implementation", None)
            vis_attn = getattr(getattr(vis_mod, "config", None), "_attn_implementation", None)
            fa_ver = _flash_attn_version()
            logger.info(
                f"[Qwen35] 注意力后端：主模型 {top_attn or '?'} / 视觉塔 {vis_attn or '?'}"
                f"（请求 {attention} → 生效 {attn_impl}，flash-attn "
                + (f"{'.'.join(map(str, fa_ver))}" if fa_ver else "未装")
                + "）"
            )
        except Exception:
            pass

        # 线性注意力快路径体检：Qwen3.5 是混合架构（24/32 层是 Gated DeltaNet），
        # 这些层在解码期占大头，而 transformers 自带一份很慢的 torch 回退。
        # 必须按实例报（见 _scan_linear_attn_fast_path 的说明）。
        try:
            n_fused, n_lin = _scan_linear_attn_fast_path(model)
            note = _linear_attn_load_note(n_fused, n_lin)
            if note:
                if n_fused == n_lin:
                    logger.info(note)
                else:
                    logger.warning(note)
        except Exception:
            pass

        # 判定「有没有层被摊到 CPU」：CPU 上出现任何非零权重都算越界
        cpu_bytes = sum(v for k, v in (dev_b or {}).items() if not k.startswith("cuda"))
        vis_cpu = sum(v for k, v in (vis_b or {}).items() if not k.startswith("cuda"))
        _CPU_OFFLOAD_GIB = cpu_bytes / (1024 ** 3)
        if cpu_bytes > 0:
            vis_note = f"（其中视觉塔 {_fmt_gib(vis_cpu)}）" if vis_cpu > 0 else ""
            need_gb = _VRAM_NEED_MIB.get(str(quantization), 9000) / 1024
            logger.warning(
                f"[Qwen35] ⚠ 有 {_fmt_gib(cpu_bytes)} 权重没放在显卡上{vis_note}。"
                f"这会让生成慢到 1 tok/s 量级 —— 视觉塔在 CPU 上时图片 prefill 会慢到上百秒。"
                f"处理办法：跑扩写前先让显卡腾出约 {need_gb:.0f}GB 空闲"
                f"（关掉占显存的浏览器/播放器，或改用更小的 max_image_side / 4bit）。"
            )

        # 量化档体检：8bit 在这台机器上是反直觉的负收益（见 _SLOW_DECODE_TOK_S 注释）。
        # 层全在 GPU 却还慢，最常见的原因就是它 —— 所以在加载时就把话说清楚，
        # 而不是等用户跑完一轮、看到慢，再回头猜。
        if str(quantization) == "8bit" and cpu_bytes == 0:
            logger.warning(
                "[Qwen35] ℹ 量化档=8bit：真机 A/B（9B，同工作流）整轮 71.01s vs "
                "4bit 54.06s / none 51.85s。多花的 17s 里 **14.3s 是加载**"
                "（33.90s vs 19.60s），解码端每 token 再慢 1.4~1.6 倍"
                "（12.4/13.1 vs 17.3/19.7 tok/s），换来的只是 17.5GiB→11.1GiB 显存。"
                f"（已把 llm_int8_threshold 设为 {_INT8_THRESHOLD}，"
                "int8 内部能快 1.7~1.9 倍，但仍追不上 bf16。）"
                "→ 要省显存请改用 **4bit**：同机实测整轮只比 none 慢 4%、"
                "显存 7.9GiB（vs none 17.5GiB）。"
            )
        # 精度体检：视觉塔被量化 → 每次前向都要实时反量化，图片 prefill 会慢几十倍。
        # 这是实测踩到的坑：视觉塔 4bit 化后 prefill 要 57~86s（占整轮 79%）。
        if vis_dt:
            total_vis = sum(vis_dt.values()) or 1
            quant_vis = 0
            fp32_vis = 0
            for k, v in vis_dt.items():
                if k in ("uint8", "int8"):
                    quant_vis += v
                elif k in ("float32", "float"):
                    fp32_vis += v
            if quant_vis / total_vis > 0.5:
                logger.warning(
                    f"[Qwen35] ⚠ 视觉塔一半以上权重被量化（{_fmt_stat(vis_dt)}）。"
                    f"量化视觉塔要在每次前向里实时反量化，会把图片 prefill 拖到几十秒"
                    f"（实测 57~86s，占整轮 79%）。请确认 _QUANT_SKIP_MODULES 生效，"
                    f"或改用 quantization=none。"
                )
            elif fp32_vis / total_vis > 0.5:
                logger.warning(
                    f"[Qwen35] ⚠ 视觉塔一半以上权重是 fp32（{_fmt_stat(vis_dt)}），"
                    f"会显著拖慢图片 prefill。理想情况应是 bfloat16。"
                )

        self._cache.update({"model": model, "processor": processor, "key": key})
        logger.info(f"[Qwen35] loaded OK -> {type(model).__name__}")
        return model, processor, True

    def _release(self, force=False):
        if self._cache["model"] is None:
            return
        try:
            del self._cache["model"]
            del self._cache["processor"]
        except Exception:
            pass
        self._cache.update({"model": None, "processor": None, "key": None})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 注意：不要在这里调 torch.cuda.ipc_collect()。ComfyUI 是单进程场景，
            # 该调用在多进程 IPC 句柄表上遍历，实测会带来数秒无谓延迟。

    # ----------------------------------------------------------------
    @staticmethod
    def _collect_images(image, image_2, image_3, image_4, max_images, max_side=0):
        """按附加顺序收集所有参考图。

        ComfyUI 的 IMAGE 张量是 [B, H, W, C] 的 batch，一个输入可能带多帧，
        这里全部展开并按 输入顺序 -> batch 内顺序 编号，与 H3 编码器把
        参考图按 <Picture 1> .. <Picture N> 编号的规则一致。

        max_side > 0 时长边超过该值的图会被等比缩小。原因：Qwen3-VL 的
        vision token 数随边长近似平方增长（1024² → ~1000 tok，1536² → ~2300 tok），
        适度缩图能显著降低 prefill 与显存开销，对提示词扩写几乎无损。
        """
        from PIL import Image

        collected = []
        for src in (image, image_2, image_3, image_4):
            if src is None:
                continue
            if not isinstance(src, torch.Tensor):
                continue
            for i in range(int(src.shape[0])):
                if len(collected) >= int(max_images):
                    return collected
                arr = (src[i].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                im = Image.fromarray(arr)
                if max_side and max(im.size) > int(max_side):
                    ratio = int(max_side) / max(im.size)
                    im = im.resize(
                        (max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                        Image.LANCZOS,
                    )
                collected.append(im)
        return collected

    # ----------------------------------------------------------------
    def enhance(self, model_name, system_prompt, user_prompt, quantization, attention,
                enable_thinking=False, mode="text2video",
                image=None, image_2=None, image_3=None, image_4=None, max_images=4,
                keep_model_loaded=False,
                unload_other_models=True, temperature=0.4, max_new_tokens=1024,
                seed=42, custom_model_path="", max_image_side=1280,
                show_progress=True, progress_interval=2.0, bilingual="off",
                unique_id=None):
        t_start = time.perf_counter()
        pbar = _ProgressReporter(
            node_id=unique_id, enabled=show_progress, interval=progress_interval
        )
        path = self._resolve_path(model_name, custom_model_path)

        # ---- 1/5 卸载其他模型 + 腾出显存 ----
        t_unload = self._free_vram(quantization, unload_other_models, pbar, _W_UNLOAD)

        # ---- 2/5 加载扩写模型（时间大头，进度条在这里最有用）----
        t0 = time.perf_counter()
        will_reuse = (
            self._cache["key"] == (path, quantization, attention)
            and self._cache["model"] is not None
        )
        if will_reuse:
            pbar.message(f"复用常驻模型：{os.path.basename(path)}")
        else:
            pbar.message(
                f"正在加载模型：{os.path.basename(path)}"
                f"（量化={quantization}，首次或换模型约 30~90s，请耐心等待）"
            )
        pbar.mark(_W_UNLOAD + 1.0)      # 先把条推到 3%，让前端知道已经开工
        model, processor, freshly_loaded = self._load(path, quantization, attention)
        t_load = time.perf_counter() - t0
        pbar.mark(_W_UNLOAD + _W_LOAD)
        if freshly_loaded:
            pbar.message(f"模型加载完成，用时 {t_load:.1f}s")

        torch.manual_seed(int(seed))

        # ---- 3/5 组装消息（多图按附加顺序编号，对应 H3 的 <Picture i>）----
        t0 = time.perf_counter()
        pil_images = self._collect_images(
            image, image_2, image_3, image_4, max_images, max_image_side
        )
        t_prep = time.perf_counter() - t0
        pbar.mark(_W_UNLOAD + _W_LOAD + _W_PREP)
        if pil_images:
            sizes = ", ".join(f"{im.width}x{im.height}" for im in pil_images)
            pbar.message(f"参考图 {len(pil_images)} 张：{sizes}")

        content = []
        for im in pil_images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": user_prompt})

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": _inject_mode_hint(system_prompt.strip(), mode),
            })
        messages.append({"role": "user", "content": content})

        logger.info(f"[Qwen35] mode={mode}  images={len(pil_images)}")

        # ---- 走 processor 的 chat template（自动处理图文混排）----
        tmpl_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if _supports_enable_thinking(processor):
            tmpl_kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            # 万一模板不认这个 kwarg，去掉再来一次；输出仍由 strip_thinking 兜底
            tmpl_kwargs.pop("enable_thinking", None)
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        inputs = inputs.to(model.device)

        in_len = inputs["input_ids"].shape[1]

        # 视觉 token 数。用来把 prefill 的绝对秒数换算成"每视觉 token 多少毫秒"——
        # 才能判断慢是"图太大（token 多）"还是"路径低效（每 token 太贵）"。
        n_vtok = _count_visual_tokens(inputs)

        # ---- 4/5 生成英文 H3 提示词（逐 token 进度 + 可中断）----
        # 双语时英文段只占生成段的一小部分（实测中文段解码量略多于英文段），
        # 但准确比例要等第一段跑完才知道，所以先按先验宽度画，之后
        # pbar.rescale 再按实测重算分界点。固定按 6:4 分会与现实不符。
        bilingual_on = (str(bilingual) == "en_then_zh")
        base_gen = _W_UNLOAD + _W_LOAD + _W_PREP
        en_span = _W_GEN * _EN_SPAN_PRIOR if bilingual_on else _W_GEN

        eos_ids = _eos_token_ids(processor, model)
        if eos_ids:
            logger.info(f"[Qwen35] generate 使用 EOS ids = {eos_ids}")

        pbar.begin_stage(base_gen, en_span)
        pbar.message(f"开始生成英文提示词：输入 {in_len} tok，上限 {int(max_new_tokens)} tok")
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            repetition_penalty=1.05,
        )
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids
        criteria = _make_progress_criteria(pbar, int(max_new_tokens))
        if criteria is not None:
            gen_kwargs["stopping_criteria"] = [criteria]

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
        except BaseException:
            # 中断或报错：按需释放显存后原样抛出，让 ComfyUI 正确标记任务状态
            pbar.finish()
            if not keep_model_loaded:
                self._release()
            raise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0
        t_prefill_en = pbar.prefill_s
        t_decode_en = max(0.0, t_gen - t_prefill_en)

        gen = out[0][in_len:]
        out_len = int(gen.shape[0])
        text = processor.decode(gen, skip_special_tokens=True)
        text = normalize_h3_sections(strip_thinking(text))
        done_msg = f"英文提示词完成：{out_len} tok / {len(text)} 字符"
        if t_decode_en > 0:
            done_msg += (
                f"（prefill {t_prefill_en:.1f}s / 解码 {t_decode_en:.1f}s"
                f" = {out_len / t_decode_en:.1f} tok/s）"
            )
        pbar.message(done_msg)
        # 放掉第一轮的输出张量与（含图的）输入，给第二轮腾出显存
        del out, gen, inputs

        # ---- 5/5 翻译成中文预览（纯文本；只给人看，不喂 H3）----
        # 中文版不进 H3，所以可读性优先；但三段标签名原样保留，方便与英文逐段对照。
        text_zh, t_zh, zh_in, zh_out, zh_budget, zh_capped = "", 0.0, 0, 0, 0, False
        t_prefill_zh, t_decode_zh = 0.0, 0.0
        if bilingual_on:
            # 预算按英文实际输出长度换算，而不是照抄 max_new_tokens：
            # 照抄等于第二段没有任何约束，一旦出现回声或思考块就会一路解码到上限。
            zh_budget = _zh_budget(out_len, int(max_new_tokens))
            boundary, span = base_gen + _W_GEN * _EN_SPAN_PRIOR, _W_GEN * (1 - _EN_SPAN_PRIOR)
            if out_len > 0:
                share = out_len / float(out_len + zh_budget)
                share = min(0.90, max(_EN_SPAN_PRIOR, share))
                boundary = base_gen + _W_GEN * share
                span = base_gen + _W_GEN - boundary
            pbar.rescale(boundary, span)
            ratio = (zh_budget / float(out_len)) if out_len else 0.0
            pbar.message(
                f"开始翻译中文预览…（预算 {zh_budget} tok = 英文 {out_len} tok 的 {ratio:.2f} 倍，贪心解码）"
            )
            try:
                text_zh, zh_in, zh_out, t_zh, zh_capped = self._translate_preview(
                    model, processor, text, out_len, zh_budget, pbar, eos_ids,
                )
            except BaseException:
                pbar.finish()
                if not keep_model_loaded:
                    self._release()
                raise
            t_prefill_zh = pbar.prefill_s
            t_decode_zh = max(0.0, t_zh - t_prefill_zh)
            pbar.message(
                f"中文预览完成：{zh_out} tok / {len(text_zh)} 字符"
                + (f"（prefill {t_prefill_zh:.1f}s / 解码 {t_decode_zh:.1f}s"
                   f" = {zh_out / t_decode_zh:.1f} tok/s）" if t_decode_zh > 0 else "")
                + (f"（触到预算上限 {zh_budget} tok，可能被截断）" if zh_capped else "")
            )
        pbar.finish()

        t_release = 0.0
        if not keep_model_loaded:
            t0 = time.perf_counter()
            self._release()
            t_release = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        # 这是本进程第几次跑同一组输入？首次 prefill 含 kernel 编译/CUDA 预热，
        # 偏慢是正常的，日志里必须区分开，免得把「首次」误判成「有病」。
        run_key = (str(path), str(quantization), str(attention), len(pil_images))
        run_idx = _RUN_COUNT.get(run_key, 0)
        _RUN_COUNT[run_key] = run_idx + 1
        first_run = (run_idx == 0)

        def _split_lines(out_tok, pre_s, dec_s):
            """把生成阶段拆成 prefill / 解码两行 —— 两者变慢的原因完全不同。"""
            if dec_s <= 0:
                return []
            return [
                f"      其中 prefill  : {pre_s:6.2f}s",
                f"      其中 解码     : {dec_s:6.2f}s  ({out_tok / dec_s:.1f} tok/s)",
            ]

        def _prefill_note(pre_s, vtoks=0, is_first=False):
            """prefill 偏慢时给定量归因 —— 把秒数换算成「每视觉 token 多少毫秒」。

            这是本项目最有用的一行：只有把绝对秒数摊到 token 上，才能区分
            「图太大（token 多）」和「路径低效（每 token 太贵）」，两者的解法完全不同。
            """
            if pre_s <= _SLOW_PREFILL_S:
                return []
            if vtoks > 0:
                per = pre_s * 1000.0 / vtoks
                out = [
                    f"  ℹ prefill 换算：{vtoks} 个视觉 token × {per:.1f} ms/token"
                    f"（GPU 上正常约 0.1~1 ms）。视觉 token 数 ∝ 边长²，"
                    f"调小 max_image_side 按平方比例缩短",
                ]
                if per < _SLOW_VISION_PER_TOKEN_MS:
                    out.append(
                        f"  ℹ 每 token {per:.1f} ms 说明视觉塔入口正常（patch_embed 已换 GEMM，"
                        f"慢于 {_SLOW_VISION_PER_TOKEN_MS:.0f} ms/token 才要查）"
                    )
                elif _PATCH_EMBED_INSTANCES == 0:
                    # 注意：这里要看「本模型实例替换了几个」，不能看「类补过没有」。
                    # 早期的写法只看类级标志，结果换 Qwen3.5 后补丁打在了另一个类上，
                    # 提示却走进了「已换成 GEMM 但仍偏贵」分支，把人往"图太大"方向带。
                    out.append(
                        "  ℹ 视觉塔入口**没被替换**（本模型里没找到可换的 patch_embed）。"
                        "本机实测 Conv3d 在 fp16/bf16 下首次调用超过 45s，而 fp32 只要 6.5ms "
                        "—— 这就是 prefill 贵的真正来源。请检查上面的「视觉塔入口」那行是否告警"
                    )
                else:
                    out.append(
                        f"  ℹ 视觉塔入口已换 GEMM（{_PATCH_EMBED_INSTANCES} 个实例）但仍偏贵："
                        "可能是图太多/太大，或「权重分布」里有层落在 CPU"
                    )
                return out
            if is_first:
                return ["  ℹ prefill 偏慢：本进程首次跑这组输入，含 kernel 编译；再跑一次可对比"]
            return ["  ℹ prefill 偏慢：对照上面「权重分布」，确认没有层落在 CPU"]

        lines = [
            "[Qwen35] ========== 耗时分解 ==========",
            f"  卸载其他模型      : {t_unload:6.2f}s",
            f"  加载扩写模型      : {t_load:6.2f}s  ({'本次新加载' if freshly_loaded else '复用常驻'} 加载)",
            f"  图片预处理        : {t_prep:6.2f}s  ({len(pil_images)} 图, 长边上限 {max_image_side or '不限'})",
            f"  生成英文提示词    : {t_gen:6.2f}s  输入 {in_len} tok -> 输出 {out_len} tok"
            f"  ({out_len / t_gen:.1f} tok/s)",
        ]
        lines += _split_lines(out_len, t_prefill_en, t_decode_en)
        lines += _prefill_note(t_prefill_en, n_vtok, first_run)
        if bilingual_on and t_zh > 0:
            lines.append(
                f"  翻译中文预览      : {t_zh:6.2f}s  输入 {zh_in} tok -> 输出 {zh_out} tok"
                f"  ({zh_out / t_zh:.1f} tok/s)  预算 {zh_budget} tok"
            )
            lines += _split_lines(zh_out, t_prefill_zh, t_decode_zh)
            lines += _prefill_note(t_prefill_zh, 0, False)
            ratio_txt = f"{zh_out / out_len:.2f} 倍" if out_len else "n/a"
            lines.append(f"  两段 token 比     : 中文/英文 = {zh_out}/{out_len} = {ratio_txt}")
            if zh_capped:
                lines.append(
                    f"  ⚠ 中文段触到预算上限 {zh_budget} tok（= 英文 {out_len} tok × "
                    f"{_ZH_TOK_RATIO} + {_ZH_BUDGET_EXTRA}）。"
                    f"正常译文只需约 1.1 倍英文 token，触顶说明译文在回声或重复，"
                    f"请检查 prompt_zh。"
                )
            if out_len and zh_out < out_len * 0.25:
                lines.append("  ⚠ 中文输出远短于英文，疑似被截断或未完整翻译，请检查 prompt_zh。")
            elif out_len and zh_out > out_len * 1.4:
                lines.append(
                    f"  ⚠ 中文段 {zh_out} tok 明显超出英文 {out_len} tok（正常约 1.1 倍），"
                    f"可能有回声或重复内容，请检查 prompt_zh。"
                )
        # 速度体检：9B/8B 在本机的正常值约 18~22 tok/s（bf16）。明显低于它时
        # 按「CPU 摊派 → 量化档 → 原因未知」三层归因（见 _decode_speed_warning）。
        # 速度体检 + 线性注意力体检，两条的互斥规则见 _decode_notes 的 docstring。
        lines += _decode_notes(out_len, t_decode_en, quantization)
        lines += [
            f"  卸载扩写模型      : {t_release:6.2f}s",
            "  --------------------------------",
            f"  合计              : {t_total:6.2f}s",
        ]
        logger.info("\n".join(lines))
        logger.info(
            f"[Qwen35] output EN {len(text)} chars"
            + (f" / ZH {len(text_zh)} chars" if bilingual_on else "  (bilingual=off)")
        )
        return (text, text_zh)

    # ----------------------------------------------------------------
    def _translate_preview(self, model, processor, en_text, en_tokens, zh_budget,
                           pbar, eos_ids=None):
        """第二阶段：把英文 H3 提示词翻成中文，供人工核对。

        纯文本输入（不带任何图片），所以 prefill 很便宜 —— 这是两阶段方案
        只比单语多花十几秒的原因。

        三个刻意的设计（都是为了压住耗时，不是为了质量）：
        1. **贪心解码**：翻译是确定性任务，给定输入只有一个正确答案，采样只会
           引入随机性并拖后 EOS。省掉 sampler 开销的同时更快收敛。
        2. **预算按英文长度换算**：见 _zh_budget()。照抄 max_new_tokens 会让
           第二段跑满上限，这是"双语慢 3 倍"最常见的成因。
        3. **EOS 显式传入**：万一 generation_config 没登记 <|im_end|>，
           generate 会一路解码到上限才停。

        返回 (中文文本, 输入 token 数, 输出 token 数, 耗时秒, 是否触到预算上限)。
        """
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content":
                "----- BEGIN H3 PROMPT -----\n" + en_text + "\n----- END H3 PROMPT -----"},
        ]
        tmpl_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if _supports_enable_thinking(processor):
            tmpl_kwargs["enable_thinking"] = False   # 翻译只需要输出，永远不要思考块
        try:
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop("enable_thinking", None)
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        inputs = inputs.to(model.device)
        in_len = inputs["input_ids"].shape[1]

        budget = max(1, int(zh_budget))
        gen_kwargs = dict(
            max_new_tokens=budget,
            do_sample=False,
            repetition_penalty=1.05,   # 贪心偶尔会绕圈，保留一个轻惩罚
        )
        if eos_ids:
            gen_kwargs["eos_token_id"] = list(eos_ids)
        criteria = _make_progress_criteria(pbar, budget)
        if criteria is not None:
            gen_kwargs["stopping_criteria"] = [criteria]

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen = out[0][in_len:]
        out_len = int(gen.shape[0])
        raw = processor.decode(gen, skip_special_tokens=True)
        # 顺序：剥思考块 → 还原中文标签 → 砍英文回声 → 规范化三段式。
        # 标签还原必须排在砍回声之前，否则回声检测认不出中文写法的三段标签。
        zh = normalize_h3_sections(_drop_english_echo(_restore_zh_labels(strip_thinking(raw))))
        return zh, in_len, out_len, elapsed, out_len >= budget


# ---------------------------------------------------------------------------
# 兜底 system prompt（MiniMax H3 官方 h3-prompt-writing 精简版）
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """[MiniMax H3 prompt-writing skill rules — follow strictly]

You are a MiniMax H3 video generation prompt rewriting expert.
Rewrite the user input (and the reference image if provided) into ONE English prompt
that strictly follows the MiniMax H3 base-en specification.

OUTPUT STRUCTURE — reproduce this EXACTLY, three labelled paragraphs separated by ONE blank line:

integrated_multimodal_description: [Shot 1] <style tag>, <composition>, ...
(blank line)
overall_soundscape: ...
(blank line)
non_diegetic_music: ...

CRITICAL: the three labels "integrated_multimodal_description:", "overall_soundscape:" and
"non_diegetic_music:" MUST each appear verbatim at the very start of their paragraph.
Never drop the "integrated_multimodal_description:" label, never merge the paragraphs,
never add any other heading.

Mandatory rules:
  1. The very first phrase must be a style tag (Live-action / Cinematic / 2D-animated / 3D CG / Claymation / Watercolor / Vintage film) followed by an opening composition (medium-wide shot / close-up / wide shot / extreme close-up ...).
  2. Shot notation: [Shot 1] carries NO timestamp; later shots use "At MM:SS.mmm, the camera cuts to ..." with strictly increasing timestamps within the requested duration.
  3. Camera motion MUST specify type (Zoom / Pan / Tilt / Dolly / Truck / Pedestal / Arc / Tracking / Static / Shake / POV / Roll), amplitude (small / large, omit if medium) and speed (slow / fast, omit if normal). Example: "The camera pushes in with small amplitude at slow speed toward the folded letter in her hands."
  4. Actions use present continuous tense: walking, flowing, rotating, drifting, opening, closing, falling, rising.
  5. At least one diegetic audio cue must appear inside the multimodal description (footsteps, fabric rustle, glass breaking, breathing, ambient sound).
  6. Speaker / dialogue / singing (only when the user asks for speech): stable IDs like (S1), format "The young woman with a quiet voice (S1) says: <d>[English] ...</d>".
  7. On-screen text: wrap the original text verbatim in English double quotes.
  8. overall_soundscape: 1-4 English sentences summarising ambient / physical sound across the whole clip. Use "N/A" only for explicit complete silence.
  9. non_diegetic_music: 1-3 English sentences on instrumentation, tempo and dynamic changes. No abstract mood words. Use "N/A" if none.
 10. Total content length MUST match the requested video duration (4-15 s). Plain English only — no Chinese, no markdown, no explanations, no thinking.

Rewrite the user input now:"""


# ---------------------------------------------------------------------------
# 双语预览：第二阶段的翻译 system prompt
#
# 中文版只给人看、不喂 H3，所以可读性优先于字面忠实；但结构标记原样保留，
# 才能与英文版逐段对照。解码方式是贪心（do_sample=False），确定性优先。
#
# 第 0 条规则是性能条款而不是风格条款：实测模型很爱先把英文原文抄一遍再翻译，
# 那样第二段要解码约 2 倍 token，整轮耗时直接翻到 3 倍。所以在 prompt 里
# 明确禁掉，代码里 _drop_english_echo() 再做一道兜底。
# ---------------------------------------------------------------------------
TRANSLATE_SYSTEM_PROMPT = """[PREVIEW TRANSLATION — human reading only, never sent to the video model]

You translate a finished MiniMax H3 video prompt into natural, fluent Simplified Chinese.
A person reads this to check whether the prompt really describes what they wanted, so
readability matters more than literal fidelity.

RULES
0. The user message contains the English prompt between BEGIN/END markers. NEVER repeat,
   quote, copy, echo, summarise or explain it. Output the Chinese translation directly,
   beginning immediately with "integrated_multimodal_description:".
1. Keep these structural tokens EXACTLY as they are — never translate, reorder or renumber:
   - the three paragraph labels: "integrated_multimodal_description:", "overall_soundscape:",
     "non_diegetic_music:"
   - shot markers "[Shot 1]", "[Shot 2]" ...
   - timestamps such as "At 00:02.500"
   - speaker ids "(S1)", "(S2)" ...
   - reference tags "<Picture 1>" ...
2. Translate ONLY the descriptive content that follows those tokens.
3. Use standard Chinese cinematography wording: 推镜 / 拉镜 / 摇镜 / 移镜 / 跟镜 / 升降 /
   环绕 / 固定 / 手持晃动 / POV / 旋滚, plus 小幅 / 大幅 for amplitude and 慢速 / 快速 for
   speed. If the English omitted an amplitude or speed because it meant medium or normal,
   omit it in Chinese too.
4. Dialogue inside <d>...</d>: keep the original line verbatim, then add a short Chinese
   gloss right after the closing tag — for example
   <d>[English] I've been waiting for you.</d>  （对白：我等你很久了。）
5. Keep the exact same three-paragraph layout, separated by one blank line. Do not add,
   merge or reorder paragraphs. The Chinese translation is at most as long as the English
   source — never pad it, never repeat a sentence.
6. Output ONLY the Chinese preview text: no title, no summary, no bullet list, no markdown
   fence, no thinking block, no explanation, no English sentences of your own.

Translate the H3 prompt given by the user now:"""


# ===========================================================================
# 批量打标节点
#
# 需求：给一个文件夹，把里面每张图喂给同一个 Qwen 模型打标，标签写到
# 「与图片同目录、同文件名」的 .txt；系统提示词可改。
#
# 与扩写节点的关系 —— **速度优化全部复用，不另做一套**：
#   ① 直接继承 Qwen35PromptEnhancer 的 _resolve_path / _free_vram / _load /
#      _release，于是下面这些优化自动带上：
#        · 视觉塔 patch_embed 的 Conv3d → 等价 GEMM 替换（实测 586x，图片
#          prefill 从几十秒降到 0.05s）。批量场景**每张图都要过视觉塔**，
#          漏了它一个文件夹根本没法跑 —— 这是最要紧的一条。
#        · 量化 none / 4bit / 8bit，含 _QUANT_SKIP_MODULES 跳过视觉塔与
#          线性注意力里那两个 Linear(4096,32) 小投影
#        · 8bit 走 llm_int8_threshold=0.0 的纯 int8_scaled_mm（内部快 1.7~1.9x）
#        · 注意力后端 auto / flash_attention_2 / sdpa / eager，失败自动退回 sdpa
#        · 加载期全部体检日志（权重分布 / 视觉塔入口 / 线性注意力 / 量化档）
#   ② **共用同一份常驻模型缓存**（_cache 是父类的类属性）：批量与扩写同时
#      出现在一个工作流里时不会各加载一份 17.5GiB。
#   ③ **一次加载打完整个文件夹** —— 批量侧最大的收益。若改用扩写节点循环 N 次、
#      keep_model_loaded=False，每张都要重载（本机 bf16 实测 23.05s/次）；
#      这里加载只做一次再摊到 N 张上。文件夹越大越占便宜。
#
# 刻意**不**复用扩写节点的地方：
#   · 不跑 H3 的 normalize_h3_sections / _inject_mode_hint —— 打标与 H3 无关，
#     擅自改写用户填的系统提示词是错的。
#   · 默认 max_new_tokens=256 而不是 1024：标签很短，给 1024 只会让偶尔跑飞
#     的那几张白等到底。
#   · 单张失败不中断整批；但 ComfyUI 的「取消」必须中断（见 _is_comfy_interrupt）。
# ===========================================================================
DEFAULT_TAGGER_USER_PROMPT = "给这张图打标。"

DEFAULT_TAGGER_SYSTEM_PROMPT = """You are an image tagging model building a training dataset.

Look at the image and output tags only.

Output format:
- One single line of comma-separated tags. Lowercase. No sentences, no explanation,
  no markdown, no code fences, no numbering, no quotes.

Tag order (keep this order; skip whatever does not apply):
1. subject count and type (1girl, 2boys, solo, no_humans)
2. subject identity or species (cat, woman, robot, building)
3. appearance (long_hair, blue_eyes, black_fur, blonde_hair)
4. clothing and held or worn items (school_uniform, glasses, holding_sword)
5. pose, action, camera view (sitting, looking_at_viewer, from_side, walking)
6. background, scene, objects (outdoors, cafe, window, night_sky)
7. lighting, time of day, style (backlighting, day, photorealistic)

Rules:
- Use common danbooru-style tag names joined by underscores: long_hair, not "long hair".
- Tag only what is clearly visible. Never guess names, artists, or hidden details.
- No quality or meta words (masterpiece, best_quality, highres), no ratings, no <lora:...>.
- Aim for 10~30 tags. Drop a tag when you are not reasonably sure.
"""

# ---------------------------------------------------------------------------
# 打标系统提示词：三套训练场景预设
#
# 三种数据集的标签体系差别很大 —— 真人写实要摄影术语（85mm / rim_light /
# film_grain），二次元要 booru 属性（twintails / serafuku / cel_shading），
# 场景概念图要建筑与光照词（vanishing_point / god_rays / matte_painting）。
# 混用等于往数据集里灌噪声，所以内置三套，用 system_preset 下拉切换。
#
# 三套共同的三条硬规则（哪一套都不许省）：
#   · 只标看得见的 —— 不猜人名、作品名、画师名、真实地点
#   · 不出质量词 / 评分 / meta 词（masterpiece、best_quality、absurdres、trending_on_pixiv）
#   · 一行逗号标签，小写下划线，不带句子、不带 markdown
# 差异只在「标签域的顺序」与「宁缺毋滥的边界」。
#
# 选 system_preset = "custom" 时用节点上那个可编辑的 system_prompt（默认值见上）。
# ---------------------------------------------------------------------------

TAGGER_PRESET_PHOTOREAL = """You are an image tagging model building a photorealistic photograph / portrait training dataset.

Output format:
- ONE single line of comma-separated tags. All lowercase. No sentences, no explanation, no markdown, no code fences, no numbering, no quotes, no trailing period.

Tag order (keep this order; omit any group that does not apply):
1. subject count and framing: 1girl, 1boy, 2girls, couple, solo, portrait, upper_body, cowboy_shot, full_body, close-up
2. subject attributes: woman, man, adult, elderly, teen, muscular, plus_size, pale_skin, dark_skin, freckles, long_hair, short_hair, wavy_hair, blonde_hair, black_hair, brown_hair, blue_eyes, glasses, beard
3. expression and mood: smile, serious, neutral_expression, parted_lips, closed_eyes, looking_at_viewer, looking_away, candid
4. clothing and accessories: white_shirt, denim_jacket, knit_sweater, turtleneck, dress, suit, jewelry, earrings, hat, sunglasses
5. pose and gesture: standing, sitting, arms_crossed, hand_on_hip, hand_in_pocket, leaning, walking, profile, three-quarter_view
6. environment and background: indoors, outdoors, studio, bedroom, cafe, street, brick_wall, plain_background, blurred_background, window
7. photographic technique: natural_light, soft_light, rim_light, golden_hour, blue_hour, studio_light, backlighting, high_key, low_key, shallow_depth_of_field, bokeh, 85mm, 50mm, wide_angle, film_grain, kodak_portra, analog_photo, digital_photography, editorial_photography
8. time of day and weather: day, night, sunset, cloudy, rainy, snow

Rules:
- Use lowercase tags joined by underscores: long_hair, not "long hair".
- Prefer established booru tag names for people and clothing, and standard photography terms for lighting, lens and film stock.
- Tag ONLY what is clearly visible. Never invent brand names, real locations, real identities, or hidden detail.
- No subjective quality words (masterpiece, best_quality, highres, 8k, ultra_detailed).
- No scores, no ratings, no safety tags, no <lora:...>, no artist names.
- Do not emit a caption sentence. Tags only.
- Aim for 15~35 tags. Drop any tag you are not reasonably sure about.
"""

TAGGER_PRESET_CHARACTER = """You are an image tagging model building an anime / illustration dataset for character training (character LoRA) and for artist-style training.

Output format:
- ONE single line of comma-separated tags. All lowercase. No sentences, no explanation, no markdown, no code fences, no numbering, no quotes.

Tag order (keep this order; omit any group that does not apply):
1. subject count and framing: 1girl, 1boy, 2girls, multiple_girls, solo, solo_focus, full_body, upper_body, close-up, portrait
2. character design: girl, boy, woman, man, chibi, long_hair, short_hair, twintails, ponytail, braid, ahoge, bangs, hair_bow, hair_ribbon, blue_hair, pink_hair, silver_hair, red_eyes, heterochromia, animal_ears, cat_ears, tail, wings, horns, eyepatch, freckles
3. expression: smile, open_mouth, blush, crying, angry, serious, surprised, closed_eyes, half-closed_eyes, :d
4. outfit (list every visible piece): serafuku, school_uniform, sailor_collar, pleated_skirt, blazer, necktie, thighhighs, boots, miko, kimono, maid, hoodie, jacket, gloves, choker, hairband, backpack
5. pose and action: standing, sitting, kneeling, walking, running, jumping, arms_up, hand_on_hip, holding_sword, holding_flower, outstretched_arm, looking_at_viewer, looking_back, from_behind
6. scene and background: outdoors, indoors, classroom, city, night, sky, cherry_blossoms, ocean, simple_background, white_background, gradient_background, window
7. art style and rendering: anime, illustration, digital_painting, sketch, lineart, flat_color, watercolor, cel_shading, thick_outlines, soft_shading, monochrome, official_art
8. camera and composition: from_above, from_below, from_side, dutch_angle, wide_shot, cowboy_shot, close-up, depth_of_field

Rules:
- Use lowercase booru-style tags joined by underscores: long_hair, twintails.
- Tag ONLY what is clearly visible. Never emit character names, series or franchise names, artist names, or copyright tags — those bind identity to a name and ruin a character LoRA.
- For a character dataset, groups 1-6 carry the identity signal and group 7 should stay generic. For an artist-style dataset, do the opposite: keep groups 1-6 about content and let groups 7-8 carry the style signal.
- No quality or meta words: masterpiece, best_quality, absurdres, highres, trending_on_pixiv, commentary_request.
- No ratings, no scores, no <lora:...>, no "(artist)" suffix, no emoji.
- Aim for 20~40 tags. Drop any tag you are not reasonably sure about.
"""

TAGGER_PRESET_SCENE = """You are an image tagging model building a dataset for background, environment and concept-art training.

Output format:
- ONE single line of comma-separated tags. All lowercase. No sentences, no explanation, no markdown, no code fences, no numbering, no quotes.

Tag order (keep this order; omit any group that does not apply):
1. people (only if actually visible): no_humans, 1girl, silhouette, crowd
2. scene type: landscape, cityscape, seascape, skyscape, interior, exterior, street, alley, market, plaza, harbor, forest, jungle, desert, mountain, canyon, field, meadow, river, lake, waterfall, cave, ruins, temple, shrine, castle, cathedral, factory, laboratory, spaceship_interior, cyberpunk_city
3. architecture and props: tower, skyscraper, bridge, street_lamp, telephone_pole, power_lines, fence, stairs, arch, pillar, stained_glass, lantern, torii, airship, neon_sign, billboard, traffic_light, market_stall
4. time of day and weather: day, night, dawn, dusk, sunset, sunrise, cloudy, overcast, fog, mist, rain, snow, storm, wind, starry_sky, moon, clouds
5. lighting and atmosphere: god_rays, volumetric_lighting, backlighting, rim_light, neon_lights, warm_lighting, cold_lighting, dusk_lighting, bioluminescence, firelight, ambient_occlusion, moody, serene, dramatic, cozy, desolate, dreamlike
6. color and composition: monochrome, pastel, vivid, muted_colors, high_contrast, panoramic, wide_shot, aerial_view, birds_eye_view, worm's_eye_view, vanishing_point, symmetry, rule_of_thirds, dutch_angle, from_above, from_below
7. style and medium: concept_art, matte_painting, digital_painting, environment_concept, photorealistic, painterly, impasto, watercolor, ink_wash, pixel_art, low_poly, 3d_render, isometric, anime_background
8. materials and surface: stone, marble, concrete, rusted_metal, wood, glass, moss, ivy, sand, snow_covered

Rules:
- Use lowercase underscore tags. Keep the scene itself as the main subject.
- Tag ONLY what is clearly visible. Never invent place names, real-world locations, franchise names, or artist names.
- If no people are visible, include no_humans and do NOT invent any.
- No subjective quality words (masterpiece, best_quality, highres, 8k).
- No ratings, no scores, no <lora:...>.
- Aim for 18~35 tags. Drop any tag you are not reasonably sure about.
"""

# 预设存在外部 JSON 里，方便直接改文本而不用动代码。
# 路径：<本节点目录>/presets/tagging_system_prompts.json
# 文件不存在时会自动生成一份（内容即下面这三套内置预设），直接编辑即可。
# 注意：改 prompt 文本在下一次执行时生效（按 mtime 热重载）；
#       新增 / 重命名 / 删除预设项需要重新加载节点（重启 ComfyUI）才会出现在下拉里。
_BUILTIN_TAG_PRESETS = {
    "photoreal": {"label": "realistic / photograph", "prompt": TAGGER_PRESET_PHOTOREAL},
    "character": {"label": "anime character / artist style", "prompt": TAGGER_PRESET_CHARACTER},
    "scene":     {"label": "scene / concept art", "prompt": TAGGER_PRESET_SCENE},
}

_PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
_PRESET_JSON = os.path.join(_PRESET_DIR, "tagging_system_prompts.json")
_PRESET_JSON_README = [
    "Qwen35 Batch Image Tagger - system prompt presets.",
    "Edit the prompt text and save; it takes effect on the next run (hot reload by mtime).",
    "Adding / renaming / removing a preset needs a node reload (restart ComfyUI) to show up in the dropdown.",
    'The preset named "custom" is reserved: it means "use the system_prompt widget on the node itself".',
    "Data set hint: photoreal = photography terms, character = booru attributes, scene = architecture + lighting + medium.",
]

# mtime -> presets 的缓存，用来实现上面那句热重载
_TAG_PRESET_CACHE = {"mtime": None, "presets": {}}


def _preset_payload(presets):
    """把 {key: {label, prompt}} 组装成写得进 JSON 的完整对象。

    这里就 strip 一次：读回来时 _parse_preset_payload 也会 strip，
    两边统一，避免「首次生成」与「之后读盘」拿到差一个尾部换行的两份文本。
    """
    return {
        "_readme": _PRESET_JSON_README,
        "presets": {
            k: {"label": v.get("label", k), "prompt": str(v.get("prompt", "")).strip()}
            for k, v in presets.items()
        },
    }


def _parse_preset_payload(data):
    """校验并归一化 JSON 内容，返回 {key: {label, prompt}}。不合法就抛异常。"""
    raw = data.get("presets") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError('顶层缺少 "presets" 对象')
    out = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key or key.lower() == "custom":
            continue                       # custom 是保留名，不允许被覆盖
        if isinstance(v, str):             # 允许简写："mykey": "整段提示词"
            v = {"label": key, "prompt": v}
        if not isinstance(v, dict):
            continue
        prompt = str(v.get("prompt") or "").strip()
        if not prompt:
            continue                       # 空 prompt 的条目直接跳过，别让它把下拉撑出个空选项
        out[key] = {"label": str(v.get("label") or key), "prompt": prompt}
    if not out:
        raise ValueError("presets 里没有一条有效预设")
    return out


def _load_tag_presets(force=False):
    """读预设 JSON（带 mtime 缓存）。文件不在就写一份内置的出来当模板。

    任何一步失败都只降级到内置预设并打日志 —— 绝不能因为一个坏 JSON
    就让整个自定义节点加载失败，连带把扩写节点也拖没。
    """
    try:
        mtime = os.path.getmtime(_PRESET_JSON)
    except OSError:
        mtime = None

    if not force and mtime is not None and mtime == _TAG_PRESET_CACHE["mtime"]:
        return _TAG_PRESET_CACHE["presets"]

    if mtime is None:
        presets = dict(_BUILTIN_TAG_PRESETS)
        try:
            os.makedirs(_PRESET_DIR, exist_ok=True)
            with open(_PRESET_JSON, "w", encoding="utf-8") as f:
                json.dump(_preset_payload(presets), f, ensure_ascii=False, indent=2)
            logger.info(f"[Qwen35] 已生成打标预设文件（可直接编辑）：{_PRESET_JSON}")
            mtime = os.path.getmtime(_PRESET_JSON)
            # 立刻读回来用，而不是把内置对象直接返回：走一遍同样的解析路径，
            # 「首次生成」与「以后读盘」拿到的文本才是逐字节一致的。
            with open(_PRESET_JSON, "r", encoding="utf-8") as f:
                presets = _parse_preset_payload(json.load(f))
        except Exception as e:
            logger.warning(f"[Qwen35] 预设文件写不出来（{e}），本次用内置预设")
            mtime = None
    else:
        try:
            with open(_PRESET_JSON, "r", encoding="utf-8") as f:
                presets = _parse_preset_payload(json.load(f))
        except Exception as e:
            logger.error(f"[Qwen35] 预设文件解析失败（{e}），回退内置预设：{_PRESET_JSON}")
            presets = dict(_BUILTIN_TAG_PRESETS)
            mtime = None                   # 置空：下次调用再试一次，别把坏内容缓存住

    _TAG_PRESET_CACHE["mtime"] = mtime
    _TAG_PRESET_CACHE["presets"] = presets
    return presets


def _tag_preset_choices():
    """下拉选项："custom" 固定第一（默认值），其余按 JSON 里的顺序。"""
    return ["custom"] + list(_load_tag_presets().keys())


def _resolve_tag_preset(preset, system_prompt):
    """按 preset 选出真正送进模型的系统提示词，返回 (文本, 预设名, 显示名)。

    preset 命中预设时直接用预设文本，**忽略 system_prompt** —— 否则下拉会被
    一个忘改的旧 widget 值悄悄顶掉，那种 bug 很难看出来。
    "custom" 或任何不认识的值（例如预设被改名/删掉）都回退到 system_prompt，
    保证老工作流不会因为预设表变了就突然拿不到提示词。
    """
    key = str(preset or "custom").strip()
    presets = _load_tag_presets()          # 每次读盘，走 mtime 缓存，改文本即时生效
    hit = presets.get(key)
    if hit is None:                        # 大小写兜底（用户手改 JSON 时常见）
        for k, v in presets.items():
            if k.lower() == key.lower():
                hit = v
                key = k                    # 归一化成 JSON 里的原名，报告里显示才规范
                break
    if hit is not None:
        return hit["prompt"], key, hit.get("label", key)
    return str(system_prompt or ""), "custom", "custom"


_IMAGE_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_TAG_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "utf-16")

# 批量打标的进度条分配。与扩写不同：加载要摊到 N 张图上，占比小得多，
# 所以给加载的宽度从 33 收到 12，剩下的全给打标循环。
_BW_UNLOAD, _BW_LOAD = 2.0, 12.0
_BW_TAG = 100.0 - _BW_UNLOAD - _BW_LOAD          # = 86.0


def _parse_image_exts(spec):
    """把 ".png,.jpg; webp" 这类写法解析成小写扩展名集合；解析不出就退回默认。

    逗号 / 分号 / 空白分隔都认，写 "png" 或 ".png" 都行 —— 这是给用户手填的框，
    不该因为少写一个点就静默一张图都扫不到（那会表现成"跑完了但什么都没生成"）。
    """
    exts = set()
    for part in re.split(r"[,;\s]+", str(spec or "")):
        part = part.strip().lower()
        if not part or part == ".":
            continue
        if not part.startswith("."):
            part = "." + part
        if part[1:].isalnum():
            exts.add(part)
    return exts or set(_IMAGE_EXTS_DEFAULT)


def _resolve_folder_path(folder_path):
    """把用户填的文件夹解析成绝对路径，支持 ComfyUI 的 input/ 与 output/ 前缀。

    解析不出来就抛 RuntimeError，不静默跳过整个任务 ——
    「路径写错了」和「文件夹里确实没有图」是两回事，前者必须让人立刻知道。
    """
    raw = str(folder_path or "").strip().strip('"').strip("'").strip()
    if not raw:
        raise RuntimeError(
            "folder_path 是空的。请填图片所在文件夹的路径，"
            r"例如 E:\datasets\mydata；"
            "也可写 input/xxx 或 output/xxx，会分别解析到 ComfyUI 的 input / output 目录。"
        )
    norm = raw.replace("\\", "/")
    for prefix, getter in (("input/", "get_input_directory"),
                           ("output/", "get_output_directory")):
        if norm.lower().startswith(prefix):
            try:
                base = getattr(folder_paths, getter)()
            except Exception:
                base = None
            if base:
                raw = os.path.join(base, norm[len(prefix):].replace("/", os.sep))
            break
    p = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(p):
        raise RuntimeError(f"文件夹不存在或不是目录：{p}")
    return p


def _scan_image_files(folder, exts, recursive):
    """扫描图片，返回**顺序固定**的绝对路径列表。

    顺序固定是硬要求：批量任务要能复现 —— 同一批图两次跑的顺序必须一样，
    否则「第 37 张出了怪结果」这类问题没法回溯。所以最后统一按
    「相对路径小写」排序，不依赖 os.listdir / os.walk 的返回顺序。
    """
    found = []
    if recursive:
        for root, dirs, names in os.walk(folder):
            dirs.sort()
            for n in names:
                if os.path.splitext(n)[1].lower() in exts:
                    found.append(os.path.join(root, n))
    else:
        for n in os.listdir(folder):
            full = os.path.join(folder, n)
            if os.path.isfile(full) and os.path.splitext(n)[1].lower() in exts:
                found.append(full)
    found.sort(key=lambda p: os.path.relpath(p, folder).replace("\\", "/").lower())
    return found


def _txt_path_for(image_path, suffix=""):
    """图片路径 -> 同目录同名 txt。suffix 插在扩展名前（xxx + "_tags" -> xxx_tags.txt）。"""
    base, _ = os.path.splitext(image_path)
    return base + str(suffix or "") + ".txt"


def _is_comfy_interrupt(exc):
    """识别 ComfyUI 的「取消」异常。

    必须显式认出来，因为 `InterruptProcessingException` 继承的是
    **BaseException**（已核对本机 comfy/model_management.py:2175），既不是
    `Exception` 也不是 `KeyboardInterrupt/SystemExit`。于是在批量循环里：

      · 只写 `except Exception`     → 它根本不会被捕获，直接冲出 tag_folder，
                                      pbar.finish() 与 _release() 都被跳过：
                                      模型留在显存里、进度条卡在中间。
      · `except BaseException` 但只白名单 KeyboardInterrupt/SystemExit
                                   → 落进「非 Exception 就原样抛出」分支，同样跳过清理。

    所以先按类名/类型认出来并走「清理后抛出」这条路径，取消才既干净又干脆。
    """
    try:
        import comfy.model_management as _mm
        cls = getattr(_mm, "InterruptProcessingException", None)
        if cls is not None and isinstance(exc, cls):
            return True
    except Exception:
        pass
    return type(exc).__name__ == "InterruptProcessingException"


def _write_text_atomic(path, text, encoding):
    """原子写 txt：先写 .qwen35tmp 再 os.replace 覆盖。

    批量写盘随时可能因为「取消」或断电中断，直接 open(w) 会留下半个文件的 txt，
    而下游训练脚本会把它当成有效标签。先写临时文件再原子替换，
    要么保留完整的旧内容，要么是完整的新内容。
    """
    tmp = path + ".qwen35tmp"
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _open_image_for_tagging(path, max_side=1280):
    """按打标的需要读图：EXIF 转正、统一 RGB、按长边等比缩小。

    三件都不是可选项：
      · EXIF：手机/相机竖拍图在文件里是横躺的，不转正模型看到的就是躺着的图。
      · RGB：PNG 带 alpha、灰度、CMYK 直接喂会报错或颜色错乱。
      · 缩图：视觉 token 数 ∝ 边长²（1024² → ~1000 tok，1536² → ~2300 tok）。
        打标不需要原分辨率，长边限到 1280 能把 prefill 砍掉近一半。
    """
    from PIL import Image, ImageOps

    im = Image.open(path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    if im.mode != "RGB":
        im = im.convert("RGB")
    cap = int(max_side or 0)
    if cap and max(im.size) > cap:
        ratio = cap / max(im.size)
        im = im.resize(
            (max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
            Image.LANCZOS,
        )
    return im


def _normalize_tag_text(text):
    """把标签输出规整成一行逗号分隔、去重保序。

    只做机械清洗、不改写内容：模型偶尔会带 markdown 围栏、项目符号、换行、
    重复标签。这些留在 txt 里，下游读标签的脚本就得各自处理一遍格式。

    仅 output_format=tags_one_line 走这里；自然语言描述请切到 raw。
    """
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"```[A-Za-z0-9_-]*", " ", s)
    s = s.replace("`", " ")
    s = re.sub(r"[\r\n]+", ",", s)
    parts, seen = [], set()
    for raw in s.split(","):
        t = raw.strip()
        # 只去真正的项目符号与「1. / 2)」式序号；不能去裸数字，
        # 否则 "2girls" 这类**主体数目标签**会被吃掉。
        t = re.sub(r"^(?:[-*\u2022\u00b7]|\d+[.)])\s+", "", t).strip()
        t = t.strip(" \t\"'\u201c\u201d\u2018\u2019")
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(t)
    return ", ".join(parts)


def _count_tags(text):
    """按逗号/换行数标签个数（tags_one_line 下就是精确值）。"""
    if not str(text or "").strip():
        return 0
    return len([p for p in re.split(r"[,\n]", str(text)) if p.strip()])


class Qwen35BatchImageTagger(Qwen35PromptEnhancer):
    """按文件夹批量打标：每张图 -> 同目录同名 .txt，系统提示词可改。

    继承只为复用 _resolve_path / _free_vram / _load / _release 与那份常驻模型
    缓存（理由见上面批量打标节点的总说明），INPUT_TYPES 与执行函数全部覆盖。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_choices(),),
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                # 三套训练场景预设，一键切换标签体系：
                #   photoreal = 真人写实（摄影术语）
                #   character = 二次元 / 画师插画（booru 属性）
                #   scene     = 场景概念图（建筑 + 光照 + 媒介）
                #   custom    = 用下面那个可编辑的 system_prompt
                # 选了内置预设时 system_prompt 会被忽略（见 _resolve_tag_preset）。
                "system_preset": (_tag_preset_choices(), {"default": "custom"}),
                # 系统提示词可改 —— 这是本节点的核心诉求之一。
                # 默认给「danbooru 风格一行逗号标签」；要自然语言描述，
                # 把这段整个换掉并把 output_format 切成 raw 即可。
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_TAGGER_SYSTEM_PROMPT,
                }),
            },
            "optional": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_TAGGER_USER_PROMPT,
                }),
                "quantization": (["none", "8bit", "4bit"], {"default": "none"}),
                "attention": (list(_ATTENTION_CHOICES), {"default": "auto"}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                # 默认 skip：**绝不覆盖已有 txt** —— 批量写盘场景唯一安全的默认值。
                # 零字节的旧 txt 视为上次写失败的残留，会被重新打标。
                "overwrite": (["skip", "overwrite"], {"default": "skip"}),
                "image_exts": ("STRING", {"default": ",".join(_IMAGE_EXTS_DEFAULT)}),
                "output_suffix": ("STRING", {"default": ""}),
                "output_encoding": (list(_TAG_ENCODINGS), {"default": "utf-8"}),
                # tags_one_line = 机械规整成一行逗号标签（配默认系统提示词）；
                # raw = 原样写模型输出（换成自然语言描述提示词时用这个）。
                "output_format": (["tags_one_line", "raw"], {"default": "tags_one_line"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 16}),
                # 默认 0.2 而不是扩写节点的 0.4：打标要稳定、可复现。
                # 设 0 则走贪心解码，同一批图两次跑结果完全一致。
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "max_image_side": ("INT", {"default": 1280, "min": 0, "max": 4096, "step": 128}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "dry_run": ("BOOLEAN", {"default": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "unload_other_models": ("BOOLEAN", {"default": True}),
                "custom_model_path": ("STRING", {"default": ""}),
                "show_progress": ("BOOLEAN", {"default": True}),
                "progress_interval": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("report", "tagged")
    FUNCTION = "tag_folder"
    CATEGORY = "Qwen35/Batch"
    OUTPUT_NODE = True

    # ----------------------------------------------------------------
    def _tag_one_image(self, model, processor, image_path, system_prompt, user_prompt,
                       max_image_side, max_new_tokens, temperature, enable_thinking,
                       seed, output_format):
        """给单张图打标。返回 (标签文本, 输出 token 数, prefill 秒, 解码秒)。

        每次调用前重设随机种子：temperature>0 时同一文件夹多次运行可复现，
        且各图种子不同（seed+i），不会整批踩同一条采样轨迹。
        """
        torch.manual_seed(int(seed))
        pil = _open_image_for_tagging(image_path, max_image_side)

        content = [
            {"type": "image", "image": pil},
            {"type": "text", "text": str(user_prompt or "")},
        ]
        messages = []
        if system_prompt and str(system_prompt).strip():
            messages.append({"role": "system", "content": str(system_prompt).strip()})
        messages.append({"role": "user", "content": content})

        # 与扩写节点同一条 chat template 路径（processor 自动处理图文混排）
        tmpl_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if _supports_enable_thinking(processor):
            tmpl_kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop("enable_thinking", None)
            inputs = processor.apply_chat_template(messages, **tmpl_kwargs)
        inputs = inputs.to(model.device)
        in_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(max_new_tokens=int(max_new_tokens))
        if float(temperature) > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(temperature),
                top_p=0.9,
                repetition_penalty=1.05,
            )
        else:
            # 贪心：打标最稳的档。日志里会明说，否则「两次结果一模一样」
            # 容易被误当成缓存命中。
            gen_kwargs.update(do_sample=False)
        eos_ids = _eos_token_ids(processor, model)
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids

        # reporter=None：这一层只负责中断检查。进度条由外层按「张」推进，
        # 若这里也按 token 推进，两个进度会互相覆盖。
        criteria = _make_progress_criteria(None, int(max_new_tokens))
        if criteria is not None:
            gen_kwargs["stopping_criteria"] = [criteria]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0
        t_pre = criteria.prefill_s if criteria is not None else 0.0
        t_dec = max(0.0, t_gen - t_pre)

        gen = out[0][in_len:]
        n_tok = int(gen.shape[0])
        text = processor.decode(gen, skip_special_tokens=True)
        text = strip_thinking(text).strip()
        if str(output_format) != "raw":
            text = _normalize_tag_text(text)
        else:
            text = text.strip()
        # 单张的输出张量/输入立刻放掉，别等下一张再回收
        del out, gen, inputs
        return text, n_tok, t_pre, t_dec

    # ----------------------------------------------------------------
    def tag_folder(self, model_name, folder_path, system_preset, system_prompt,
                   user_prompt=DEFAULT_TAGGER_USER_PROMPT, quantization="none",
                   attention="auto", enable_thinking=False, recursive=False,
                   overwrite="skip", image_exts=",".join(_IMAGE_EXTS_DEFAULT),
                   output_suffix="", output_encoding="utf-8",
                   output_format="tags_one_line", max_new_tokens=256,
                   temperature=0.2, seed=42, max_image_side=1280, limit=0,
                   dry_run=False, keep_model_loaded=False, unload_other_models=True,
                   custom_model_path="", show_progress=True, progress_interval=2.0,
                   unique_id=None):
        t_start = time.perf_counter()
        pbar = _ProgressReporter(
            node_id=unique_id, enabled=show_progress, interval=progress_interval
        )

        # 预设优先于 widget：选了预设就用预设文本（理由见 _resolve_tag_preset）
        system_prompt, preset_used, preset_label = _resolve_tag_preset(
            system_preset, system_prompt
        )

        folder = _resolve_folder_path(folder_path)
        exts = sorted(_parse_image_exts(image_exts))
        files = _scan_image_files(folder, set(exts), bool(recursive))
        scanned = len(files)
        if int(limit) > 0:
            files = files[: int(limit)]

        # 「这个 txt 要不要写」在加载模型之前全部判完：整个文件夹都已打好标时，
        # 不该白等二十秒加载一次模型。
        todo, skipped = [], []
        for src in files:
            tgt = _txt_path_for(src, output_suffix)
            if (str(overwrite) != "overwrite" and os.path.isfile(tgt)
                    and os.path.getsize(tgt) > 0):
                skipped.append(src)
            else:
                todo.append(src)

        head = [
            "[Qwen35] ========== 批量打标 ==========",
            f"  文件夹      : {folder}",
            f"  系统提示词  : {preset_used}"
            + (f"（预设：{preset_label}）" if preset_used != "custom"
               else "（custom，取节点上填写的文本）"),
            f"  扫描到      : {scanned} 张（扩展名 {'/'.join(exts)}，"
            f"{'含子目录' if recursive else '仅当前目录'}）",
            f"  待处理      : {len(todo)} 张（已有非空标签跳过 {len(skipped)} 张，"
            f"overwrite={overwrite}）",
            f"  输出        : 与图片同目录同名 .txt"
            f"（后缀 '{output_suffix}'，编码 {output_encoding}，格式 {output_format}）",
        ]
        if int(limit) > 0:
            head.append(f"  ⓘ limit={int(limit)}：只取扫描结果里的前 {int(limit)} 张")

        if dry_run:
            head.append("  ⓘ dry_run=true：只列清单，不加载模型、不写任何文件")
            for src in todo[:20]:
                head.append(
                    f"     {os.path.relpath(src, folder)}"
                    f"  ->  {os.path.relpath(_txt_path_for(src, output_suffix), folder)}"
                )
            if len(todo) > 20:
                head.append(f"     … 另有 {len(todo) - 20} 张")
            report = "\n".join(head)
            logger.info(report)
            pbar.finish()
            return {"ui": {"text": [report]}, "result": (report, 0)}

        if not todo:
            head.append("  ⚠ 没有需要处理的图片 —— 不加载模型，直接结束。")
            report = "\n".join(head)
            logger.info(report)
            pbar.finish()
            return {"ui": {"text": [report]}, "result": (report, 0)}

        path = self._resolve_path(model_name, custom_model_path)

        # ---- 1/3 卸载其他模型 + 腾出显存（与扩写节点共用同一条链路）----
        t_unload = self._free_vram(quantization, unload_other_models, pbar, _BW_UNLOAD)

        # ---- 2/3 加载模型：整个文件夹只加载这一次 ----
        t0 = time.perf_counter()
        will_reuse = (
            self._cache["key"] == (path, quantization, attention)
            and self._cache["model"] is not None
        )
        if will_reuse:
            pbar.message(
                f"复用常驻模型：{os.path.basename(path)}（{len(todo)} 张共用）"
            )
        else:
            pbar.message(
                f"正在加载模型：{os.path.basename(path)}"
                f"（量化={quantization}；之后 {len(todo)} 张共用这一次加载）"
            )
        pbar.mark(_BW_UNLOAD + 1.0)
        model, processor, freshly_loaded = self._load(path, quantization, attention)
        t_load = time.perf_counter() - t0
        pbar.mark(_BW_UNLOAD + _BW_LOAD)
        if freshly_loaded:
            pbar.message(f"模型加载完成，用时 {t_load:.1f}s")

        # ---- 3/3 逐张打标 ----
        pbar.begin_stage(_BW_UNLOAD + _BW_LOAD, _BW_TAG)
        pbar.message(
            f"开始打标：{len(todo)} 张，温度 {float(temperature):.2f}"
            + ("（贪心解码，同一批两次跑结果一致）" if float(temperature) <= 0 else "")
        )

        ok = n_fail = 0
        tok_sum = tag_sum = 0
        t_tag_total = 0.0
        first_prefill = None
        failures, examples = [], []

        for idx, src in enumerate(todo, start=1):
            name = os.path.relpath(src, folder)
            target = _txt_path_for(src, output_suffix)
            t_one = time.perf_counter()
            text, n_tok, t_pre, t_dec = "", 0, 0.0, 0.0
            err = None
            try:
                text, n_tok, t_pre, t_dec = self._tag_one_image(
                    model, processor, src, system_prompt, user_prompt,
                    int(max_image_side), int(max_new_tokens), float(temperature),
                    bool(enable_thinking), int(seed) + idx, output_format,
                )
                if not text:
                    raise RuntimeError(
                        "输出为空（可能整段都是思考块，或第一个 token 就是 EOS）"
                    )
                _write_text_atomic(target, text, output_encoding)
            except BaseException as e:
                # 取消必须立刻中止整批；其余异常只算这一张失败。
                if _is_comfy_interrupt(e) or isinstance(e, (KeyboardInterrupt, SystemExit)):
                    pbar.finish()
                    if not keep_model_loaded:
                        self._release()
                    logger.warning(
                        f"[Qwen35] 已取消：处理到第 {idx}/{len(todo)} 张"
                        f"（前面已写好的 txt 保留，未完成的那张不会留下半个文件）"
                    )
                    raise
                if not isinstance(e, Exception):
                    raise
                err = f"{type(e).__name__}: {e}"
            dt = time.perf_counter() - t_one
            t_tag_total += dt

            if err is None:
                ok += 1
                tok_sum += n_tok
                n_tag = _count_tags(text)
                tag_sum += n_tag
                if first_prefill is None:
                    first_prefill = t_pre
                if len(examples) < 3:
                    examples.append((name, text[:160]))
                logger.info(
                    f"[Qwen35] [{idx}/{len(todo)}] \u2713 {name} -> {n_tag} 标签"
                    f" / {n_tok} tok / {dt:.1f}s"
                    + (f"（prefill {t_pre:.2f}s、解码 {t_dec:.1f}s"
                       f" = {n_tok / t_dec:.1f} tok/s）" if t_dec > 0 else "")
                )
                pbar.tick(idx, len(todo), note=f"{dt:.1f}s/张", unit="张")
            else:
                n_fail += 1
                failures.append((name, err))
                logger.warning(f"[Qwen35] [{idx}/{len(todo)}] \u2717 {name} -> {err}")
                pbar.tick(idx, len(todo), note="失败", unit="张")

        pbar.finish()

        t_release = 0.0
        if not keep_model_loaded:
            t0 = time.perf_counter()
            self._release()
            t_release = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start
        per = (t_tag_total / ok) if ok else 0.0
        rate = (tok_sum / t_tag_total) if t_tag_total > 0 else 0.0

        lines = head + ["", "[Qwen35] ========== 打标汇总 =========="]
        lines.append(f"  成功 / 失败 : {ok} / {n_fail}")
        if ok:
            lines.append(f"  标签合计    : {tag_sum} 个（平均 {tag_sum / ok:.1f} 个/张）")
            lines.append(f"  输出 token  : {tok_sum}（平均 {tok_sum / ok:.1f} tok/张）")
        lines.append(
            f"  打标耗时    : {t_tag_total:6.2f}s"
            + (f"（{per:.2f}s/张，{rate:.1f} tok/s）" if ok else "")
        )
        if first_prefill is not None:
            lines.append(
                f"  首张 prefill: {first_prefill:6.2f}s"
                f"（含 CUDA 预热与 kernel 编译，后续张比它快是正常的）"
            )
        lines += [
            f"  加载模型    : {t_load:6.2f}s  ({'本次新加载' if freshly_loaded else '复用常驻'})",
            f"  卸载其他模型: {t_unload:6.2f}s",
            f"  卸载打标模型: {t_release:6.2f}s",
            "  --------------------------------",
            f"  合计        : {t_total:6.2f}s",
        ]
        for nm, txt in examples:
            lines.append(f"  例          : {nm} -> {txt}")
        if failures:
            lines.append(
                f"  ⚠ 失败清单（{len(failures)} 张；原图与已有 txt 均未被改动）:"
            )
            for nm, why in failures[:20]:
                lines.append(f"     - {nm}：{why}")
            if len(failures) > 20:
                lines.append(f"     … 另有 {len(failures) - 20} 张")
        else:
            lines.append("  ✓ 全部成功")

        report = "\n".join(lines)
        logger.info(report)
        return {"ui": {"text": [report]}, "result": (report, ok)}


# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer": Qwen35PromptEnhancer,
    "Qwen35BatchImageTagger": Qwen35BatchImageTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer": "Qwen3.5 / Qwen3-VL Prompt Enhancer",
    "Qwen35BatchImageTagger": "Qwen3.5 Batch Image Tagger (txt)",
}
