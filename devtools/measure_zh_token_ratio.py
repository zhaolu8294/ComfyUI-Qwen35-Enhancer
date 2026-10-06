# -*- coding: utf-8 -*-
"""实测：中文译文的 token 数是英文原文的几倍（只加载分词器，纯 CPU，不碰显存）。

用途：把"双语慢 3 倍"从体感变成数字。第二阶段的解码量 = 中文 token 数，
它就是双语全部额外成本的来源（prefill 只有几百 token，可忽略）。

跑法：
  E:\\AI\\ComfyUI-aki-v3\\python\\python.exe measure_zh_token_ratio.py
"""
import os
import re
import sys

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\prompt_generator\Qwen3-VL-8B-Instruct"
NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"

sys.path.insert(0, NODE_DIR)

# 从节点源码里取两个 system prompt（不 import torch / folder_paths）
src = open(os.path.join(NODE_DIR, "nodes.py"), encoding="utf-8").read()


def grab(name):
    m = re.search(name + r' = """(.*?)"""', src, re.S)
    return m.group(1) if m else ""


DEFAULT_SYS = grab("DEFAULT_SYSTEM_PROMPT")
TRANS_SYS = grab("TRANSLATE_SYSTEM_PROMPT")

from transformers import AutoTokenizer  # noqa: E402

print(f"[1] 加载分词器：{MODEL_DIR}")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"    vocab = {tok.vocab_size}  class = {type(tok).__name__}")


def nt(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------
# 三组真实的 H3 英文提示词 + 忠实的简体中文译文
# （中文由我翻译，不是机器翻的，所以比例反映"人读得顺"的译文长度）
# --------------------------------------------------------------------------
PAIRS = []

PAIRS.append(("4s / 单镜头", """integrated_multimodal_description: [Shot 1] Cinematic, medium-wide shot of a young woman in a beige trench coat standing at a rain-soaked crosswalk at dusk. The camera pushes in with small amplitude at slow speed toward the folded letter in her hands. Neon signage reflects in the puddles as passing cars blur past her. She raises her head slowly, rain streaming down her face, and her fingers tighten around the paper. Diegetic audio: rain hissing on asphalt, the wet rustle of her coat, distant traffic.

overall_soundscape: Steady rain on pavement and passing traffic build a cold, damp ambience. Footsteps splash through shallow water.

non_diegetic_music: A solo cello holds a low sustained note, joined by sparse piano at 00:03.400 as the camera settles; the texture thins out and fades before the cut.""",
"""integrated_multimodal_description: [Shot 1] 电影感，中景镜头拍摄一位身穿米色风衣的年轻女子站在暮色中积水的斑马线上。摄影机以小幅慢速向推，推向她手中折起的信纸。霓虹招牌倒映在水洼中，过往车辆在她身后虚化成流光。她缓缓抬起头，雨水顺着脸颊流下，手指收紧纸面。画内音效：雨水打在柏油路面上的嘶嘶声、大衣被打湿的窸窣声、远处车流声。

overall_soundscape: 持续的雨声与往来的车流营造出寒冷潮湿的氛围。脚步踏过浅水溅起水花。

non_diegetic_music: 大提琴持续低音，在 00:03.400 摄影机停稳时加入稀疏的钢琴；织体逐渐变薄并在切镜前淡出。"""))

PAIRS.append(("8s / 双镜头", """integrated_multimodal_description: [Shot 1] 3D CG, wide shot of a snow-covered mountain valley at first light. The camera trucks right at slow speed, revealing a lone wooden cabin with smoke rising from its chimney. Snow drifts across the frame in the wind while the first sunbeam climbs the far ridge. At 00:04.200, the camera cuts to a close-up of a gloved hand pushing open the cabin door; warm interior light spills onto the snow. Diegetic audio: wind gusting, snow crunching, the creak of a wooden door, a kettle whistling inside.

overall_soundscape: Cold wind dominates, with snow crunching underfoot and a distant avalanche rumble. The interior kettle whistle briefly overrides the wind.

non_diegetic_music: A low string drone underpins the valley, with a soft harp figure entering at 00:04.200 and rising in volume until the cut.""",
"""integrated_multimodal_description: [Shot 1] 3D CG，广角镜头拍摄破晓时分被积雪覆盖的山谷。摄影机以慢速向右移镜，逐渐显现一栋孤零零的木屋，烟囱里升起炊烟。雪花在风中飘过画面，第一缕阳光正爬上远处山脊。在 00:04.200，镜头切到一只戴着手套的手推开木屋门的特写；温暖的室内灯光洒落在雪地上。画内音效：风声阵阵、积雪被踩碎的声响、木门吱呀声、屋内水壶的鸣笛声。

overall_soundscape: 寒风主导整个声场，伴随脚下积雪的碎裂声与远处雪崩的闷响。屋内水壶的鸣笛声短暂盖过风声。

non_diegetic_music: 低音弦乐持续铺底，在 00:04.200 加入轻柔的竖琴音型，音量逐渐升高直至切镜。"""))

PAIRS.append(("12s / 三镜头 + 对白", """integrated_multimodal_description: [Shot 1] Live-action, extreme close-up of a chef's hands folding dumplings on a flour-dusted steel counter. The camera arcs left at slow speed, small amplitude, keeping the fingers centred. Steam curls up from a bamboo steamer at the left edge of frame. At 00:03.800, the camera cuts to a medium shot of the chef lifting a finished dumpling with chopsticks, then At 00:07.400 the camera cuts to a wide shot of the whole kitchen as a small crowd applauds. The chef, a middle-aged woman with a hoarse voice (S1) says: <d>[English] This recipe is four generations old.</d> Diegetic audio: flour puffing, chopsticks clicking, steamer hissing, applause swelling.

overall_soundscape: A busy professional kitchen with metallic clatter, steam hiss and low conversational murmur. Applause rises sharply in the final shot.

non_diegetic_music: Plucked erhu with light percussion at a moderate walking tempo; percussion drops out at 00:07.400 for a sustained erhu note under the applause.""" ,
"""integrated_multimodal_description: [Shot 1] 实拍，极特写镜头拍摄一位厨师在撒满面粉的不锈钢台面上包饺子。摄影机小幅慢速向左环绕，始终让手指保持在画面中央。画面左侧边缘的竹蒸笼里升起蒸气。在 00:03.800，镜头切到厨师用筷子夹起一只包好的饺子的中景，随后在 00:07.400，镜头切到整个厨房的广角画面，一小群人正在鼓掌。厨师是一位嗓音沙哑的中年女性（S1）说：<d>[English] This recipe is four generations old.</d>（对白：这个配方传了四代。）画内音效：面粉扬起的声音、筷子碰撞声、蒸笼的嘶嘶声、逐渐高涨的掌声。

overall_soundscape: 繁忙的专业厨房，充满金属碰撞声、蒸气的嘶嘶声与低声交谈。最后的镜头里掌声骤然升高。

non_diegetic_music: 二胡拨弦配轻打击乐，中速行板；在 00:07.400 打击乐退出，留下一段持续的二胡长音托在掌声之下。"""))

print()
print("=" * 92)
print(f"{'场景':<16}{'英文字符':>9}{'英文tok':>9}{'中文字符':>9}{'中文tok':>9}"
      f"{'字符比':>8}{'tok 比':>8}")
print("=" * 92)

rows = []
for tag, en, zh in PAIRS:
    en_c, en_t = len(en), nt(en)
    zh_c, zh_t = len(zh), nt(zh)
    rows.append((tag, en_c, en_t, zh_c, zh_t))
    print(f"{tag:<16}{en_c:>9}{en_t:>9}{zh_c:>9}{zh_t:>9}"
          f"{zh_c / en_c:>7.2f}x{zh_t / en_t:>7.2f}x")

print("=" * 92)
avg = sum(r[4] / r[2] for r in rows) / len(rows)
print(f"平均：中文 token / 英文 token = {avg:.2f}x")
print(f"     中文字符 / 英文字符     = {sum(r[3] for r in rows) / sum(r[1] for r in rows):.2f}x")
print(f"     （中文每个 token 约 {sum(r[3] for r in rows) / sum(r[4] for r in rows):.2f} 字；"
      f"英文每个 token 约 {sum(r[1] for r in rows) / sum(r[2] for r in rows):.2f} 字符）")

print()
print("[2] prefill 侧的固定开销")
print(f"    DEFAULT_SYSTEM_PROMPT（英文段）: {len(DEFAULT_SYS)} 字符 / {nt(DEFAULT_SYS)} tok")
print(f"    TRANSLATE_SYSTEM_PROMPT（中文段）: {len(TRANS_SYS)} 字符 / {nt(TRANS_SYS)} tok")
wrap = "----- BEGIN H3 PROMPT -----\n\n----- END H3 PROMPT -----"
print(f"    中文段的 BEGIN/END 包裹: {nt(wrap)} tok")

print()
print("[3] 推算整轮耗时倍数（只看解码，prefill 几十~几百 tok 可忽略）")
print(f"    {'英文tok':>8}{'中文tok':>9}{'纯解码倍数':>12}")
for tag, en_c, en_t, zh_c, zh_t in rows:
    t_en = en_t + nt(DEFAULT_SYS)
    t_zh = zh_t + nt(TRANS_SYS) + nt(wrap) + en_t
    print(f"    {en_t:>8}{zh_t:>9}{t_zh / t_en:>11.2f}x"
          f"   （中文段还多花 {t_zh - en_t} tok 的 prefill：翻译指令 + 英文原文）")

print()
print("[4] 结论")
print("    第二阶段的解码量 = 中文 token 数，它就是双语全部额外成本的来源。")
print("    若实测比例明显低于 1.6，可以把 nodes.py 的 _ZH_TOK_RATIO 调低以收紧预算；")
print("    反之则调高，避免译文被预算截断。")
