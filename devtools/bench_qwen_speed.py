# -*- coding: utf-8 -*-
"""扩写耗时诊断 —— Part B：分阶段计时（需要 GPU）

分别测量：模型加载 / 首次生成 / 二次生成（模型已驻留）/ 带图生成
用来定位到底是「反复加载模型」慢，还是「生成本身」慢。
"""
import os
import time
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\prompt_generator\Qwen3-VL-8B-Instruct"

SYS_PROMPT = """[MiniMax H3 prompt-writing skill rules — follow strictly]

You are a MiniMax H3 video generation prompt rewriting expert.
Rewrite the user input into ONE English prompt following the H3 base-en specification.

integrated_multimodal_description: [Shot 1] <style>, <composition>, ...
overall_soundscape: ...
non_diegetic_music: ...

Plain English only. No thinking."""

USER_TEXT = "一只橘猫在清晨的咖啡馆窗边打哈欠，阳光斜射进来，背景是巴黎街景。"
MAX_NEW = 512

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image


def run(model, proc, images, tag):
    content = []
    for im in images:
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": USER_TEXT})
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": content},
    ]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    in_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW, do_sample=False,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    out_len = out[0][in_len:].shape[0]
    print(f"  {tag:<22} 输入{in_len:>5} tok  输出{out_len:>4} tok  耗时{dt:>6.2f}s  "
          f"解码{out_len/dt:>6.1f} tok/s")
    return dt, in_len, out_len


print("=" * 72)
print("扩写耗时分阶段诊断 (Qwen3-VL-8B-Instruct / 8bit)")
print("=" * 72)

QUANT = os.environ.get("QUANT", "4bit")
print(f"\n[阶段 1] 模型加载 (量化={QUANT})")
if QUANT == "8bit":
    QC = BitsAndBytesConfig(load_in_8bit=True)
else:
    QC = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=QC,
    dtype="auto",
    attn_implementation="sdpa",
)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL_DIR)
torch.cuda.synchronize()
t_load = time.perf_counter() - t0
print(f"  加载耗时: {t_load:.2f}s")
print(f"  显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB "
      f"(峰值 {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")

print("\n[阶段 2] 生成耗时")
print("  " + "-" * 68)
t_txt1, i1, o1 = run(model, proc, [], "纯文本 (首次)")
t_txt2, i2, o2 = run(model, proc, [], "纯文本 (二次)")
big = Image.new("RGB", (1024, 1024), (128, 128, 128))
t_img1, i3, o3 = run(model, proc, [big], "1图 1024x1024")
t_img4, i4, o4 = run(model, proc, [big] * 4, "4图 1024x1024")

print("\n" + "=" * 72)
print("结果汇总")
print("=" * 72)
print(f"  模型加载一次的成本          : {t_load:>7.2f}s")
print(f"  纯文本生成 (模型已常驻)      : {t_txt2:>7.2f}s")
print(f"  1图 生成 (模型已常驻)        : {t_img1:>7.2f}s")
print(f"  4图 生成 (模型已常驻)        : {t_img4:>7.2f}s")
print()
print(f"  >>> 若每次重载模型，单次耗时 ≈ {t_load + t_txt2:.1f}s")
print(f"  >>> 若模型常驻，   单次耗时 ≈ {t_txt2:.1f}s")
print(f"  >>> keep_model_loaded=True 可省 {t_load:.1f}s ({t_load/(t_load+t_txt2)*100:.0f}%)")
print()
print(f"  首次 vs 二次生成差异         : {t_txt1-t_txt2:+.2f}s (预热/编译开销)")
print(f"  显存峰值                     : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
