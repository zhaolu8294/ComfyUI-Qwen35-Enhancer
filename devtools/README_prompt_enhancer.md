# H3 Prompt Enhancer 工作流（已实测可用）

基于 **MiniMax 官方 h3-prompt-writing skill**（`MiniMax-AI/MiniMax-H3/skills/h3-prompt-writing`） + 本地 Qwen3-VL 模型实现的提示词扩写器。**ComfyUI 0.37.0 + aki v3 环境实测已装插件可直接拖入使用**。

---

## 一、已识别的资产（盘点 `E:\AI\ComfyUI-aki-v3\`）

### LLM 文本改写模型（`ComfyUI\models\`）

| 路径 | 大小 | 用途 | 节点 |
|------|------|------|------|
| `prompt_generator/Qwen3-VL-8B-Instruct/` | 17 GB | **首选** 多模态，完整 HF 文件夹（config + tokenizer 齐全） | `Qwen3_VQA` |
| `text_encoders/qwen_3_8b_fp8mixed.safetensors` | 8.6 GB | 单文件，缺 config.json，需要补齐才能用 `LLM_local_loader` | 待补 config 后可用 |
| `text_encoders/qwen_3_4b.safetensors` | 8 GB | 单文件，缺 config.json | 同上 |
| `text_encoders/qwen3vl_4b_bf16.safetensors` | 8.9 GB | VL 4B，单文件 | 同上 |

### 反推/视觉模型

| 路径 | 用途 | 节点 |
|------|------|------|
| `LLM/Florence-2-large/` | 看图反推（pytorch_model.bin 格式） | `Florence2ModelLoader` + `Florence2Run` |
| `LLM/llama-joycaption-beta-one-hf-llava/` | JoyCaption 多模态反推 | 暂未集成进工作流（备选） |

### 已装插件（`ComfyUI\custom_nodes\`）

| 插件 | 节点类 | 状态 |
|------|--------|------|
| `ComfyUI_Qwen3-VL-Instruct` | `Qwen3_VQA` | ✅ 已装 |
| `ComfyUI_QwenVL_PromptCaption` | `Qwen3Caption` 等 | ✅ 已装（备选反推） |
| `ComfyUI-KJNodes` | `JoinStringMulti` | ✅ 已装（字符串拼接） |
| `ComfyUI-Florence2` | `Florence2ModelLoader`/`Florence2Run` | ✅ **本次新增** |
| `comfyui_LLM_party` | `LLM_local_loader` 等 | ✅ **本次新增**（暂未用到，等用户补 qwen config.json） |
| `ComfyUI-GGUF` | `GGUFLoader` 等 | ✅ 已装（GGUF 量化用） |

### H3 全部模型（已装）

| 文件 | 路径 | 用途 |
|------|------|------|
| `minimax_h3_fl2va_int8_convrot.safetensors` | `diffusion_models/minimax_h3/` | T2VA/I2V 主模型 |
| `minimax_h3_ref2va_int8_convrot.safetensors` | `diffusion_models/minimax_h3/` | Ref2V 主模型 |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders/minimax_h3/` | H3 文本编码器 |
| `minimax_h3_video_vae_fp16.safetensors` | `vae/minimax_h3/` | 视频 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/minimax_h3/` | 音频 VAE |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors` | `loras/H3/` | 4-step Turbo |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | `loras/H3/` | 8-step Turbo |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | `loras/H3/` | Ref2V Turbo |

---

## 二、本次交付的两个工作流

### A. `prompt_enhancer_text_only.json` — 纯文本改写

```
[PrimitiveString: Skill 规则] ─┐
                                ├─→ JoinStringMulti ─→ Qwen3_VQA ─→ show_text_party
[PrimitiveString: 用户简单提示] ─┘
```

**5 个节点 / 4 条 link**。在 `PrimitiveString（用户输入）` 节点里写中文/英文都行，模型会自动按 H3 规范扩写。

### B. `prompt_enhancer_image_text.json` — 图片可选 + 文字

```
[LoadImage] ─→ [Florence2Run] ─┐
                               ├─→ JoinStringMulti ─┐
[PrimitiveString: Skill 规则] ─┤                    ├─→ JoinStringMulti ─→ Qwen3_VQA ─→ show_text_party
[PrimitiveString: 用户输入] ────┘                    │
                              (caption+user) ───────┘
```

**9 个节点 / 8 条 link**。Florence-2 自动反推图片描述，附加到用户输入后改写。

---

## 二·五、v2 修正记录（2026-10-06 20:10）

首次跑报错 `Failed to validate prompt for output 9` 已修复。根因与修正：

| 问题 | 根因 | 修正 |
|------|------|------|
| `model: 'none' not in [...]`<br>`quantization: False not in [...]`<br>`temperature: 1920.0 bigger than max of 1`<br>`seed: 'eager' 无法转 int`<br>`max_pixels: -1 smaller than min` | `Qwen3_VQA` 的 `text` 虽是连接输入，但 **无 `forceInput` 标记**，ComfyUI 要求 `widgets_values[0]` 仍放它的占位符。少了这个占位符导致后面 **全部左移一位** | 在 `widgets_values` 开头补 `""` 占位，变成 10 个值：`["", model, quantization, keep_model_loaded, temperature, max_new_tokens, min_pixels, max_pixels, seed, attention]` |
| `PrimitiveString` 装不下 2000+ 字符的多行 skill 规则 | 单行 widget | 换成 `PrimitiveStringMultiline` |
| `LoadImage` 默认 `example.png` 不存在 | input 目录没有该文件 | 改成实际存在的 `1 (15).png`，用户可自行 `Ctrl+A` 换图 |

**关键规律（写 ComfyUI 工作流通用）**：
- 被 link 连接的 widget，如果**没有** `forceInput: True`（如 `Qwen3_VQA.text`），`widgets_values` 里**必须留占位符**
- 有 `forceInput: True` 的（如 `JoinStringMulti.string_1`、`show_text_party.text`）**不占** `widgets_values`
- 纯 widget（没被 link 的）直接按声明顺序填值

---

## 三、节点连线详解（以 text_only 为例）

| ID | 节点 | widget 值 | 备注 |
|----|------|-----------|------|
| #1 | `PrimitiveString`（Skill） | MiniMax H3 prompt-writing skill 全文 | 用户可编辑改写规则 |
| #2 | `PrimitiveString`（User） | `"雨夜霓虹街道上的赛博朋克猫..."` | 简单中文提示词 |
| #3 | `JoinStringMulti` | `inputcount=2, delimiter="\n\n"` | 拼接 string_1+string_2 |
| #4 | `Qwen3_VQA` | 见下方 widget 表 | 主改写器 |
| #5 | `show_text_party` | （empty） | 预览输出 |

### `Qwen3_VQA` widget 配置（用户可改）

| 字段 | 值 | 说明 |
|------|----|----|
| `model` | `Qwen3-VL-8B-Instruct` | 默认加载 `prompt_generator/Qwen3-VL-8B-Instruct/`；可在 UI 下拉切 4B/8B、FP8/原始、Instruct/Thinking |
| `quantization` | `none` | 显存够用不量化；12GB 显存请选 `8bit` |
| `keep_model_loaded` | `False` | 跑完释放 VRAM（避免和 H3 抢显存） |
| `temperature` | `0.4` | 较低，更稳 |
| `max_new_tokens` | `1920` | 限制 H3 提示词长度（4-15 秒视频建议 60-200 词 ≈ 800 tokens） |
| `seed` | `-1` | 随机 |

### `Florence2Run` widget 配置（image_text 工作流）

| 字段 | 值 | 说明 |
|------|----|----|
| `task` | `more_detailed_caption` | 详细反推；可选 `caption`（短）、`detailed_caption`（中） |
| `fill_mask` | `True` | 仅 `region_*` 任务相关，留 True 即可 |
| `max_new_tokens` | `1024` | 默认 |

---

## 四、把改写器接到 H3 主链路

两个工作流的 `show_text_party` 输出端口是 STRING 列表，可以**直接连到 H3 节点的 `prompt` 输入**：

```
prompt_enhancer_text_only.json
    └─ show_text_party ─→ (STRING) ─→ [H3 主工作流] MiniMaxH3ImageToVideo.prompt
```

或者直接拖入本目录另外两个文件 `h3_t2va_with_llm.json`（已用 `int8_convrot` 模型名重写过），把其中 `show_text_party` 替换成这个工作流的输出。

---

## 五、模型可替换清单

`Qwen3_VQA` 节点的下拉支持 8 个模型（用户可在 UI 切换）：

| 模型名 | 适用场景 | VRAM |
|--------|----------|------|
| `Qwen3-VL-4B-Instruct-FP8` | 轻量、首选（4090 8GB+ 显存） | ~6GB |
| `Qwen3-VL-8B-Instruct-FP8` | 显存够用，质量更高 | ~10GB |
| `Qwen3-VL-4B-Instruct` | 原始精度（4090 16GB+） | ~8GB |
| `Qwen3-VL-8B-Instruct` | 原始精度（4090 24GB） | ~16GB |
| `Qwen3-VL-4B-Thinking-FP8` | 长链思考版（适合复杂场景） | ~6GB |
| `Qwen3-VL-8B-Thinking-FP8` | 同上，更大 | ~10GB |
| `Qwen3-VL-4B-Thinking` | 原始精度 | ~8GB |
| `Qwen3-VL-8B-Thinking` | 原始精度 | ~16GB |

---

## 六、避坑清单

- ⚠️ `Qwen3_VQA` 节点的 `system prompt` 是硬编码 `"You are QwenVL..."`，**所以 H3 规则必须塞到 `text` 开头**（已在生成时做好）
- ✅ Florence-2-large 已在 `LLM/Florence-2-large/`，**不用再下载**
- ✅ Qwen3-VL-8B-Instruct 完整模型已在 `prompt_generator/Qwen3-VL-8B-Instruct/`，**不用再下载**
- ⚠️ 如果跑 H3 视频生成时显存吃紧，把 `Qwen3_VQA` 的 `keep_model_loaded` 改成 `False`，改写完会自动卸载 LLM，腾出显存给 H3
- ⚠️ 提示词长度别超过 `max_new_tokens=1920`（token 数），约等于 1500 英文词，足够 4-15 秒 H3 视频
- ⚠️ Florence-2 第一次加载会从 transformers 模型缓存目录拉东西，确保网络通**

---

## 八、配套文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `prompt_enhancer_text_only.json` | ~6 KB | 纯文本改写 |
| `prompt_enhancer_image_text.json` | ~10 KB | 图片+文本改写 |
| `h3_t2va_with_llm.json` | ~22 KB | H3 T2VA 主链路（已用本地模型文件名修正） |
| `h3_i2va_with_llm.json` | ~23 KB | H3 I2V 主链路 |
| `h3_ref2va_with_llm.json` | ~23 KB | H3 Ref2V 主链路 |
| `h3_system_prompt.txt` | ~4 KB | 备用 system prompt |