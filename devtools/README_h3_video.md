# MiniMax H3 本地工作流（LLM-Party 提示词改写）

> 把官方 h3-prompt-writing skill 的规则做成本地 ComfyUI 工作流。
> 你只需要写中文/英文提示词，Qwen3-8B 本地模型会自动改写成符合 H3 规范的英文三段式 prompt，再喂给 H3 生成视频。

---

## 📁 目录内容

```
h3_workflow/
├── h3_t2va_with_llm.json      # 纯文字生成视频（最常用）
├── h3_i2va_with_llm.json      # 图生视频（首帧 + 提示词）
├── h3_ref2va_with_llm.json    # 参考图/视频生成（多模态）
├── h3_system_prompt.txt       # 改写器系统提示词（已嵌入到工作流里，可单独查阅）
├── h3_t2v_template.json       # 官方 H3 T2V 模板备份（仅参考）
└── README.md                  # 本文件
```

---

## ✅ 前置依赖（必须已装好）

### 1. ComfyUI 主程序 ≥ 0.30.0

H3 原生节点在 PR #15224 合并入主仓：

```bash
cd ComfyUI
git pull
pip install -r requirements.txt
```

### 2. ComfyUI-LLM-Party 插件

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/heshengtao/comfyui_LLM_party.git
cd comfyui_LLM_party
pip install -r requirements.txt
```

> ⚠️ 如果用 GGUF 模型，还要装 `llama-cpp-python`，建议直接拿官方 wheel：
> ```
> pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
> ```
> （按你的 Python 版本选对应 wheel）

### 3. 提示词改写模型（本地 LLM）

推荐 `Qwen3-8B-Instruct-Q5_K_M.gguf`，下载到：

```
ComfyUI/models/LLM/Qwen3-8B-Instruct-Q5_K_M.gguf
```

下载源（任选）：
- HuggingFace: `https://huggingface.co/bartowski/Qwen3-8B-Instruct-GGUF`
- 国内镜像: `https://hf-mirror.com/bartowski/Qwen3-8B-Instruct-GGUF`

> 文件约 5.5 GB，Q5_K_M 量化需要 ~6 GB 显存。
> 如果显存不够，换 `Qwen3-4B-Instruct-Q4_K_M.gguf`（2.5GB / ~4GB 显存）

### 4. H3 全套模型（已开源，2026-08-03）

```
ComfyUI/models/
├── diffusion_models/
│   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors   # 19.5 GB
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       # 14.6 GB
├── vae/
│   ├── minimax_h3_video_vae_int8_convrot.safetensors       # 2.6 GB
│   └── minimax_h3_audio_vae_fp32.safetensors              # 577 MB
└── loras/
    └── minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors  # 1.8 GB（可选）
```

下载源：`https://huggingface.co/Comfy-Org/MiniMax-H3`（国内走 `hf-mirror.com`）

---

## 🚀 使用方法

### 步骤 1：把工作流拖入 ComfyUI

把 `h3_*.json` 文件直接拖进 ComfyUI 画布（或 File → Load）。

### 步骤 2：修改用户提示词

找到 **`LLM`** 节点（左边第一个非加载器节点），双击修改 `user_prompt` widget 字段：

- 中文输入：`一只橘猫在清晨的咖啡馆窗边打哈欠，阳光斜射进来，背景是巴黎街景。`
- 英文输入：直接写也可以，LLM 会按 H3 规范补全镜头语言

### 步骤 3：（仅 I2VA / Ref2VA）上传图片

- **I2VA**：把首帧图重命名为 `first_frame.png`，放到 `ComfyUI/input/`
- **Ref2VA**：把参考图重命名为 `reference.png`，放到 `ComfyUI/input/`

### 步骤 4：调整时长

找到 **`PrimitiveFloat`** 节点（标着 `value=5.0`），改为你想要的秒数（4–15 之间）。

### 步骤 5：点击 Queue 跑

第一次跑会先加载 GGUF 模型（约 10–30 秒），再调 LLM 改写提示词（1–3 秒），最后 H3 采样生成。

---

## 🎯 三个工作流的区别

| 工作流 | 用途 | 需要的输入 | 输出 |
|--------|------|-----------|------|
| `h3_t2va_with_llm.json` | 纯文字 → 视频 | 仅文字提示词 | 768p 视频 + 原生音频 |
| `h3_i2va_with_llm.json` | 图 → 视频 | 首帧图 + 文字提示词 | 768p 视频 + 原生音频（从首帧演化） |
| `h3_ref2va_with_llm.json` | 多模态参考 → 视频 | 参考图 + 文字提示词 | 768p 视频 + 原生音频（参考图风格/主题） |

---

## ⚙️ 关键参数速查

| 节点 | 字段 | 默认值 | 含义 |
|------|------|--------|------|
| `PrimitiveFloat` | value | 5.0 | 视频时长（秒），snap 到 `17k+5` 帧（24fps） |
| `BasicScheduler` | steps | 8 | 采样步数（配 turbo LoRA，4 步也能跑） |
| `BasicScheduler` | scheduler | beta | 调度器 |
| `KSamplerSelect` | sampler_name | euler | 采样器 |
| `MiniMaxH3SigmaShift` | shift_video | 12.0 | 视频流时间步偏移 |
| `MiniMaxH3SigmaShift` | shift_audio | 3.0 | 音频流时间步偏移 |
| `RandomNoise` | noise_seed | 42 | 随机种子 |
| `GGUFLoader` | max_ctx | 4096 | Qwen3 上下文窗口 |
| `GGUFLoader` | gpu_layers | 31 | GPU offload 层数 |

---

## 🛠️ 提示词改写器到底改了什么？

`h3_system_prompt.txt` 里定义了 10 条强制规则（基于官方 `base-en.txt` 规范）：

1. **风格开场**：第一句必须 `Live-action, cinematic,` 等风格词开头
2. **镜头标注**：`[Shot 1]` 无时间戳，后续 `[Shot 2] At MM:SS.mmm,` 严格递增
3. **镜头三要素**：类型 + 振幅 + 速度（`push in with small amplitude at slow speed`）
4. **现在进行时**：动作必须 walking, flowing, rotating 这种进行时
5. **音频线索**：至少一处 diegetic audio（脚步声、风声、布料摩擦）
6. **对话格式**：`(S1) says: <d>[English] ...</d>`
7. **字幕格式**：英文双引号包裹原文
8. **overall_soundscape**：1–4 句环境音
9. **non_diegetic_music**：1–3 句纯器乐描述（不许用"beautiful"这类抽象词）
10. **时长匹配**：内容长度必须匹配用户给的视频时长（4–15 秒）

LLM 输出严格遵循 `integrated_multimodal_description: ...` / `overall_soundscape: ...` / `non_diegetic_music: ...` 三段式。

---

## 🔧 故障排查

### LLM 节点报错 / 不返回结果

1. 检查 GGUF 模型路径是否正确（默认 `ComfyUI/models/LLM/Qwen3-8B-Instruct-Q5_K_M.gguf`）
2. 打开 ComfyUI 控制台，看是否有 `llama-cpp-python not found` —— 需要装
3. 显存不够 → 把 `gpu_layers` 调到 20 或更小

### 改写后的提示词不符合 H3 规范

- LLM 输出的 prompt 直接喂给 `MiniMaxH3ImageToVideo` 节点的 `prompt` 字段
- 如果输出格式不对，会触发 H3 的输入验证失败
- 临时方案：手动复制 `show_text_party` 显示的内容，按官方 `base-en.txt` 模板微调

### H3 节点报错 "MiniMaxH3ImageToVideo expects a string prompt"

- 确认 `show_text_party` 节点有红色边（未执行）→ 等 LLM 先跑完
- 如果仍然报错，把 `show_text_party` 直接换成 `TextEncode` 节点手动粘贴英文 prompt

### 显存溢出

- 把 `BasicScheduler` 的 `steps` 调低（4 起步）
- 把 `MiniMaxH3ImageToVideo` 的 `length` 调小（73 → 56 → 39）
- 把分辨率从 `1344×768` 调低到 `864×480`（约 0.4 MP）

---

## 📚 参考资料

- **官方 skill**：`https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing`
- **官方 ComfyUI 模板**：`https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json`
- **LLM-Party 文档**：`https://github.com/heshengtao/comfyui_LLM_party`
- **H3 模型仓库**：`https://huggingface.co/Comfy-Org/MiniMax-H3`

---

## 📝 版本

- 生成日期：2026-10-06
- 适配：ComfyUI ≥ 0.30.0、ComfyUI-LLM-Party ≥ 2.0
- H3 模型版本：FL2VA / Ref2VA（开源首版）
