# Qwen 提示词扩写节点 + MiniMax H3 视频工作流

用**本地 Qwen 多模态模型**把一句简单描述（+ 可选参考图）扩写成 MiniMax H3 官方规范的三段式视频提示词，并直接接到 H3 生成链路。

节点位置：`ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer\`
节点名：`Qwen3.5 / Qwen3-VL Prompt Enhancer`（分类 `Qwen35/Prompt`）

> **改了 nodes.py 必须重启 ComfyUI** 才会生效。

---

## 一、工作流清单

| 文件 | 用途 | 节点数 | 模式 |
|------|------|--------|------|
| `prompt_enhancer_qwen35_text.json` | 只扩写提示词（纯文字） | 1 | text2video |
| `prompt_enhancer_qwen35_image.json` | 只扩写提示词（带参考图，最多 4 张） | 2 | image2video |
| `h3_i2v_qwen35.json` | **图生视频**（首帧 + 可选尾帧 → 成片） | 29 | image2video |
| `h3_ref2v_multi_qwen35.json` | **多图参考生视频**（3 张参考图 → 成片） | 30 | reference |

`h3_*` 两个是**端到端**的：模型加载 → 分辨率/时长 → 扩写 → 采样 → 视频+音频解码 → 保存，直接 Queue 就能出片。

---

## 二、两个新工作流怎么用

### 图生视频 `h3_i2v_qwen35.json`

```
LoadImage(首帧图) ─┬─→ MiniMaxH3ImageToVideo.first_frame
                   └─→ Qwen35PromptEnhancer.image
LoadImage(尾帧图) ───→ MiniMaxH3ImageToVideo.last_frame（可留空）
Qwen35PromptEnhancer.prompt ─→ MiniMaxH3ImageToVideo.prompt
```

- 首帧图**同时**喂给 Qwen 和 H3：Qwen 看着这张图把描述扩写成"从这一帧开始怎么动"。
- 模式 `image2video` 会让 Qwen **禁用** `<Picture i>` 标签（首帧不是参考图，是起点）。
- 用 `minimax_h3_fl2va_int8_convrot.safetensors` + `fl2v_turbo_4step` LoRA，默认 4 步。

### 多图参考生视频 `h3_ref2v_multi_qwen35.json`

```
LoadImage(参考图1) ─┬─→ MiniMaxH3ReferenceToVideo.ref_image_0
                    └─→ Qwen35PromptEnhancer.image
LoadImage(参考图2) ─┬─→ MiniMaxH3ReferenceToVideo.ref_image_1
                    └─→ Qwen35PromptEnhancer.image_2
LoadImage(参考图3) ─┬─→ MiniMaxH3ReferenceToVideo.ref_image_2
                    └─→ Qwen35PromptEnhancer.image_3
Qwen35PromptEnhancer.prompt ─→ MiniMaxH3ReferenceToVideo.prompt
```

- 用 `minimax_h3_ref2va_int8_convrot.safetensors` + `ref2v_turbo_8step` LoRA，默认 8 步。
- 参考图**同时**送进两个地方：H3 的参考编码器（图片潜变量参与每一步采样），以及 Qwen（让模型"看得见"你给的是什么）。

---

## 三、多图参考的关键：`<Picture i>` 编号

H3 的参考生视频不是"把图拼起来"，而是靠提示词里的 `<Picture 1>`、`<Picture 2>` 标签**点名**要用哪张图的什么特征。节点按官方规则做了两件事：

1. **编号对齐**：Qwen 收到的图顺序 = H3 参考编码器的编号顺序（`image` → `<Picture 1>`，`image_2` → `<Picture 2>` …）。
2. **强制引用**：`mode = reference` 时 system prompt 追加规则，要求输出里**必须**显式写出
   `<Picture 1> defines the lead character's face and outfit; <Picture 2> defines the rooftop set.`
   这类映射句，且禁止编号错位、禁止为没接的图编造标签。

如果输出里一句 `<Picture i>` 都没有，参考图基本不起作用 —— 所以这个约束是必须的。

---

## 四、节点参数

**required**

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_name` | 自动扫描 | 下拉，扫描 `text_encoders` / `prompt_generator` / `LLM` 下所有完整 HF 文件夹 |
| `system_prompt` | H3 skill 规则 | 已内置 MiniMax 官方 h3-prompt-writing 精简版 |
| `user_prompt` | 示例 | 你的一句话描述 |
| `quantization` | `8bit` | `none` / `8bit` / `4bit` |
| `attention` | `sdpa` | `sdpa` / `eager` |
| `enable_thinking` | `False` | 关掉思考，直接出提示词 |
| `mode` | `text2video` | **新增**，见下表 |

**mode 三档**

| 值 | 行为 |
|----|------|
| `text2video` | 不加额外规则（纯文字扩写） |
| `image2video` | 声明"附带的图是首帧"，禁止输出 `<Picture i>` |
| `reference` | 声明"附带的图是 `<Picture 1>..<Picture N>`"，要求显式引用 |

**optional**

| 参数 | 默认 | 说明 |
|------|------|------|
| `image` | 无 | 参考图 1 / 首帧 |
| `image_2` `image_3` `image_4` | 无 | **新增**，参考图 2/3/4 |
| `max_images` | `4` | **新增**，喂给模型的最大图数（1~9） |
| `keep_model_loaded` | `False` | 跑完是否留在显存 |
| `unload_other_models` | `True` | 跑之前卸掉其它模型（给 H3 让路） |
| `temperature` | `0.4` | 改写建议 0.3~0.6 |
| `max_new_tokens` | `1024` | H3 提示词够用 |
| `seed` | `42` | 固定可复现 |
| `custom_model_path` | 空 | 指向下拉里没有的目录 |

每个图片输入都可以接 **batch**（一个输出多帧），节点会按 `输入顺序 → batch 内顺序` 展开编号。

---

## 五、模型选择

| 模型 | 架构 | 大小 | 8bit 占用 | 建议 |
|------|------|------|-----------|------|
| `Qwen3-VL-8B-Instruct` | `Qwen3VLForConditionalGeneration` | 17 GB | ~9.5 GB | 新工作流默认。官方 Instruct 版，**格式遵循更稳**，多图引用不易跑偏 |
| `huihui-ai_Huihui-Qwen3.5-9B-abliterated` | `Qwen3_5ForConditionalGeneration` | 19 GB | ~10.5 GB | 已实测通过（60.5s）。abliterated 创意更放开，但容易漏首段标签，靠节点后处理兜底 |

**首次运行若报错或输出异常，把 `model_name` 切到 `huihui-ai_Huihui-Qwen3.5-9B-abliterated`** —— 那条路径已在真机验证过。

---

## 六、显存与性能（RTX 4090 D 24GB）

| 量化 | Qwen3.5-9B | Qwen3-VL-8B | 说明 |
|------|-----------|-------------|------|
| `none` | ~19 GB | ~17 GB | 和 H3 同时驻留必 OOM，不建议 |
| `8bit` | ~10 GB | ~9.5 GB | **推荐** |
| `4bit` | ~6 GB | ~5.5 GB | 最省，质量略降 |

- 已实测：Qwen3.5-9B + 8bit + sdpa，**加载+推理 60.5s**，输出 715 字符，三段式完整、无 think 残留。
- Qwen3-VL-8B 的耗时未实测（结构相同，预期略快）。
- **务必保持 `unload_other_models=True` + `keep_model_loaded=False`**：24GB 装不下 H3 + 8B 扩写模型。

---

## 七、故障排查

| 现象 | 处理 |
|------|------|
| 拖进 ComfyUI 提示 widget 数量不对 | 用的是旧 JSON。本版节点 widgets 已从 12 项变为 **14 项**，四个工作流都已同步更新 |
| 参考图槽显示不正常 | `ref_images.ref_image_N` 的 `name` 必须写成点号路径（Autogrow 格式），手改 JSON 时容易漏 |
| 输出没有 `<Picture i>` | 确认 `mode` 是 `reference`；仍不行就把模型切到 Instruct 版（abliterated 版更容易漏） |
| 下拉里没有想要的模型 | 确认目录含 `config.json`；或用 `custom_model_path` 填绝对路径 |
| 输出带 `<think>` | 关掉再开 `enable_thinking` 刷新模板 |
| 显存不足 | 量化降到 `4bit`，并确认 `unload_other_models=True` |
| 节点没变化 | **重启 ComfyUI**，看控制台 `[Qwen35]` 日志 |

---

## 八、相关的其他文件

| 文件 | 说明 |
|------|------|
| `README.md` | 最早的 LLM-Party 方案（历史） |
| `README_prompt_enhancer.md` | Qwen3_VQA 节点方案（历史） |
| `h3_t2va_with_llm.json` 等 | 上两代工作流（历史，模型名可能与本地不一致） |
| `gen_h3_video_workflows.py` | 由官方模板生成两个新工作流的脚本 |
| `verify_h3_workflows.py` | 结构校验脚本（连线双向一致 / 槽位索引 / 资源存在性） |
