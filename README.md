# ComfyUI-Qwen35-Enhancer

> 两个节点，同一套本地 Qwen 模型、同一条加载链路。
> **① 扩写**：把一句大白话扩写成符合 **MiniMax H3 官方规范**的三段式视频提示词。
> **② 批量打标**：给一个文件夹里的所有图片打标，标签写到与图片同目录同名的 `.txt`。

> Expand a one-line idea into a **MiniMax H3 spec-compliant** video prompt using a **local** Qwen multimodal model. No API calls, fully offline.

用**本地 Qwen 多模态模型**，把一句大白话扩写成符合 **MiniMax H3 官方规范**的三段式视频提示词；
也可以把整个图片文件夹批量打标（输出同名 `.txt`）。

支持 `text2video` / `image2video` / `reference`（多图参考）三种模式，模型可自由替换。
可选输出**中英双语** —— 英文版喂给 H3，中文版只作人工核对预览。

---

## 为什么需要它

MiniMax H3 对提示词格式有硬性要求：

- 必须写成**三段式** —— `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`
- 镜头切换要用 `[Shot N]` 分段
- 对白要包在 `<d></d>` 里
- 多图参考时要用 `<Picture i>` **点名**引用第几张图的什么特征

手写门槛很高。这个节点让你只写「女孩在雨里回头」，剩下的交给本地模型补全。

**全程本地推理** —— 不联网、不调 API、不产生费用。

## 特点

- **架构自动适配** —— 用 `AutoModelForImageTextToText`，同一份代码同时支持
  `Qwen3_5ForConditionalGeneration` 与 `Qwen3VLForConditionalGeneration`，换模型不用改代码
- **模型自动发现** —— 扫描 `models/text_encoders`、`models/prompt_generator`、`models/LLM`
  下所有含 `config.json` 的完整 HF 文件夹，自动过滤 Florence-2 / CLIP / T5 等非对话架构
- **多图参考** —— `image` / `image_2` / `image_3` / `image_4` 最多 4 路输入（每路可为 batch），
  按接入顺序编号，与 H3 的 `<Picture 1>`..`<Picture N>` 严格对齐
- **思考块自动处理** —— 先嗅探 processor 的 chat template 是否真的认 `enable_thinking`
  （Qwen3.5 认、Qwen3-VL 不认），认才传，避免无效参数与警告；无论哪种情况都有正则兜底剥离
- **输出规范化** —— 模型常漏掉首段标签，节点自动补齐三段结构
- **中英双语预览** —— `bilingual=en_then_zh` 时先出英文 H3 提示词，再由同一个模型翻一份
  中文到 `prompt_zh` 端口，供人工核对扩写是否符合预期。中文版保留英文结构标签、
  `[Shot N]`、`<Picture i>`、`(S1)` 等标记，可与英文版逐段对照。
  翻译复用已加载的模型（纯文本、无图），**实测只多约 20%** ——
  忠实中文译文只要英文的 1.07~1.12 倍 token（见「双语的真实开销」）
- **显存可控** —— 支持 8bit / 4bit 量化，跑完自动释放，方便与 H3 交替调度。
  但**默认 `none`**：量化是省显存的手段，本机真机 A/B 实测（9B，同工作流）
  `none` 51.85s / **`4bit` 54.06s（只慢 4%，显存 7.9GiB）** / `8bit` 71.01s。
  **要省显存就选 `4bit`** —— `8bit` 多花的 17 秒里 84% 是加载，纯亏。
- **进度可见** —— 生成阶段逐 token 上报进度：ComfyUI 前端在该节点上直接画出进度条，
  同时控制台按间隔输出进度行（真实终端下额外有一条原地刷新的实时条）。
  并接上了 ComfyUI 的中断标志，前端点「取消」能真正打断长生成。
- **批量打标**（第二个节点 `Qwen35BatchImageTagger`）—— 填一个文件夹路径，给里面每张图
  生成**与图片同目录同名**的 `.txt`，系统提示词可改。内置 **6 套预设**（3 主题 × 中英），
  输出「只写内容、不写风格」的现代自然语言描述，NSFW 如实描述（敏感内容直接、具体地写）；
  可选**中英双语同时输出**（`图名.txt` + `图名_zh.txt`）、**输出字符数上限**，以及
  **优化已有打标**（拿图片旁边已有的 txt 当初稿让模型校订补全，按「内容变没变」决定
  要不要重做 —— 手改过就继续优化，没动过就跳过）。**一次加载打完整个文件夹**（走扩写节点
  循环 N 次的话每张都要重载）；默认 `overwrite=skip` **绝不覆盖已有标签**；单张失败不中断
  整批，但「取消」会立刻中止并保留已完成的部分。详见「批量打标节点」。

## 安装

**ComfyUI-Manager**（上架后）：搜索 `Qwen H3 Prompt Enhancer`。

**手动安装**：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zhaolu8294/ComfyUI-Qwen35-Enhancer
cd ComfyUI-Qwen35-Enhancer
pip install -r requirements.txt
```

重启 ComfyUI。两个节点分别出现在 `Qwen35/Prompt`（扩写）与 `Qwen35/Batch`（批量打标）分类下。

> 依赖 `transformers >= 5.0`（原生支持 `qwen3_5` 架构）。Qwen3.5 的线性注意力走
> transformers 内置的纯 torch 回退，**不需要** `flash-linear-attention` / `causal-conv1d`。

### 图片 prefill 慢的真凶：视觉塔入口的 Conv3d（已自动修掉）

**Qwen3-VL 和 Qwen3.5 两个族的 `patch_embed` 都是同一个写法**：
`Conv3d(3 → 1152, kernel=stride=(2,16,16))`，输出空间尺寸恒为 `1×1×1`
—— 数学上就等于一个 `(N,1536) @ (1536,1152)` 的 GEMM，理论约 12 GFLOP。
但在 Windows + `torch 2.9.1+cu130` + cuDNN 9.12 上实测：

| 算子 | 首次调用 | 稳态 |
|---|---|---|
| `Conv3d` **fp32** | 0.05 s | **4.4 ms** |
| `Conv3d` **fp16** | **> 60 s 未返回** | — |
| `Conv3d` **bf16** | **> 60 s 未返回** | — |
| 等价 `GEMM`（bf16） | — | **0.43 ms** |

视觉塔整段 **29 秒里约 96% 卡在这个卷积上**。

**节点加载模型后会自动把它的 forward 换成 `F.linear`**：权重只做 reshape 视图、
不复制，`state_dict` 键名不变，实测与原 Conv3d 的最大绝对误差 **2e-06**。
日志里那条 `视觉塔入口：patch_embed 走 …` 就是它的状态。
临时关闭：设环境变量 `QWEN35_NO_PATCH_EMBED_FIX=1`。

> **补丁按「结构」识别，不按「类名」**。两个族的类名不同：
> Qwen3-VL 是 `Qwen3VLVisionPatchEmbed`，Qwen3.5 是 `Qwen3_5VisionPatchEmbed`，
> 类与所在模块都不一样。早期版本只按类名补 Qwen3-VL 那个类，
> 一换 Qwen3.5 补丁就静默失效 —— 日志照报"已换成 GEMM"，
> 而 prefill 从 **0.69 s 退回 75.61 s**（35.1 ms/视觉token）。
> 现在改成遍历加载好的模型，凡类名含 `patchembed` 且 `.proj` 是
> `kernel==stride`、`padding=0`、`groups=1` 的 `Conv3d` 就替换，
> 换模型族不会再漏。

> **别被"装 flash-attn 提速"误导**：同一份权重、同一份输入下，视觉塔跑
> eager / sdpa / flash_attention_2 的耗时是 **29.16 / 28.91 / 29.30 秒 —— 毫无差别**。
> 瓶颈根本不在注意力，装 flash-attn 不会让 prefill 变快。

### 可选：装 flash-attn（省显存、LLM 长上下文更快）

装不装都不影响图片 prefill，但它对显存和 LLM 侧长上下文有益，想装可以这样装
（Windows 没有官方 wheel，别直接 `pip install flash-attn` 去源码编译）：

找与本机 `torch` CUDA 大版本、Python 版本对应的社区预编译 wheel，例如
`flash_attn-2.8.3+cu130torch2.9-cp313-cp313-win_amd64.whl`
（cu130 = torch 的 CUDA 13.0，cp313 = Python 3.13），然后

```bash
pip install /path/to/flash_attn-2.8.3+cu130torch2.9-cp313-cp313-win_amd64.whl
```

装好后把 `attention` 设成 `auto` 即可 —— auto 检测到合规的 flash-attn
（>= 2.3.3）就自动用 `flash_attention_2`，没装则退回 `sdpa`，所以卸载也不会跑挂。
切后端时 `PretrainedConfig` 的 setter 会把值**递归**写到 `vision_config`，
视觉塔因此自动跟着切，不需要手工传播。

### 解码慢的一半：线性注意力走了 torch 回退（建议装 fla）

**Qwen3.5 不是普通 Transformer。** 它的 32 层里 **24 层是 Gated DeltaNet
线性注意力**，只有 8 层是全注意力（`full_attention_interval=4`）。

transformers 为这些层准备了两套实现，每个层实例上挂哪个取决于可选依赖：

```python
self.chunk_gated_delta_rule     = chunk_gated_delta_rule     or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_...
```

前者来自 `fla`（flash-linear-attention）。**没装 fla 时 decode 会落到 torch 回退**，
而那份实现是：

```python
[x.transpose(1, 2).contiguous().to(torch.float32) for x in (...)]   # fp32
for i in range(sequence_length): ...                               # 逐 token
```

即 **fp32 + 逐个 token + 一串小张量算子**。本机 4090D 实测（同权重同形状，seq=1）：

| | 单层 | ×24 层 | 占解码 |
|---|---|---|---|
| torch 回退 | 1.009 ms | **24.21 ms/token** | **50.6%** |
| fla 融合内核 | 0.659 ms | 15.81 ms/token | 33.0% |

单看 gated delta rule 本身（裸算子）是 0.321 ms → 0.056 ms，**5.8×**。
它的等效带宽只有峰值 1008 GB/s 的 **1.2%** —— 完全是「小算子太多」的延迟问题，
**不是算力问题、不是带宽问题、更不是图太大**。所以调 `max_image_side`
或换量化都救不了它。

装法（**纯 Python wheel，不需要编译**，装完重启 ComfyUI 即生效）：

```bash
pip install flash-linear-attention
```

约省 1/3 解码时间（整轮 51.85 s → 约 47 s）。数值上与原实现相对误差约
**6e-03**（bf16 正常量级），24 步连续 decode 无漂移。不想要了
`pip uninstall flash-linear-attention fla-core` 即可退回。

> 注意 `causal-conv1d`（同一快路径需要的另一个库）**没有 Windows 预编译 wheel**，
> 只能从源码编。不装它也没关系：那个 `is_fast_path_available` 标志只控制
> transformers 自己那条 warning，**不决定用哪份实现** —— 卷积那一小段仍走 torch
> 回退，代价可忽略（kernel 只有 4）。

节点每次加载模型后会主动报一行，让你不必再靠猜：

```
[Qwen35] 线性注意力：24 层里 24 层走 torch 回退 —— 这些层在解码时逐 token 跑
         fp32 小算子，实测 1.0ms/层（融合内核 0.66ms/层），24 层即占解码约一半。
```

装了 fla 则变成 `24 层全部走融合内核（fla 已生效）`。

## 准备模型

### 1. 扩写用的 Qwen 模型（二选一）

放到 `ComfyUI/models/text_encoders/` 或 `models/prompt_generator/` 下，
**必须是含 `config.json` + tokenizer 的完整 HF 文件夹**（不能是单个 safetensors）。

| 模型 | 大小 | 说明 |
|------|------|------|
| `Qwen3-VL-8B-Instruct` | ~17GB | **推荐**。指令版，对格式约束遵循最稳 |
| `Qwen3.5-9B`（任意微调版） | ~19GB | 架构更新，长上下文更省；去审查版创意更放开，但容易漏首段标签 |

> 节点只会扫描带 `config.json` 的文件夹。`models/text_encoders/` 下那些
> 单个 `*.safetensors`（给扩散模型当 text encoder 用的）**不会**出现在下拉列表里，这是正常的。

### 2. MiniMax H3 视频模型

| 目录 | 文件 |
|------|------|
| `models/diffusion_models/minimax_h3/` | `minimax_h3_fl2va_int8_convrot.safetensors`（图生视频）<br>`minimax_h3_ref2va_int8_convrot.safetensors`（参考生视频） |
| `models/text_encoders/minimax_h3/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `models/vae/minimax_h3/` | `minimax_h3_video_vae_fp16.safetensors`<br>`minimax_h3_audio_vae_fp32.safetensors` |
| `models/loras/H3/` | `minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors`<br>`minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |

模型下载见 [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)。

## 跑得动 27B：GGUF 后端（llama.cpp）

上面的 HF 路径对 8B / 9B 够用，但对 **Qwen3.8-27B** 这类大模型直接废掉：
27.78B dense、BF16 权重 **55.56GB**，24GB 卡装不下。而要命的是
`device_map="auto"` **不会报错** —— 它只会安静地把一部分层摊到 CPU，
速度从 40~80 tok/s 掉到 **0.6~3.4 tok/s**（掉一两个数量级），日志里一个字都不提。

把 55GB 压到能装下的唯一现实做法是量化。几条路的对比：

| 路线 | 需要下载 | 驻留显存 | 结论 |
|------|---------|---------|------|
| bnb 4bit（HF 路径） | 55.56GB 原版权重 | ~19GB | 能跑，但为了一份量化结果先下 55GB |
| FP8 | 30.9GB | >24GB | 装不下 |
| AWQ / GPTQ | — | — | 本机未装 autoawq 等工具链 |
| **GGUF Q4_K_M** | **16.2GB + 0.9GB** | **~19GB** | **✅ 下载量最小、开箱即用** |

所以这里走 **GGUF + llama-server**：模型以量化形式落盘，显存里放的就是量化权重，
不存在「下 55GB 再压成 16GB」这种浪费。

### 1. 装 llama.cpp（Windows CUDA 预编译包，解压即用）

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下这两个
**同 build** 的包，解压到**同一个目录**（例如 `H:\AI\llama.cpp\`）：

| 文件 | 作用 |
|------|------|
| `llama-<build>-bin-win-cuda-13.4-x64.zip` | `llama-server.exe` 及 ggml/llama 的 dll |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` | CUDA 运行时（`cublas64_13.dll` 等）。**缺了会启动失败** |

> CUDA 版本按你的驱动选：`cuda-12.4` / `cuda-13.4` 都有对应包。
> 解压后子目录层级无所谓，节点会在三层深度内自动找到 `llama-server.exe`。

### 2. 下模型（主干 + 视觉投影，缺一不可）

以 Qwen3.8-27B 的去审查版为例，仓库
[`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`](https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)
（国内镜像 `hf-mirror.com`，前缀统一为
`https://hf-mirror.com/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/`
+ 文件名 + `?download=true`）：

| 文件 | 大小 | 必需 |
|------|------|------|
| `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` | 16.19GB | 主干。备选档位：`Q4_K` 15.66GB / `UD-IQ4_XS` 13.41GB / `Q5_K_XL` 18.19GB |
| `mmproj-model-bf16.gguf` | 0.89GB | **打标必须**。这是视觉塔，不下它 llama-server 会加载成纯文本模型，图片被直接丢掉 |

存放位置（任选，节点都会扫）：

- `ComfyUI/models/LLM/`
- `H:\AI\models\LLM\`（`H:\AI\llama.cpp\` 的上一级 + `models\LLM`）
- `H:\AI\LLM\`
- 或设环境变量 `QWEN35_GGUF_DIR` 指向任意目录（多个用 `;` 分隔）

> **mmproj 不用手动下 —— 节点会自己取。**
> 下一节的 `mmproj_url` 默认已填好上面那个地址，节点发现本地没有就会自己去下
> （先写 `.part` 再原子改名，断了再跑一次会**断点续传**）。
>
> **联网只发生在「文件确实不在本地」的那一刻。** 文件下好之后，
> 节点靠本地扫描（`find_existing()`）直接命中，**一个网络请求都不会发**，
> 之后完全是离线的 —— llama-server 是本地进程，GGUF 也在本地磁盘上。
> 想彻底禁掉联网：把 `mmproj_url` 清空，或把 `auto_download` 关掉。

### 3. 接线

新节点 **`GGUF Backend (llama.cpp) - Qwen3.8 / 27B`**（分类 `Qwen35/Backend`）：

| 控件 | 默认 | 说明 |
|------|------|------|
| `model` | 自动扫描 | 主干 `.gguf` |
| `mmproj` | `（无 / 纯文本）` | **打标必须选** |
| `server_exe` | `auto` | 自动找；也可手填路径 |
| `context_size` | 8192 | 打标用不了多长。**吃显存的是 KV 不是权重** |
| `kv_cache_type` | `q8_0` | `q8_0` 是速度/显存的好平衡 |
| `n_gpu_layers` | **-1** | -1 = 不传 `-ngl`，交给 llama.cpp 自动 fit。**填具体数字会关掉自动 fit**，估错就静默退回 CPU |
| `parallel` | **1** | **别调大**。Qwen3.8 有 48 层线性注意力（Gated DeltaNet），每层按**序列槽**存 recurrent state；llama-server 默认值是 4，槽位一多白吃约 0.44GiB |
| `flash_attn` | true | 更省 KV 也更快 |
| `reasoning_format` | `deepseek` | 把思考块引到独立的 `reasoning_content`，`content` 里只剩最终答案 |
| `extra_args` | 空 | 原样追加，例如 `--image-max-tokens 1024` 限制视觉 token 数 |
| `auto_download` | `true` | 上面选的文件本地找不到时，按下面的地址自动下。断了再跑一次会断点续传 |
| `model_url` | 空 | 主干 gguf 的下载地址。**默认留空** —— 16GB 级别的东西，建议确认好地址再填 |
| `mmproj_url` | 已填好 | 视觉投影的地址（Qwen3.8-27B 那个，约 888MiB）。清空 = 不下 |

`auto_download` 的判据是**「文件真的不在本地」**，而不是「下拉里没选」：
你早就手动下好的话，节点会在所有搜索路径里找到同名文件直接用，不会白下一遍。

实测下载速度（hf-mirror，本机直连）：mmproj 888MiB 用 78s，11.6 MB/s。

把这个节点的 `backend` 输出，连到**扩写节点**或**批量打标节点**的 `backend` 输入即可。

> `backend` 是**可选输入线**：不连的节点完全走原来的 transformers 路径，
> 行为一个字节都没变 —— 旧工作流不用改。

### 4. 实测

4090D 24GB + Qwen3.8-27B `UD-Q4_K_XL`：

```
[显存] 加载后已用 23638 MiB / 24564 MiB
[耗时] 加载 109.2s
  回复     : 'The capital of France is Paris.'
  prompt   : 36 tok / 0.27s
  生成     : 8 tok / 0.18s  =  45.1 tok/s
```

**45.1 tok/s** 是满血 GPU 的量级。如果打标日志里掉到个位数 tok/s，
先看有没有 `offloaded ... to CPU` —— 那说明显存不够、层被摊到 CPU 了。

### 5. 显存要腾出来

llama-server 是**独立进程**，和 ComfyUI 抢同一块显存。第一次推理前节点会先
`unload_all_models()` + `free_memory()` 把 ComfyUI 自己常驻的那份放掉
（打标场景本来也不需要 SD 模型）—— 不做这步，llama-server 只能捡零头，
然后就是上面说的「不报错、悄悄变慢」。

### 什么时候别用这条路

- **ComfyUI-GGUF 插件不适用**：那是给扩散模型写的，它把 GGUF **反量化回浮点**再喂 torch。
  对 9B 无所谓，对 27B 等于绕一圈回到 55GB。
- **`llama-cpp-python` 也不走**：官方只发 CPU-only 的 Windows wheel，
  CUDA 版要自己编译（Python 3.13 + 新版 llama.cpp，坑多，且升级二进制要重编整个包）。
- **`transformers` 的 GGUF 支持同理**，也是反量化路线。

走外部进程还有一个附带好处：llama-server 崩了拖不垮 ComfyUI，
也可以单独开它的网页界面调参。

## 节点参数

分类：`Qwen35/Prompt` ｜ 输入：文本（+ 可选图） ｜ 输出：`STRING`（符合 H3 规范的提示词）

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_name` | 自动扫描 | 扩写用的 Qwen 模型，下拉选择 |
| `system_prompt` | H3 规范 | 系统提示词，一般不用改 |
| `user_prompt` | 示例文本 | **你的简单描述**，一句话即可 |
| `quantization` | **`none`** | `none` / `8bit` / `4bit`。**默认 none** —— 显存够就用它。要省显存选 **`4bit`**（真机 A/B 整轮 54.06s vs none 51.85s、驻留 7.9GiB vs 17.5GiB），**别选 8bit**（整轮 71.01s，多花的 17s 里 14.3s 在加载；即使已自动把 `llm_int8_threshold` 调到 0 也追不上） |
| `attention` | `auto` | `auto` / `flash_attention_2` / `sdpa` / `eager`。auto = 装了 flash-attn 就用 FA2，否则退回 sdpa。**实测图片 prefill 与它无关**（三种后端 29.16 / 28.91 / 29.30 s），真正起作用的是自动的 patch_embed→GEMM 替换 |
| `enable_thinking` | `False` | 是否让模型先输出思考过程 |
| `mode` | `text2video` | 见下表 |
| `image` ~ `image_4` | 空 | 可选图像输入，最多 4 路 |
| `max_images` | `4` | 喂给模型的上限（1~9） |
| `keep_model_loaded` | `False` | 跑完是否常驻显存 |
| `unload_other_models` | `True` | 加载前是否卸载其他模型 |
| `temperature` | `0.4` | 采样温度 |
| `max_new_tokens` | `1024` | 输出长度上限 |
| `seed` | `42` | 随机种子 |
| `custom_model_path` | 空 | 兜底：手填模型文件夹绝对路径 |
| `max_image_side` | `1536` | 参考图长边上限，超出则等比缩小（`0` = 不限制）。1536 比 1280 只多约 44% 的图片 token，小物件却认得更准 |
| `show_progress` | `True` | 是否上报进度（前端进度条 + 控制台进度行） |
| `progress_interval` | `2.0` | 控制台进度行的最小间隔（秒）；终端实时条不受它限制 |
| `bilingual` | `en_then_zh` | `en_then_zh` = 额外输出一份中文预览（实测多约 20% 耗时）；`off` = 只出英文 |

### mode 的三种取值

| mode | 注入的规则 | 配合的 H3 节点 |
|------|-----------|---------------|
| `text2video` | 不追加额外规则 | — |
| `image2video` | 声明附带的图是**首帧**，**禁止**输出 `<Picture i>` | `MiniMaxH3ImageToVideo` |
| `reference` | 声明附带的图是 `<Picture 1>`..`<Picture N>`，**要求显式引用** | `MiniMaxH3ReferenceToVideo` |

> `reference` 模式下，图片顺序与 H3 参考编码器的编号必须一致，否则参考图等于白给。
> 节点已按接入顺序自动对齐（`image`→`<Picture 1>`，`image_2`→`<Picture 2>`…）。

## 示例工作流

`examples/` 目录下的 JSON 直接拖进 ComfyUI 即可运行，**不是只有扩写节点，而是从模型加载到出片的完整链路**。

| 文件 | 用途 |
|------|------|
| `h3_i2v_qwen35.json` | **图生视频** —— 首帧图 + 简单描述 → 视频 |
| `h3_ref2v_multi_qwen35.json` | **多图参考生视频** —— 4 张参考图 + 描述 → 视频 |
| `prompt_enhancer_qwen35_text.json` | 仅扩写（纯文本），输出接到你自己的链路 |
| `prompt_enhancer_qwen35_image.json` | 仅扩写（带图） |
| `batch_tagger_qwen35.json` | **批量打标** —— 填个文件夹路径，给里面每张图生成同名 `.txt`；也可用 `caption_mode` 优化已有的那些 |

> 示例工作流基于 ComfyUI 官方 MiniMax H3 模板改造，模型文件名为 Comfy-Org 官方发布的版本。
> 若你的文件名不同，在每个 Loader 节点里重新选一次即可。

> 每个示例都在 Qwen 节点右侧接了一个 `PreviewAny`（Preview as Text）节点来显示中文预览。
> 喂给 H3 的一直是**第 1 个端口 `prompt`（英文）**。

## 输出格式

```
integrated_multimodal_description: [Shot 1] <风格/画质>, <构图>, <主体与动作>, ...

overall_soundscape: ...

non_diegetic_music: ...
```

### 两个输出端口

| 端口 | 内容 | 用途 |
|------|------|------|
| `prompt` | 英文 H3 提示词 | **喂给 H3**。第 1 个输出，位置不变，旧连线不受影响 |
| `prompt_zh` | 中文预览 | 只给人看，用来核对扩写内容是否符合预期 |

`bilingual=off` 时 `prompt_zh` 返回空串。

中文预览与英文结构对齐，方便逐段对照：

| 保留原样 | 翻译成中文 |
|---------|-----------|
| `integrated_multimodal_description:` 等三段标签 | 标签后面的描述性内容 |
| `[Shot N]`、`At 00:02.500` | 镜头内容、动作、光线、构图 |
| `<Picture i>`、`(S1)` 等编号 | 运镜术语（推镜/拉镜/摇镜/跟镜/环绕 + 幅度/速度） |
| `<d>...</d>` 内的对白原文 | 在 `</d>` 后附一句中文释义 |

> 中文版**不参与** H3 推理解析，格式不必严格 —— 保留英文标记纯粹是为了让你在改英文版时
> 能快速定位到对应段落。翻译用**贪心解码**（`do_sample=False`）：翻译是确定性任务，
> 采样只会引入随机性并拖后 EOS。

### 双语的真实开销

第二阶段的解码量就是全部的额外成本（prefill 只有几百 token，可忽略）。用本机
Qwen3-VL 分词器对三组真实 H3 提示词实测（`h3_workflow/measure_zh_token_ratio.py`）：

| 场景 | 英文字符 | 英文 tok | 中文字符 | 中文 tok | token 比 |
|------|---------|---------|---------|---------|---------|
| 4s / 单镜头 | 808 | 185 | 310 | 207 | **1.12x** |
| 8s / 双镜头 | 839 | 204 | 334 | 229 | **1.12x** |
| 12s / 三镜头 + 对白 | 1013 | 254 | 443 | 273 | **1.07x** |

中文字符数只有英文的 0.41 倍，但每个中文 token 只装 1.53 个字，英文能装 4.14 个字符，
两者大致抵消。**所以双语正常只该多花约 20%。**

如果实测多花了 2~3 倍，几乎一定是第二阶段在**「先把英文原文抄一遍再翻译」**——
那样第二段要解码约 2 倍 token。节点对此有三道防线：

1. `TRANSLATE_SYSTEM_PROMPT` 第 0 条明令禁止重复、引用、复述英文原文；
2. 代码里 `_drop_english_echo()` 兜底：第一个中文字符之前若出现三段标签，判定为回声并砍掉；
3. 中文段预算按英文实际长度换算（`英文 tok × 1.6 + 64`），触顶即停，不会跑满 `max_new_tokens`。

运行日志会打印两段 token 比，正常应贴近 1.1；超过 1.4 会自动告警。

## 批量打标节点（Qwen35BatchImageTagger）

给一个文件夹，把里面每张图喂给同一个 Qwen 模型打标，标签写到**与图片同目录、同文件名**
的 `.txt`：

```
E:\datasets\mydata\
├── 001.png
├── 001.txt      ← 生成
├── 002.jpg
└── 002.txt      ← 生成
```

### 最小用法

1. 把 `examples/batch_tagger_qwen35.json` 拖进 ComfyUI（或右键添加
   `Qwen3.5 Batch Image Tagger (txt)`，分类在 `Qwen35/Batch`）。
2. 填 **`folder_path`**。绝对路径即可；也可写 `input/xxx` 或 `output/xxx`（会解析到
   ComfyUI 的对应目录），从资源管理器复制来的带引号路径也认。
3. 按训练目标选 **`system_preset`** —— 三套主题（写实照片 / 二次元角色 / 场景环境）
   × 中英两种输出语言 = **6 套**。要英文描述就选不带后缀的（`photoreal`），要中文
   描述就选带 `_zh` 的（`photoreal_zh`）。想自己写提示词就保持 `custom`（默认）。
4. 想先确认清单，就把 `dry_run` 打开跑一次 —— 它只列出「哪些图会被处理、txt 写到哪」，
   **不加载模型、不写任何文件**。确认无误后关掉 `dry_run` 再跑。
5. 想**改造已有的一批 caption**（而不是从零打标）？把 `caption_mode` 设成 `refine`
   就行 —— 它会拿图片旁边已有的 `.txt` 当初稿让模型校订补全，并且按「内容变没变」
   决定要不要重做。详见「优化已有打标」。

### 写入规则

| 规则 | 说明 |
|---|---|
| 文件名 | `图片名.txt`（`output_suffix` 可改成 `图片名_tags.txt`） |
| 双语命名 | `bilingual=en_then_zh` 时英文仍写 `图名.txt`、中文写 `图名_zh.txt`。`output_suffix` 排在前面：`_tags` → `图名_tags.txt` / `图名_tags_zh.txt` |
| 位置 | **图片所在目录**，不新建标签文件夹 |
| 编码 | 默认 `utf-8`，可选 `utf-8-sig` / `gbk` / `utf-16` |
| 已有标签 | 默认 `overwrite=skip`：**绝不覆盖**。零字节的旧 txt 视为上次写失败的残留，会重新打标 |
| 双语下跳过 | **按语言各自判断** —— 英文已有、中文还缺时只补中文那份，不重跑已写好的英文 |
| 优化已有打标 | `caption_mode=refine` 时反向判定：**按内容指纹决定要不要动它**。旁边会留下 `图名.txt.orig`（第一次优化前的原稿，只留一份）与 `图名.txt.q35state`（输出指纹记录，删掉无副作用） |
| 原子写 | 先写 `.qwen35tmp` 再原子替换 —— 中途取消或断电不会留下半个文件被下游当成有效标签 |
| 空输出 | 视为该张失败，**不写空 txt**（否则下游会以为这张已打好标） |
| 画面处理 | 自动 EXIF 转正（手机竖拍图不会躺着喂给模型）、统一 RGB（灰度 / 带 alpha / CMYK 都能读）、按 `max_image_side` 等比缩小 |

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `folder_path` | 空 | 图片文件夹。**必填** |
| `system_preset` | `custom` | **6 套**内置预设（3 主题 × 中英，见下节）。选非 `custom` 时**忽略** `system_prompt` 与 `output_format` |
| `system_prompt` | danbooru 风格 | 打标规则，可整个换掉（仅 `system_preset=custom` 时生效） |
| `user_prompt` | `给这张图打标。` | 跟图片一起发出去的那句话 |
| `quantization` / `attention` / `enable_thinking` | `none` / `auto` / `False` | 与扩写节点同义 |
| `recursive` | `False` | 是否含子目录 |
| `overwrite` | `skip` | `skip` / `overwrite` |
| `image_exts` | `.png,.jpg,.jpeg,.webp,.bmp` | 认哪些扩展名；写 `png` 或 `.png` 都行 |
| `output_suffix` / `output_encoding` | 空 / `utf-8` | 文件名后缀、文本编码 |
| `output_format` | `tags_one_line` | **仅在 `system_preset=custom` 时生效**。`tags_one_line` = 规整成一行逗号标签；`raw` = 原样写模型输出。6 套内置预设各自带 `raw`，所以选预设时不用手动改这里 |
| `max_new_tokens` | `256` | 标签很短，给 1024 只会让偶尔跑飞的那几张白等到底 |
| `temperature` | `0.2` | 打标要稳。设 `0` 走贪心解码，同一批图两次跑结果完全一致 |
| `seed` | `42` | 每张图的种子是 `seed + 序号`，各图不共享采样轨迹 |
| `max_image_side` | `1536` | 长边上限。视觉 token 数 = `(w/32)×(h/32)`，1280→1536 只多约 44%，小物件与细节却认得更准 |
| `limit` | `0` | 只处理前 N 张（0 = 不限）。先小批量试跑很方便 |
| `dry_run` | `False` | 只列清单，不加载模型、不写文件 |
| `keep_model_loaded` / `unload_other_models` | `False` / `True` | 与扩写节点同义 |
| `bilingual` | `off` | `en_then_zh` = 每张跑两次：英文写 `图名.txt`、中文写 `图名_zh.txt`。默认 `off`，不加倍耗时 |
| `max_output_chars` | `0` | 输出字符数上限（0 = 不限）。超出时优先退到句末标点，避免切在半句话里 |
| `caption_mode` | `off` | **优化已有打标**。`off` = 从零打标；`refine` = 只优化已有 txt；`refine_or_new` = 有 txt 就优化、没有就从零补写。详见「优化已有打标」 |
| `desc_length` | `preset` | **描述长度档位**。`preset` = 预设原文一个字不动；`short` / `medium` / `long` / `extra_long` / `custom` = 覆盖提示词里那句长度条款。详见「描述长度档位」 |
| `desc_words` | 空 | 仅 `desc_length=custom` 时生效。填 `60~150` 这样的字数区间（英文按 words、中文按字计） |
| `bilingual_sync` | `auto` | **双语互同步**（需 `bilingual=en_then_zh` + `caption_mode != off`）。`auto` = 哪一份被手改过就以它为源改写另一份；`zh_to_en` = 中文被改过时无条件以中文为准；`off` = 两份各判各的。详见「双语互同步」 |
| `verify` | `off` | **一轮带图自检**。`once` = 生成后把图与描述一起送回，逐条核验后只改判错的部分。实测能纠正细节类错误，但耗时约 **×5**（自检要连逐条核验清单一起输出）。详见「一轮带图自检」 |

> 最后七个控件是**追加在最末尾**的。旧工作流（`widgets_values` 少这几项）载入时会自动用
> 默认值补齐，不需要手工改 JSON。

### 六套训练场景预设（`system_preset`）

不同训练场景要描述的内容差别很大 —— 真人写实看外貌衣着、二次元角色看设定与动作、
场景概念图看地理与建筑。所以内置 **3 主题 × 中英 2 语言 = 6 套**，用 `system_preset`
下拉一键切换；选 `custom` 才用下面那个可编辑的 `system_prompt`。

| 预设 | 主题 | 输出语言 |
|---|---|---|
| `photoreal` / `photoreal_zh` | 写实照片、人像写真 | 英文 / 中文 |
| `character` / `character_zh` | 二次元角色、画师插画 | 英文 / 中文 |
| `scene` / `scene_zh` | 场景、环境、概念图 | 英文 / 中文 |
| `custom` | 自定义（用节点上的 `system_prompt`） | — |

三套主题共用同一套规则，**这三条就是本轮改造的重点**：

1. **只写内容，不写风格。** 明确禁掉风格 / 技法 / 质量 / 构图 / 打光类词 ——
   `anime`、`photorealistic`、`cinematic`、`watercolour`、`matte painting`、
   `rule of thirds`、`rim light`、`masterpiece`、`8k` 等一律不出现。数据集里这些词
   互相冲突，而且下游模型本来就会自己学风格，写进去只添噪声。
2. **一律现代自然语言，不是逗号标签堆。** 输出是 1~3 句连贯短描述（约 30~80 字），
   大致按「主体 → 外观 → 衣着 → 动作 → 环境 → 光照/时间」组织。
   这个长度写在每套预设的提示词里，**模型严格照办** —— 想让它更长或更短，
   用节点上的 `desc_length` 档位改（见「描述长度档位」），不用手改 JSON。
3. **NSFW 内容不排除、如实描述。** 不因尺度回避、不含糊其辞，而且**直接、具体地写** ——
   是什么部位、什么行为、什么状态就写什么，不委婉、不软化、不换成含糊的替代说法，
   也不绕着说、暗示而不点明。

各套的完整提示词见下面的 JSON，可自己改。

#### 中英双语同时输出（`bilingual`）

`system_preset` 决定**主题**，`bilingual` 决定**要不要两种语言都出**：

| `bilingual` | 行为 |
|---|---|
| `off`（默认） | 只按预设那一种语言写一份 txt，零额外耗时 |
| `en_then_zh` | 每张图跑两次：英文写 `图名.txt`、中文写 `图名_zh.txt` |

- 主题由预设的 `group` 决定 —— 选 `photoreal` 还是 `photoreal_zh`，双语结果完全一样。
- 中英两份提示词由**同一主题**自动配对，切语言只是换一段系统提示词。
- `custom` 只有你手写的那一份，推不出另一种语言，所以双语会**退化回单份**并在报告里说明原因。
- 跳过判断**按语言各自进行**：英文已有、中文还缺时只补中文，不覆盖已写好的英文。
- 耗时约翻倍（每张两次前向，串行跑），显存不变。

#### 输出字符数上限（`max_output_chars`）

自然语言描述偶尔会啰嗦，而下游训练常有 caption 长度预算。`max_output_chars`
给一个**按字符数**的硬上限（默认 `0` = 不限）：

- **为什么不用 `max_new_tokens`**：中英文的 token / 字符比差很多，同一个 token 上限
  对中文太松、对英文太紧，控不住实际长度。
- 超限时**优先退到句末标点**（`。！？.`）并保留标点，不切在半句话里；退得太狠
  （不足上限 40%）再退到分句标点（`，、；`）并丢掉悬空标点；都不行才硬切。
- 被截断的条数会在报告里单列一行（`按上限截断: N 条`）。
- 注意它和 `desc_length` 是**两回事**：这个只往下压（超了才切），档位是往上抬。
  两个一起用、且上限比档位下限还小时，报告里会给出冲突提醒。

#### 描述长度档位（`desc_length`）

六套内置预设的系统提示词里都写着同一句长度条款：

```
- Plain prose. 1~3 sentences, roughly 30~80 words.      （英文预设）
- 白话散文。1~3 句，30~80 字。                            （中文预设）
```

模型是严格照办的 —— 所以**输出短是提示词规定的，不是 `max_new_tokens` 卡住的**
（那个默认 256，从来没被顶到过）。

`desc_length` 在送进模型之前把那句话里的数字换掉：

| 档位 | 结果 | 英文预设注入 | 中文预设注入 |
|---|---|---|---|
| `preset`（默认） | 一个字都不动 | 1~3 sentences / 30~80 words | 1~3 句 / 30~80 字 |
| `short` | 比默认还短 | 1~2 sentences / 15~40 words | 1~2 句 / 15~40 字 |
| `medium` | 略长 | 2~4 sentences / 50~100 words | 2~4 句 / 50~100 字 |
| `long` | 明显更长 | 4~8 sentences / 120~250 words | 4~8 句 / 120~250 字 |
| `extra_long` | 很长 | 8~15 sentences / 250~450 words | 8~15 句 / 250~450 字 |
| `custom` | 自己定 | 用 `desc_words` 填的区间 | 同 |

三个刻意的设计：

- **只动数字，不动句式。** 替换保留原有的连接词、标点与语种 —— 英文预设还是
  `4~8 sentences, roughly 120~250 words`，中文预设还是 `4~8 句，120~250 字`，
  不会出现中文提示词里冒出 `words` 这种事。
- **`refine` 指令里的同一句话一起改。** 「优化已有打标」用的 `refine_prompt` 里也有
  `Make it read as 1~3 plain sentences, about 30~80 words`；不一起改的话，
  长档位会在这道工序里被压回短的。
- **`max_new_tokens` 自动跟着抬。** `long` 档英文要 464 token，默认的 256 装不下，
  不抬就一定被硬截断。节点按「取两者较大值」处理，并在报告里写明抬到了多少。

`desc_length=custom` 时在 `desc_words` 里填字数区间，四种写法都认：

```
60~150      60-150      60 到 150      60,150
```

句数由字数自动反推（每句按 30~45 字算），所以不会出现「1 句写 250 字」这种挤成一坨的区间。
填的东西解析不出来（比如只写了个 `80`）会**退回 `preset` 并在报告里说明** ——
不会猜一个数字硬套上去。

> 如果你用 `system_preset=custom` 且自己写的提示词里**没有**长度条款，档位不会静默失效：
> 节点会在末尾追加一条覆盖指令，并在报告里注明这次是「追加」而不是「改写」。
> （默认那份 danbooru 标签提示词就是这种情况，它只有 `Aim for 10~30 tags` ——
> 标签模式下想加长描述，建议直接从六套自然语言预设里挑一套，而不是靠 `custom`。）

实测（4090D + Qwen3.8-27B Q4，同一张图、同一套 `scene` 预设）：

| `desc_length` | 输出字符 | 词 | token |
|---|---|---|---|
| `preset` | 486 | 97 | 108 |
| `long` | 767 | 141 | 167 |
| `extra_long` | 974 | 179 | 207 |

`bench_tag_length.py`（`devtools/` 里有副本）就是跑这个对比的，换自己的图也可以：

```
E:\AI\ComfyUI-aki-v3\python\python.exe bench_tag_length.py --image my.jpg
```

#### 提示词存在哪、怎么改

预设不是写死在代码里的，而是放在：

```
ComfyUI-Qwen35-Enhancer/presets/tagging_system_prompts.json
```

首次加载节点时会自动生成（内容即上面六套）。直接编辑里面的 `prompt` 文本即可，
**改完保存、下一次执行就生效，不需要重启 ComfyUI**。

```json
{
  "_readme": "…（文件里自带的使用说明，别删）",
  "_version": 3,
  "presets": {
    "photoreal": {
      "label": "写实照片（English）",
      "lang": "en",
      "group": "photoreal",
      "format": "raw",
      "prompt": "Describe what is in the image in 1~3 plain sentences …",
      "refine_prompt": "Here is an EXISTING caption for the image above. …"
    },
    "photoreal_zh": {
      "label": "写实照片（中文）",
      "lang": "zh",
      "group": "photoreal",
      "format": "raw",
      "prompt": "用中文描述这张图片里有什么 …",
      "refine_prompt": "下面是这张图片现有的一段描述。…"
    }
  }
}
```

字段含义：

| 字段 | 作用 |
|---|---|
| `label` | 只给人看的名字，日志与报告里显示；下拉里用的是 key（`photoreal` 等） |
| `lang` | `en` / `zh`，决定这份写给谁读；`bilingual` 靠它选该出哪一份 |
| `group` | 把中英两份**配成一对**。双语时找同 `group` 的另一份；改主题只看 `group`，不看 key |
| `format` | `raw` / `tags_one_line`。选了这个预设就用它声明的格式，不用管节点上的 `output_format` |
| `prompt` | 系统提示词正文 |
| `refine_prompt` | **可选**。只在 `caption_mode != "off"` 时用；留空则按 `lang` 回落到内置校订指令（见「优化已有打标」） |
| `sync_prompt` | **可选**。只在跨语言同步（`bilingual_sync != "off"`）时用；留空则按 `lang` 回落到内置同步指令（见「双语互同步」）。老 JSON 不加这一项完全不影响 —— 它是纯附加字段，结构版本号没变 |

- 想**加一套**：在 `presets` 里加一个 key 就行，简写 `"mystyle": "整段提示词"` 也认
  （此时 `lang=en`、`group=mystyle`、`format=raw`）。要参与双语，就把中英两份设成同一个 `group`。
  **新增 / 改名 / 删除预设需要重载节点**（重启 ComfyUI）才会出现在下拉里 —— 这是
  ComfyUI 下拉列表在加载时就固定了的限制；只改文本不必重启。
- `custom` 是保留名，写进 JSON 会被忽略。
- JSON 写坏了不会拖垮节点：自动回退到内置六套，并在日志里报错。
- `_version` 用来提示结构升级：若你手上的 JSON 版本比代码旧（例如缺 `lang` / `group` / `refine_prompt`），
  日志会**提醒一次**但**不会**自动覆盖你的文件 —— 里面有你手改的内容，是否升级交给你决定。

> 选了预设时 `system_prompt` 与 `output_format` 都会被**忽略**。这样即使 widget 里
> 留着上次改到一半的旧值，也不会把预设悄悄顶掉。

### 默认系统提示词（`custom` 模式）

`system_preset=custom` 时用的是这一套，它要的是 **danbooru 风格一行逗号标签**
（`1girl, solo, long_hair, ...`），顺序固定为：主体数量 → 主体 → 外观 → 衣着 →
姿态/视角 → 背景 → 光照/风格，并明确禁止质量词与 `<lora:...>`。作为备用的标签风格基线保留。

**想要自然语言描述，不必手写提示词 —— 从上面六套预设里挑一套即可**（它们已经各自带了
`format=raw`）。若坚持自定义，就把 `system_prompt` 整个换掉，并把 `output_format` 切成 `raw`：

```
Describe the image in 2~4 English sentences for LoRA training captions.
Cover the subject, what it is doing, the setting and the lighting.
Output the caption only — no tags, no bullet points, no preamble.
```

`output_format=raw` 保留换行与原始措辞；`tags_one_line` 则去围栏、去项目符号、
换行转逗号、去重保序（`2girls` 这类主体数目标签**不会**被当成序号吃掉）。

### 优化已有打标（`caption_mode`）

已经有一批 caption，想让模型**在它基础上改好**。两个典型场景：

- **批量改写**：之前用别的工具打了一批，风格词一堆，想统一改成「只写内容」的口径。
- **反复迭代**：模型给了一版，你自己手工改了几句，再让模型接着优化。

把 `caption_mode` 从 `off` 改成 `refine`（或 `refine_or_new`），其余照旧。模型收到的
不再是「从零写」，而是 **图片 + 已有 caption + 校订指令**：保留正确的、删掉风格词与
画面里没有的东西、补齐漏掉的，输出一版完整描述。

| `caption_mode` | 行为 |
|---|---|
| `off`（默认） | 从零打标。已有非空 txt 的图**跳过** —— 与以前完全一致 |
| `refine` | 只优化**已有 txt** 的图；旁边没有 txt 的图跳过 |
| `refine_or_new` | 有 txt 就优化；没有的就从零补写一张 |

#### 要不要动这一张：看**内容变没变**，不是看「处理过没有」

这是本功能的关键。每次写回都会在同目录留一份 `图名.txt.q35state`，里面记着
**这次输出的内容指纹**（外加配置指纹）。下次跑时：

- 当前 txt 的指纹 == 记录里的输出指纹 → **内容没变，跳过**：不覆盖、不重跑、不烧显存。
- 当前 txt 的指纹 ≠ 记录 → **内容变了** → 继续优化，并把**你改的那一版**当作新初稿。

由此得到几条很实用的性质：

- 跑第二遍**不会**把结果再改写一遍 —— 没变就是没变；
- 你手工改完再跑，**不需要任何额外开关**，它自己就认出来了；
- 换了 `system_preset` / 语言 / 字符上限 / 提示词，即使文件没动也会重做
  （否则「换预设」等于没换 —— 内容指纹一样，配置指纹不一样）；
- 想无视指纹强制重跑，把 `overwrite` 设成 `overwrite` 即可。

> 早期版本用「有没有处理过」当闸门，结果是**手改完之后再也优化不了**（会被当成
> 「已处理」跳过）。现在改成按内容指纹判定，两个方向都成立。

#### 原稿去哪了：`.orig` 保留最初那一版

写回是**原地覆盖** `图名.txt`，但在第一次覆盖之前会先把原稿按**字节**备份成
`图名.txt.orig`：

- `.orig` **只写第一次**。之后不管重跑多少次、`overwrite` 怎么设，都不会再动它 ——
  你最初那一版永远留着；想还原，把它复制回去覆盖 `图名.txt` 就行。
- `图名.txt.q35state` 是机器记录，**删掉无副作用**：只会让这张下次多跑一遍。
- 从零补写的那几张（`refine_or_new`）本来就没有原稿，所以不会留 `.orig`。
- 写盘失败时刚建的 `.orig` 会被撤回、也不记账 —— 不会出现「txt 没改、看着却像处理过了」。

```
E:\datasets\mydata\
├── 001.png
├── 001.txt          ← 优化后的最终描述
├── 001.txt.orig     ← 第一次优化前的原稿（字节级备份，只留一份）
├── 001.txt.q35state ← 输出指纹记录（删了只会重跑一次）
└── 002.png / 002.txt
```

#### 校订指令写在哪

校订指令是**用户侧**文本；系统提示词仍是预设那份，负责「只写内容不写风格 / 用哪国语言 /
一句到三句」这些全局口径。所以每条预设可以在 JSON 里单独配一份 `refine_prompt`；
没配就按 `lang` 回落到内置的英文 / 中文指令，`custom`（没有 lang）则回落到
「**与你原稿同一种语言**」那一份。改完保存、下次执行生效，不用重启 ComfyUI。

三份内置校订指令里都写了「**不要顺手消毒**」—— 改写时不许把原稿里的裸露 / 性内容
洗掉或含糊化，该直接写就还直接写。

#### 双语下的表现

双语 + refine 时，**每种语言各读各的初稿**：英文那份读 `图名.txt`、中文那份读
`图名_zh.txt`，各自与自己的旧版比对指纹。所以「英文早改好了、中文还没弄」这种状态
会被精确地只补中文那一半。

#### 双语互同步（`bilingual_sync`）：改中文，英文自动跟上

实际用法常常是**中文顺手改、英文才是最终产物**。但两份各自独立判定的话，你改完中文、
英文那份还停在旧版 —— 越跑越对不上。`bilingual_sync` 补的就是这一刀。

前提：`bilingual=en_then_zh` 且 `caption_mode != off`（没有两份就没有可同步的东西）。

| 档位 | 行为 |
|---|---|
| `auto`（默认） | 只有**一份**被手改过 → 以它为源把另一份改写过去；两份都被改过 → 各改各的，谁也不覆盖谁 |
| `zh_to_en` | 中文被改过时**无条件**以中文为准，英文那份的改动会被覆盖 |
| `off` | 两份各判各的（旧行为，一个字都没变） |

「被改过」用的还是上面那套判定：文件现在的指纹 ≠ 上次输出的记录。

**典型用法**：只改 `图名_zh.txt`，跑一次 —— 中文先被校订，英文再以**校订之后**的中文
为源重写一遍。源排在目标前面执行，所以英文翻到的一定是最新版，而不是你在文件里手打
的那一版（两者只差一道校订，但校订会把风格词、错漏一起清掉，所以顺序很重要）。

几个刻意的设计：

- **是翻译，不是改编。** 同步指令里写死「一一对应、不增不删」和「不要照着图片重新描述，
  也不要顺手纠正」—— 少了这两句，模型会把你改的内容当又一份初稿重写，改动就丢了。
- **两份都改过时 `auto` 不硬来。** 两边都有你的手笔，代码判不出谁对谁错，硬盖一定丢东西。
  报告里会单列一行提示（`双边手改: N 张`）；真要以中文为准就切 `zh_to_en`。
- **幂等。** 同步完再跑一次是**零生成** —— 不会每次都把中文重翻一遍。
- **目标那份的旧内容会备份成 `.orig`**（仍然是只留第一次那份），翻崩了能还原。
- **源这一趟失败时**，目标那份不会拿空内容去翻，而是明确报错跳过，下次自动重来。
- **`zh_to_en` 加了一道闸**：中文没被改过时它退回常规判定，不会每次跑都重刷一遍英文。

报告里会有一行如实说明当前档位与它实际做了什么：

```
  双语同步    : auto —— 哪一份被手改过就以它为源改写另一份；两份都被改过则各改各的
  初稿        : 待优化 1 次，跨语言同步 1 次，内容未变跳过 0 次
```

> `sync_prompt` 也做成了预设 JSON 的可选字段（与 `refine_prompt` 同构），没配就按
> **目标语言**回落到内置的英文 / 中文指令。

> 实测（4090D + Qwen3.8-27B Q4，白底 + 左红方块 + 右蓝圆的合成图）：手改中文之前，
> 英文那份是 `…while a blue circle is positioned to its right.`；改完中文跑一次，
> 英文被重写成 `Two geometric shapes are placed side by side on a light gray background,
> with a red square on the left and a blue circle on the right.` —— 叙述跟着中文走，
> 而不是拿自己那份旧稿再校一遍。这一次只发生 **2 次生成**（复用已加载的后端，2.7s），
> 再跑第三次是 **0 次生成**。
>
> 首次优化（两份都还没有输出记录）**不会**被当成「双边手改」—— 没记录 ≠ 你改过，
> 那种情况老老实实各改各的，报告里也不会出现那句警告。

### 一轮带图自检（`verify`）：认错东西怎么救

模型「认错东西」其实很少是把 A 认成 B，绝大多数是**细节层面的漂移**：嘴张着还是
闭着、画面切到肩膀还是胸口、丝带在颈后还是两侧、发色算 silver 还是 blonde、
左边还是右边。这类断言每条都能回到图上核对，所以让它**对着图逐条查一遍**就能救回来。

把 `verify` 设成 `once`，每张图会多跑一次推理：第一次照常生成描述，第二次把
**图 + 刚写好的描述**一起送回去，要求它：

1. 逐条列出描述里所有可核查的断言（数量、发色、瞳色、每件衣服、每个配饰、姿势、背景…）；
2. 每条回到图上判 `SUPPORTED` / `CONTRADICTED` / `UNCLEAR`，并写出「图里实际是什么」；
3. 只改判错的那部分，输出到 `CORRECTED:` 段。

**为什么必须这么问**：泛泛地问「你觉得写得对不对」会被自洽偏差带跑 —— 模型会顺着
自己刚写的话往下圆，把错的也说成对的。拆成一条条具体断言、并逼它重新看一次图，
它才不得不改口。同理，必须写明「只改判错处、禁止顺手润色」，否则它会把这道工序
当成「再写一遍」，没认错的地方也被改掉。

实测（4090D + Qwen3.8-27B Q4，两张角色图）：

| 原描述里的错 | 自检后改成 |
|---|---|
| 「画面只切到肩膀」（实际露出上半身） | `head, neck, and upper torso` |
| 「嘴微微张开」（实际是闭合的） | `mouth closed` |
| 「颈后有一条红色丝带」（实际两侧都有） | `red fabric visible on both sides of their neck` |

**已写对的句子一个字没动。**

**成本要说清楚：约 ×5，不是 ×2。** 实测同一批两张图，`off` 4.97s /
`once` 28.32s（5.7×）—— 因为自检要连 CHECKS 清单一起输出，token 数是正文的
4~5 倍（88 tok/次 → 464 tok/次）。所以它更适合「关键图才开」：先全量跑一遍
`off`，再对不放心的那批开 `once`，或者用 `limit` 先试一小批。

#### 它不是只会做好事：区分「被证伪」与「无法确认」

第一次真机验证（同样两张图）就抓到一次**信息丢失**：`white ruffled top with
frilled straps` 被改成 `white ruffled top` —— 荷叶边客观上是在的，模型只是
没能在这一步确认，就当成错处删了。

所以提示词里专门补了一条：

> A claim may be changed ONLY if you judged it CONTRADICTED. If it is merely
> UNCLEAR, keep it exactly as written: failing to confirm a detail is not proof
> that it is absent.

即只有**被证伪**的断言能动，**无法确认**的必须原样保留。少了这句，模型会把
「我没看清」当成「它不存在」，结果越改越短 —— 比原稿更糟。

补上这条之后复测（同样两张图）：被删的 `with frilled straps` 保住了（那张判定为
「无错、不动」，一个字没改），另一张只把 `mouth slightly open` 改成 `mouth closed`
—— 而那确实是原稿写错的地方。

#### 只跑一轮，不做迭代

这是刻意写死的：实测**第二轮会来回震荡**。同一张图第二轮把已经正确的
`pinkish-red` 改成 `salmon-pink`、再改成 `maroon`，还自相矛盾（同一块布料，
先判该叫 `collar` 又说该叫 `scarf`，两条都标 CONTRADICTED）。

原因不难理解：颜色、材质这类描述本来就没有唯一正确答案，第二轮总能「挑出新毛病」。
**没有客观的停止信号，迭代就没有收敛点。** 所以 `verify` 只有 `off` / `once` 两档。

#### 改不坏：所有失败路径都退回原稿

`_extract_corrected()` 是唯一决定「要不要采用修正稿」的地方，判据全部朝保守一侧倒 ——
宁可保留一份有错的原稿（你肉眼能改），也不把一个残缺或跑偏的结果静默写进数据集：

| 情况 | 状态 | 结果 |
|---|---|---|
| 正常 | `fixed` | 采用修正稿 |
| 输出里没有独占行首的 `CORRECTED:` | `no_mark` | 保留原稿 |
| 修正段是空的 | `empty` | 保留原稿 |
| 修正稿长度 < 0.4 倍 或 > 2.5 倍原稿 | `bad_len` | 保留原稿 |
| 修正稿与原稿逐字一致 | `same` | 保留原稿（本来就没查出错） |
| 原稿本身是空的 | `skipped` | 什么都不做 |
| 自检这一步整个抛异常 | — | 保留原稿，只记一条 warning |

自检的 `max_new_tokens` 自动取 `max(1536, max_new_tokens × 3)`：它要连 CHECKS
清单一起输出，比正文长得多（实测 350~400 token 起步）。这是上限不是配额，写够了就停。

> 建议组合：`verify=once` + `max_image_side=1536`。前者治「看出来了但没写对」，
> 后者治「根本没看清」—— 两者治的是不同的错。保真度那项几乎免费，**建议一直开着**；
> 自检贵 5 倍，按这批图的重要性决定要不要开。

### 速度：所有优化都复用，且整个文件夹只加载一次

这个节点直接继承扩写节点的加载链路，所以**下面这些优化一个都不少**：

- 视觉塔 `patch_embed` 的 `Conv3d` → 等价 GEMM 替换（实测 586x）。批量场景**每张图都要过
  视觉塔**，漏了它单张 prefill 会从 0.05s 变成几十秒，一个文件夹基本没法跑
- 量化 `none` / `4bit` / `8bit`，含跳过视觉塔与线性注意力里两个 `Linear(4096,32)` 小投影
- 8bit 走 `llm_int8_threshold=0.0` 的纯 `int8_scaled_mm`
- 注意力后端选择，失败自动退回 `sdpa`
- 加载期全部体检日志（权重分布 / 视觉塔入口 / 线性注意力 / 量化档）

在此之上多了一条**只有批量才有**的收益：**加载只做一次，摊到 N 张图上**。
若改用扩写节点循环 N 次且 `keep_model_loaded=False`，每张都要重载
（本机 bf16 实测 23.05s/次）。两个节点还**共用同一份常驻模型缓存**，
所以工作流里同时放批量和扩写也不会各加载一份 17.5GiB。

### 日志长什么样

逐张一行（带标签数 / token / 字符数 / 耗时 / prefill 与解码分解），末尾给汇总与失败清单：

```
[Qwen35] [ 1/24] ✓ im01.png -> 21 标签 / 58 tok / 96 字符 / 3.4s（prefill 0.42s、解码 2.9s = 19.8 tok/s）
[Qwen35] [ 2/24] ✗ im02.png -> UnidentifiedImageError: cannot identify image file
```

```
[Qwen35] ========== 打标汇总 ==========
  成功 / 失败 : 23 / 1
  标签合计    : 487 个（平均 21.2 个/张）
  打标耗时    :  78.31s（3.40s/张，17.1 tok/s）
  首张 prefill:   0.42s（含 CUDA 预热与 kernel 编译，后续张比它快是正常的）
  ⚠ 失败清单（1 张；原图与已有 txt 均未被改动）:
     - im02.png：UnidentifiedImageError: cannot identify image file
```

开了双语（`bilingual=en_then_zh`）或字符上限（`max_output_chars>0`）时，会多出对应信息：

```
[Qwen35] [ 1/24] ✓ im01.png [英文] -> 1 片段 / 34 tok / 78 字符 / 2.1s …
[Qwen35] [ 1/24] ✓ im01.png [中文] -> 1 片段 / 40 tok / 62 字符 / 2.3s（已按上限截断）
[Qwen35] ========== 打标汇总 ==========
  生成次数    : 26 次（13 张 × 2 语言；以下按「次」计）
  成功 / 失败 : 26 / 0
  片段合计    : 26 个（平均 1.0 个/次）
  按上限截断  : 4 条（max_output_chars=80）
  ⚠ 失败清单（1 张；原图与已有 txt 均未被改动）:        ← 双语时条目带语种
     - im07.png [中文]：输出为空（…）
```

失败**只算这一张**：原图不动、已有 txt 不动、剩下几张继续。双语时同一张的另一种语言
照跑，不会因为英文失败就丢掉中文那份。但你在前端点「取消」时会立刻中止整批：已写好的
txt 保留、进度条收尾、模型按 `keep_model_loaded` 释放。这与「单张失败容错」是两套不同
语义，不要混。

开了优化已有打标（`caption_mode=refine` / `refine_or_new`）时，头部会多出这几行，
每张的日志带 `（校订原有）` 或 `（从零补写）`：

```
[Qwen35] ========== 批量打标 ==========
  打标模式    : 优化已有打标（旁边没有 txt 的图跳过）
  系统提示词  : photoreal（预设：写实照片（English））
  初稿        : 待优化 18 次，内容未变跳过 6 次
  变更判定    : 拿上次输出的指纹比对，内容没变就不动它；你手动改过的会自动继续优化…
  写回        : 原地覆盖 .txt；原稿按字节备份成 <图名>.txt.orig（只留第一次那份）…
  …
[Qwen35] [ 1/24] ✓ im01.png（校订原有） -> 1 片段 / 51 tok / 132 字符 / 3.1s …
[Qwen35] ========== 打标汇总 ==========
  成功 / 失败 : 18 / 0
  原稿备份    : 12 个 <图名>.txt.orig（字节级备份，只留第一次那份，可随时还原）
  输出记录    : 18 个 <图名>.txt.q35state（记的是输出指纹；删掉只会让这张下次重跑一遍）
```

（另一条常见用法：想「重新审视一遍已经优化过的图」时，不用删任何东西 —— 把
`overwrite` 设成 `overwrite` 跑一次即可，`.orig` 不会被覆盖。）

## 显存、耗时与优化

### 显存（24GB 卡参考）

| 量化 | 扩写模型占用 | 真机 A/B 整轮（9B，同工作流） | 建议 |
|------|------|------|------|
| **`none`** (bf16) | **~17.5GiB** | **51.85 s**（加载 23.05s｜解码 20.9 / 21.6 tok/s） | **默认，显存够就用它** |
| `4bit` (nf4) | **~7.9GiB** | **54.06 s**（加载 19.60s｜解码 17.3 / 19.7 tok/s） | **要省显存就用它 —— 只慢 4%** |
| `8bit` | ~11.1GiB | **71.01 s**（加载 33.90s｜解码 12.4 / 13.1 tok/s） | 不建议（省 6.4GiB 换慢 37%） |

这三行是**同一台机器、同一工作流、同一组参考图**的实跑（`comfyui.log` 2026-10-07
06:03 / 06:09 两次，加上 04:55 的 bf16 基线），不是估算。
**要点：8bit 多花的 16.95 秒里，14.30 秒（84%）花在「加载模型」上，只有小头在解码。**
所以"把解码调快"治不了它 —— 换档才有用。

受控对照实测（`bench_qwen35_proj_ab.py`：真实层形状、bf16 与候选**背靠背交替**计时、
各取最小值，避免上一次那种被 GPU 抢占污染的数字）：

| 组（每 token 调用次数） | int8 t=6.0 | int8 t=0.0 | 4bit nf4 |
|---|---|---|---|
| MLP ×32（96 次，4.83B 参数，占参数大头） | ~1.64x | ~0.95x | **0.49x（比 bf16 快一倍）** |
| 线性注意力投影 ×24（120 次，1.62B 参数） | 5.62x / 3.45x | 3.23x / 1.82x | 1.95x / 1.62x |
| 全注意力投影 ×8（32 次，0.34B 参数） | 7.21x / 4.30x | 4.18x / 2.16x | 2.39x / 2.07x |

（比值 >1 表示比 bf16 慢；后两行是 batch=1 / batch=256。按你实际负载
prefill 3000 tok ×2 + decode 543 tok 加权：**4bit 总投影成本约 1.0x bf16** ——
这条当时标注为「外推」，现已被上面的真机整轮 **1.04x** 证实；
8bit 在 t=0.0 时约 1.66x、默认 t=6.0 约 2.89x。）

**`none` 与 H3 不能同跑**（17.5 + 33GB 装不下），靠
`unload_other_models=True` 错峰即可 —— 这本来就是默认行为。
**别为了"省点显存顺便提速"选 8bit，它两头都不赚**（详见下面「量化是省显存、不是提速」）。

**调度建议**：H3 本身有 33B，24GB 卡上必须和扩写模型错峰。保持
`unload_other_models=True` + `keep_model_loaded=False`，扩写完自动释放再跑 H3。

### 进度与中断

任务被拆成几段，进度条按权重映射到 0~100：

| 阶段 | 进度区间 | 粒度 |
|------|---------|------|
| 卸载其他模型 | 0 → 2% | 阶段点 |
| **加载扩写模型** | 2 → 35% | 阶段点（加载期间停在 3%，见「已知限制」） |
| 图片预处理 | 35 → 38% | 阶段点 |
| **生成英文提示词** | 38 → 60% | **每生成一个 token 更新一次** |
| **翻译中文预览**（仅双语时） | 60 → 100% | **每生成一个 token 更新一次** |

`bilingual=off` 时后两段合并，生成独占 38 → 100%。

> 英文段跑完后，节点会拿实际输出长度重算一次分界点（`pbar.rescale`），
> 所以英文段显示的 38 → 60% 只是先验宽度，真正的分界点会随实际 token 比移动。
> 重算只前进不后退，进度条不会回跳。

两条出口同时生效：

1. **前端** —— 该节点上出现原生进度条（走 `comfy.utils.ProgressBar`，
   经 `main.py` 的进度 hook 发 websocket `progress` 事件）。
2. **控制台** —— 阶段切换即时打一行；生成阶段每 `progress_interval` 秒打一行：

```
[Qwen35] 正在加载模型：Qwen3-VL-8B-Instruct（量化=none，首次或换模型约 30~90s，请耐心等待）
[Qwen35] 模型加载完成，用时 38.4s
[Qwen35] 参考图 1 张：1024x1536
[Qwen35] 开始生成：输入 1148 tok，上限 1024 tok
[Qwen35] 生成 ██████░░░░░░░░░░░░░░░░  25.1%  128/512 tok   28.4 tok/s  剩余 ~13.5s
[Qwen35] 生成 ██████████████████████ 100.0%  512/512 tok   28.0 tok/s
```

如果 stderr 是真实终端，还会用 `\r` 原地刷新一条实时条（日志行出现前会自动擦除，
不会串行）。终端码页画不出方块字符时自动回落成 `#` / `-`。

**中断**：生成阶段的每一步都会检查 ComfyUI 的中断标志，因此前端点「取消」能真正
打断一次可能长达数十秒的生成，不必等 `max_new_tokens` 跑完。控制台进度行末尾的
`剩余 ~Ns` 是线性外推的估算值，仅供参考。

### 耗时构成

节点每次运行都会在控制台打印耗时分解，方便定位瓶颈到底在哪：

```
[Qwen35] ========== 耗时分解 ==========
  卸载其他模型      :   0.12s
  加载扩写模型      :  38.40s  (本次新加载 加载)
  图片预处理        :   0.08s  (1 图, 长边上限 1280)
  生成英文提示词    :  18.60s  输入 1148 tok -> 输出 520 tok  (28.0 tok/s)
  翻译中文预览      :  16.20s  输入 1305 tok -> 输出 448 tok  (27.7 tok/s)
  卸载扩写模型      :   1.20s
  --------------------------------
  合计              :  74.60s
```

**若「加载扩写模型」占大头**（多数情况下如此）—— 瓶颈不在推理，而在于每次都要把
十几 GB 权重读进内存、量化、再搬上显卡。对策：

1. **让模型常驻**：`keep_model_loaded=True`，省掉每次的加载开销。代价是常占
   6~10GB 显存，需要相应减少 H3 的显存占用。
2. **换更小的模型**：`Qwen3-VL-4B` 之类，加载与生成时间都成比例下降。
3. **降低量化位宽**：`4bit` 对显存和系统内存的压力都更小。

**若「文本生成」占大头** —— 节点会把生成拆成 prefill / 解码两行，因为两者变慢的
原因完全不同：

- **prefill 慢**（每视觉 token 明显高于 1 ms）→ 视觉塔入口问题，见上面 Conv3d 那节。
  这时才轮到 `max_image_side`。
- **解码慢** → 先看加载时那行「线性注意力」。若是 Qwen3.5 且报「走 torch 回退」，
  装 `fla` 能省约 1/3（见上一节）。**这一步和 `max_image_side` 完全无关**。

降低 `max_new_tokens`（H3 三段式通常 300~600 token）也直接按比例减少解码时间。

### 量化是省显存、不是提速 —— 8bit 纯亏（还亏在加载上）；要省显存请用 4bit

9B 上五次运行，**同模型、同一工作流、同一组参考图，只改量化档**：

| 运行 | 量化 | 加载 | 英文解码 | 中文解码 | 整轮 |
|---|---|---|---|---|---|
| 04:55 | `none` | 23.05 s | **20.9 tok/s** | **21.6 tok/s** | **51.85 s** |
| 05:05 | `8bit`（t=6.0 旧） | 27.07 s | 6.1 tok/s | 6.3 tok/s | 119.34 s |
| 05:11 | `8bit`（t=6.0 旧） | 33.52 s | **2.3 tok/s** | **2.3 tok/s** | 262.85 s |
| 05:30 | `8bit`（t=6.0 旧） | 33.22 s | **2.6 tok/s** | 6.4 tok/s | **319.38 s** |
| **06:03** | **`4bit`** | 19.60 s | 17.3 tok/s | 19.7 tok/s | **54.06 s** |
| **06:09** | `8bit`（t=0.0 新） | **33.90 s** | 12.4 tok/s | 13.1 tok/s | **71.01 s** |

**最后两行是本轮的真机定案**，回答"8bit 还能不能优化"：

| 对比 | 加载 | 解码（墙钟） | 整轮 |
|---|---|---|---|
| `4bit` → `8bit` | **+14.30 s（占增量的 84%）** | 31.10 → 31.55 s（几乎持平） | +16.95 s |

解码**墙钟**几乎没变，是因为 8bit 那次生成的 token 更少（EN 195 vs 286、
ZH 207 vs 288）—— 按**每 token** 算，8bit 仍慢 1.4~1.6 倍（17.3→12.4 tok/s）。
也就是说：**8bit 的代价有两处，加载占大头（84%），解码占小头。**
只调解码参数（比如 `llm_int8_threshold`）最多把它从 319s 拉回 71s，
但永远追不上 4bit 的 54s —— **只有换档才治得好。**

顺带修正一条早先的说法：`threshold` 改成 0.0 之后，8bit 解码从 2.6~6.4 tok/s
提升到 12.4~13.1 tok/s（约 2~5 倍），但**仍慢于 4bit**，所以"8bit 是负收益"这个
结论不变，只是程度比最早记录的"慢 3~8 倍"轻。对照 Qwen3-VL-8B 同样成立：
`none` 22.6~23.4 tok/s vs `8bit` 7.0~7.1 tok/s。

原因是 `LLM.int8()` 的算子路径在这台机器上先天不利，受控对照实测分解为三条：

1. **bf16 已经没有余量可省**：batch=1 时 bf16 的 `Linear` 实测 **754~855 GB/s**，
   4090D 峰值约 1008 GB/s —— 权重流本来就接近极限。int8 砍一半字节理论上限也就 2x，
   而实测 bnb 的 int8 内核只有 **44~117 GB/s**，反而更差。
2. **调用次数太多、矩阵太小**：Qwen3.5-9B 每 token 约 **248 次** Linear 调用
   （24 层线性注意力 ×5 + 8 层全注意力 ×4 + 32 层 MLP ×3），bnb 每次都有固定开销。
   最典型的是全注意力投影组：只有 0.34B 参数，int8 却慢 **7.21x** —— 慢的全是开销。
3. **混合内核额外更贵**：bnb 0.49.2 的 `MatMul8bitLt.forward` 里，`threshold > 0`
   走 `int8_mixed_scaled_mm`，`== 0` 走 `int8_scaled_mm` —— 这是**算子二选一**，
   不是数值参数。transformers 默认给 6.0，本节点已改为 **0.0**，稳定快 **1.7~1.9 倍**。
   （注意"离群列是否真触发"不是主因：阈值 6.0 是绝对值，post-RMSNorm 激活 RMS≈1，
   batch=1 时几乎挑不出离群列，核仍贵；实测两种阈值的数值误差完全相同 ——
   混合内核只是更贵，并不更准。）
4. **加载也要重新量化一遍**：第 ⑤ 条真机数据说明 8bit 还额外拖慢加载 73%
   （33.90s vs 4bit 19.60s）——`LLM.int8()` 要在加载时逐层算 absmax / 拆离群列，
   而 4bit 的 nf4 分块量化便宜得多，甚至比 bf16 直载（23.05s）还快。
   这一条是"解码之外"的代价，只盯 `tok/s` 是看不到的。

权重也没真砍半 —— 实测 `8bit` 是 `int8 6.4GiB + bfloat16 4.7GiB`，17.5 → 11.1 GiB
（1.58×）；`4bit` 是 `uint8 3.2GiB + bfloat16 4.7GiB`，17.5 → **7.9 GiB（2.2×）**。
那份 `bfloat16 4.7GiB` 两组都有：embedding 与 lm_head（各约 2GiB）不是 `nn.Linear`
或落在 skip 名单里，norm / 视觉塔本来也不量化 —— 所以量化能动的只有中间 6.6B 参数。

> 这里还错过一次推理方向：曾用「解码等效带宽只有峰值的 37%」推出「8bit 应有收益」——
> **方向正好反了**。利用率低恰恰说明瓶颈不在带宽，那砍权重字节就无从发力。

→ **要省显存请用 `4bit`，不要用 `8bit`。** 真机 A/B：4bit 整轮 **54.06 s vs bf16
51.85 s（1.04×）**、驻留 **7.9GiB vs 17.5GiB** —— 它才是"省显存但不减速"的那个档。
`none` 则是显存够时的默认；8bit 只在已经选了它、又不想重载模型时才值得用
（此时本节点已自动把 threshold 调到 0，少亏一点）。

**若想省掉翻译那一份开销** —— 设 `bilingual=off`。双语只多花第二次生成的时间：
模型此时已在显存里，第二次是**纯文本** prefill（不带图，约 720~790 token、1~2 秒）
加上中文生成（约英文 token 数的 1.1 倍），合计约 +20%。**不需要**为了翻译再加载
第二个模型 —— 那反而更慢。若实测多花了 2~3 倍，先查「双语的真实开销」一节。

> 另需留意**系统内存**：8B 模型加载阶段 CPU 侧需要十几 GB 可用内存。
> 若 ComfyUI 已常驻大量内存，加载会被迫走页面文件，速度显著下降甚至失败。

### 图片分辨率的影响

Qwen3-VL 的 vision token 数 = `(w/32) × (h/32)`（16px patch + 2×2 merge，实测吻合）。
实测（已含 615 token 的 system prompt）：

| 输入 | 总 token | 相对纯文本 |
|------|---------|-----------|
| 纯文本 | 648 | 1.0x |
| + 1 图 512² | 906 | 1.4x |
| + 1 图 1024² | 1674 | 2.6x |
| + 1 图 1536² | 2954 | 4.6x |
| + 4 图 1024² | 4752 | 7.3x |

所以 `max_image_side` 默认设 **1536**：比 1280 只多约 44% 的图片 token
（单图 prefill 1.0s → 1.5s），小物件与细节的认对率却明显更高 ——
「认错东西」里有相当一部分就是这一步压掉的分辨率。再往上收益递减、代价却很陡。

**编码格式是另一个隐藏的坑，而且修它不要钱。** 图像缩完之后早先是重新存成
JPEG q92 —— 实测把一张 2620KB 的 PNG 压到 274KB，差 10 倍，睫毛、发丝、
细小配饰这类细节全糊在这一步。而视觉 token 数只由**尺寸**决定，跟编码格式
无关，所以改成存 PNG 是**无损且零 token 成本**的（代价只有请求体变大，
本地回环无所谓）。现在默认 `lossless=True`，即输出 PNG。

## 已知限制

- **一轮自检不是万能的**。「改不坏」保证的是**格式异常时回退原稿**，不保证
  「模型判错的地方一定能改对」，也不排除它把本来写对的地方顺手改掉（实测很少，
  但不是零）。第一次用建议先 `limit=20` 小批量试跑、看一遍再全量。
  另外它治的是「看出来了但没写对」，治不了「根本没看清」—— 那要靠
  `max_image_side`（或者对可疑区域裁出来放大再看一遍，本轮没做）。
- **中文预览是同一个模型的二次生成**，不是独立翻译插件。绝大多数情况下准确，
  但专有名词、品牌名可能有轻微偏差 —— 核对时以英文版为准。
- **中文标签的还原只覆盖常见译法**。模型偶尔会把三段标签也翻成中文，节点内置了
  `综合多模态描述:` → `integrated_multimodal_description:` 之类的映射兜底，
  但不可能穷举所有译法。因为中文版不喂 H3，即使没兜住也只影响阅读观感。
- **未在所有模型上实测多图参考**。`reference` 模式依赖模型主动输出 `<Picture i>`，
  指令遵循能力弱的模型可能忽略；节点通过强制系统提示缓解，但不保证 100%。
- **去审查（abliterated）模型容易漏首段标签**。节点用输出规范化兜底补齐，
  但若模型连第二三段都写不全，建议换回 Instruct 版。
- **加载阶段没有细粒度百分比**。`transformers` 的 `from_pretrained` 不暴露分片级回调，
  所以加载期间进度条停在 3%，只靠控制台文字提示。生成阶段才是逐 token 的真实进度。
- **双语互同步是「翻译」不是「校订」**：目标那份只做语言转换，会连源文本里的错误
  一起搬过去（哪怕描述与图片不符）。这是刻意的 —— 你手改的内容优先级最高，
  让它「顺手纠正」等于把改动丢掉。源那一份自己在同一趟里会先被校订一遍；
  真想让它凭图纠正，就别手改，直接用 `overwrite=overwrite` 重跑两份。
- 需要 `transformers >= 5.0`。低版本无法识别 `qwen3_5` 架构。

## 致谢

- 示例工作流基于 [ComfyUI 官方 MiniMax H3 模板](https://github.com/comfyanonymous/ComfyUI) 改造
- 提示词规范来自 [MiniMax H3](https://www.minimax.io/blog/minimax-h3)
- 模型权重由 [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) 提供

## License

[MIT](LICENSE)
