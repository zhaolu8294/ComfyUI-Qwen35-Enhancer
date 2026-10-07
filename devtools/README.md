# devtools — 开发脚本备份

**这是什么**：节点本体（`nodes.py` 等）一直在仓库根目录，但开发过程中写的**测试、基准、
诊断、探针脚本**此前只存在于本机 WorkBuddy 工作区（`h3_workflow/`）。工作区一旦丢失，
400+ 项断言的回归测试和全部性能证据都没了。本目录就是那批脚本的备份副本。

**这是副本，请勿在这里编辑。**改工作区，然后重新同步：

```bash
cd <工作区>/h3_workflow
E:\AI\ComfyUI-aki-v3\python\python.exe sync_devtools_to_repo.py --dry-run   # 先看变化
E:\AI\ComfyUI-aki-v3\python\python.exe sync_devtools_to_repo.py            # 正式同步
```

同步只覆盖有变化的文件，可反复执行。

## 内容（125 份备份副本，约 1.4 MB）

| 类别 | 数量 | 说明 | 代表文件 |
|---|---|---|---|
| 回归测试 | 8 | 纯 Python，用测试桩替代 ComfyUI，**不占显卡** | `test_qwen35_tagger.py`、`test_qwen35_v6_bilingual.py` |
| 独立校验 | 4 | 复现修复效果、校验工作流结构 | `verify_h3_workflows.py`、`verify_qwen35_patch.py` |
| 性能基准 | 10 | 量化档 A/B、注意力后端、精度对照、**P/E 核单核性能** | `bench_real_quant.py`、`bench_core_type.py` |
| 故障诊断 | 13 | 定位 NaN、执行顺序、速度异常、**CPU 亲和性因果 A/B** | `diag_fla_nan.py`、`diagnose_core_pinning.py` |
| 探针 / 剖析 | 8 | 探查环境支持哪些算子、显存去向、**线程级 CPU 占用** | `probe_qwen35_linear_attn.py`、`probe_thread_cpu.py` |
| 火焰图 | 2 | 解析 py-spy speedscope dump | `analyze_pyspy.py`、`pyspy_thread_stacks.py` |
| 冒烟测试 | 3 | 端到端快速跑一遍，只给结论 | `smoke_verify.py` |
| 生成器 | 4 | 生成示例工作流、生成 `push.bat`（含 `h3_system_prompt.txt`） | `make_batch_tagger_example.py`、`make_push_bat.py` |
| 历史迁移 | 6 | widgets 数量演进时的一次性脚本，**均已应用** | `update_wf_17widgets.py` |
| 其他工具 | 10 | 只读检查、中文 token 比测量、**钉 P-core**、备份同步 | `pin_comfyui_pcores.py`、`sync_devtools_to_repo.py` |
| 原始日志 | 37 `.txt` + 5 `.out` | 脚本的真实输出，是 README / 诊断报告里那些数字的证据 | `_tagger_out.txt`、`bench_real_8bit.out` |
| 工作流 JSON | 9 | 见「已知注意点」第 1 条 | `h3_i2va_with_llm.json` |
| 文档 | 4 | 工作区的四份说明，含诊断报告 | `诊断报告_扩写变慢.md`、`README_h3_video.md` |

## 运行前提（硬约束）

- 本机 ComfyUI 装在 `E:\AI\ComfyUI-aki-v3`。**47 个脚本把 `CV` 硬编码成该路径**，
  换机器必须逐个改，或备好同名目录。
  （例外：`pin_comfyui_pcores.py` 只吃 PID，**不含任何硬编码路径**，可直接移植。）
- 必须用 ComfyUI 自带解释器：`E:\AI\ComfyUI-aki-v3\python\python.exe`（不是系统 Python）。
- 需要同样的模型：`minimax_h3`（视频工作流）、`Qwen3.5 / Qwen3-VL`（节点）。

> 这批脚本是**环境绑定的实验工具**，不是可移植的测试套件。备份的意义是保住代码和证据，
> 不是让它们在任意机器上一键跑通。

## 最常用的五条命令

```bash
cd devtools
E:\AI\ComfyUI-aki-v3\python\python.exe test_qwen35_tagger.py        # 批量打标节点，400+ 断言
E:\AI\ComfyUI-aki-v3\python\python.exe test_qwen35_v6_bilingual.py  # 扩写节点
E:\AI\ComfyUI-aki-v3\python\python.exe verify_h3_workflows.py       # 期望输出「总问题数: 0」
E:\AI\ComfyUI-aki-v3\python\python.exe verify_qwen35_patch.py       # 复现 patch_embed 修复，需要显卡
E:\AI\ComfyUI-aki-v3\python\python.exe pin_comfyui_pcores.py --set  # 把 ComfyUI 钉到 P-core（见下）
```

## 已知注意点

1. **⚠ 4 个同名工作流 JSON 可能滞后于 `../examples/`。**
   `h3_i2v_qwen35.json`、`h3_ref2v_multi_qwen35.json`、`prompt_enhancer_qwen35_image.json`、
   `prompt_enhancer_qwen35_text.json` 在本目录里是**工作区副本**（保持备份原貌），
   而 `../examples/` 是发布口径。例：`quantization` 在 `examples/` 已改为 `none`，
   本目录副本仍是旧的 `8bit`。**要判断示例工作流的正确状态，以 `../examples/` 为准。**
2. `WF` 变量（工作流查找目录）在同步时被改写为「脚本自身所在目录」，
   于是本目录下的脚本能在任意位置定位同级 JSON。工作区里的原件仍是绝对路径写法。
3. **未备份**：`*.bak_attention`（4 个示例工作流的修改前快照，与 `../examples/` 重复）、
   `__pycache__/`、`_tmp_*`（见 `.gitignore`，`verify_qwen35_patch.py` 的临时基准文件），
   一个早前 selftest 留下的垃圾文件 `returns`，
   **`_pyspy_native.json`（py-spy `--native` 的原始采样 dump，3.2 MB）** ——
   它对应的结论是错的（采样偏线程），但「为什么读错了」写在诊断报告第十四章里，
   要复现重跑 `analyze_pyspy.py` 即可，不必把 3 MB 原始样本塞进仓库；
   以及 **`_fla_probe*/`（4 个目录、700+ 文件、约 7 MB）** ——
   那是探 fla 版本兼容性时解包在工作区的**第三方包本体**
   （fla 0.5.2 / 0.2.2 / 0.3.2 / 0.4.2 的 site-packages），不是本项目的代码。
   同步脚本按前缀 `_fla_probe` 排除（`EXCLUDE_DIR_PREFIXES`）。
4. `update_wf_*.py` / `update_existing_enhancers.py` 是历史迁移脚本，已经跑过了，
   留着只为记录 widgets 顺序的演进过程，正常不需要再执行。
5. 本目录会随 git 一起进 ComfyUI Registry 的发布包（如果将来上架）。
   真要发布时，可用 `.comfyignore` 把 `devtools/` 排除掉。
