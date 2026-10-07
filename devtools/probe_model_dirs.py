# -*- coding: utf-8 -*-
"""探针：查 discover_local_models / _resolve_path 现状，找出「自定义路径失败」的原因。
用 ComfyUI 自带 python 跑（不需要显卡）。"""
import os, sys, types, importlib.util, json

NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
COMFY_ROOT = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
for p in (COMFY_ROOT, NODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- 造桩，让 nodes.py 能被导入（不打显卡）----
import folder_paths
fp = folder_paths
for k, v in [("models_dir", r"E:\AI\ComfyUI-aki-v3\ComfyUI\models"),
             ("get_folder_paths", lambda *a, **k2: []),
             ("get_filename_list", lambda *a, **k2: [])]:
    if not hasattr(fp, k):
        setattr(fp, k, v)
try:
    import comfy.utils  # noqa
except Exception:
    m = types.ModuleType("comfy.utils"); m.ProgressBar = object
    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules["comfy.utils"] = m
try:
    import comfy.model_management  # noqa
except Exception:
    m = types.ModuleType("comfy.model_management")
    class _IPE(BaseException):
        pass
    m.InterruptProcessingException = _IPE
    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules["comfy.model_management"] = m

spec = importlib.util.spec_from_file_location("qm_nodes", os.path.join(NODE_DIR, "nodes.py"))
QM = importlib.util.module_from_spec(spec)
sys.modules["qm_nodes"] = QM
spec.loader.exec_module(QM)

print("=" * 72)
print("1) SEARCH_DIRS 现状")
for d in QM.SEARCH_DIRS:
    ok = os.path.isdir(d)
    print(f"   {'OK ' if ok else 'MISS'} {d}")
    if ok:
        ents = sorted(os.listdir(d))
        print(f"        共 {len(ents)} 项: {ents[:12]}")
        for e in ents:
            full = os.path.join(d, e)
            if os.path.isdir(full):
                has_cfg = os.path.isfile(os.path.join(full, "config.json"))
                print(f"        - [DIR] {e}  config.json={'有' if has_cfg else '无'}")

print()
print("2) discover_local_models() 返回：")
mapping = QM.discover_local_models()
if not mapping:
    print("   （空！下拉框会显示 <no local HF model found>）")
for k, v in mapping.items():
    print(f"   {k!r} -> {v}")

print()
print("3) _model_choices() =", QM._model_choices())

print()
print("4) _resolve_path 各种输入的行为（用真类的一个实例，不加载模型）")
E = QM.Qwen35PromptEnhancer

def trial(name, model_name, custom):
    try:
        r = E._resolve_path(E.__new__(E), model_name, custom)
        print(f"   {name:52s} -> {r}")
    except Exception as e:
        print(f"   {name:52s} -> 抛错 {type(e).__name__}: {e}")

first = next(iter(mapping.values()), None)
trial("空自定义 + 有效名字", next(iter(mapping), "<none>"), "")
trial("空自定义 + 名字也不在列表", "不存在的模型  [X]", "")
trial("自定义=不存在的路径", "不存在的模型  [X]", r"Z:\nope\Qwen3.5-9B")
trial("自定义=存在的目录但没 config.json", "不存在的模型  [X]", r"E:\AI\LLM")
trial("自定义=文件(如 .gguf)", "不存在的模型  [X]",
      r"E:\AI\ComfyUI-aki-v3\ComfyUI\models\LLM\Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf")
if first:
    trial("自定义=有效 HF 目录", "不存在的模型  [X]", first)
    trial("自定义=有效 HF 目录 + 末尾反斜杠", "不存在的模型  [X]", first + "\\")
    trial("自定义=有效 HF 目录 + 两边带引号", "不存在的模型  [X]", '"' + first + '"')
    trial("自定义=有效 HF 目录 + 中文引号", "不存在的模型  [X]", "\u201c" + first + "\u201d")

print()
print("5) 环境变量 / ComfyUI 额外路径支持？")
for k in ("QWEN35_MODEL_DIRS", "QWEN35_HF_DIRS", "COMFYUI_MODEL_DIRS"):
    print(f"   {k} = {os.environ.get(k)!r}")
print("   extra_model_paths.yaml 存在？",
      os.path.isfile(r"E:\AI\ComfyUI-aki-v3\ComfyUI\extra_model_paths.yaml"))
