# -*- coding: utf-8 -*-
"""只负责“提交一个 HF 打标任务并等它跑完”，不采集任何指标。

用途：给 py-spy 提供一个稳定的采样窗口 —— 探针脚本自己每秒跑 nvidia-smi 会引入噪声，
所以把“提交”和“采样”分开。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe _submit_only.py --images 4
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe _submit_only.py --images 4 --keep
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import urllib.request

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
ALL = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png",
       "Qwen_image_2.1_00027.png", "Qwen_image_2.1_00028.png",
       "Qwen_image_2.1_00039.png", "Qwen_image_2.1_00044.png"]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="跑完不卸载模型（便于连续采样）")
    ap.add_argument("--tag", default="job")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix=f"qwen35_{a.tag}_")
    for n in ALL[:a.images]:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))

    inputs = {
        "model_name": MODEL, "folder_path": tmp,
        "system_preset": "character", "system_prompt": "",
        "user_prompt": "给这张图打标。", "quantization": "none", "attention": "auto",
        "enable_thinking": False, "recursive": False, "overwrite": "overwrite",
        "image_exts": ".png,.jpg,.jpeg,.webp,.bmp", "output_suffix": "",
        "output_encoding": "utf-8", "output_format": "raw",
        "max_new_tokens": 784, "temperature": 0.2, "seed": 42,
        "max_image_side": 1536, "limit": 0, "dry_run": False,
        "keep_model_loaded": bool(a.keep), "unload_other_models": True,
        "custom_model_path": "", "show_progress": True, "progress_interval": 2.0,
        "bilingual": "off", "max_output_chars": 0, "caption_mode": "off",
        "desc_length": "extra_long", "desc_words": "", "bilingual_sync": "auto",
        "verify": "off",
    }
    req = urllib.request.Request(
        API + "/prompt",
        data=json.dumps({"prompt": {"1": {
            "class_type": "Qwen35BatchImageTagger", "inputs": inputs}}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        pid = json.loads(r.read())["prompt_id"]
    print(f"SUBMITTED pid={pid}  imgs={a.images}  tmp={tmp}", flush=True)

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 600:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{API}/history/{pid}", timeout=10) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if pid in h:
            e = h[pid]
            if e.get("outputs") or e.get("status", {}).get("status_str") == "error":
                print(f"DONE in {time.perf_counter()-t0:.1f}s", flush=True)
                for _n, o in (e.get("outputs") or {}).items():
                    for t in (o.get("text") or []):
                        for ln in str(t).splitlines():
                            if "打标耗时" in ln or "合计" in ln or "输出 token" in ln:
                                print("  " + ln.strip(), flush=True)
                return
    print("TIMEOUT", flush=True)


if __name__ == "__main__":
    main()
