# -*- coding: utf-8 -*-
"""只读：核对示例工作流里 Qwen35 节点的 quantization / 关键参数现值。"""
import json
import os
import sys

EX = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer\examples"
KEYS = ("quantization", "attention_backend", "threshold", "unload_other_models",
        "keep_model_loaded", "output_format", "overwrite", "folder_path")


def main():
    for fname in sorted(os.listdir(EX)):
        if not fname.endswith(".json") or ".bak_" in fname:
            continue
        p = os.path.join(EX, fname)
        with open(p, "r", encoding="utf-8") as f:
            wf = json.load(f)
        print(f"\n=== {fname} ===")
        for n in wf.get("nodes", []):
            t = n.get("type", "")
            if not t.startswith("Qwen35"):
                continue
            wv = n.get("widgets_values", [])
            print(f"  [{n.get('id')}] {t}  widgets={len(wv)}")
            # 打印前若干项，帮助肉眼对齐
            print(f"      widgets_values[:10] = {wv[:10]!r}")


if __name__ == "__main__":
    main()
