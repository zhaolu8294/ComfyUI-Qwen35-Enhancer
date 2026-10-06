# -*- coding: utf-8 -*-
"""把工作流里 Qwen35PromptEnhancer 的 attention 从 sdpa 改成 auto。

为什么必须改工作流而不只是改代码默认值：
  ComfyUI 的 COMBO 控件把**值**（字符串）存进 widgets_values，代码里的
  default 只影响新拖出来的节点。已保存的工作流会继续用存下来的 "sdpa"，
  装了 flash-attn 也不会走 flash 路径。

为什么用 auto 而不是直接写 flash_attention_2：
  auto 在装了合规 flash-attn 时选 flash_attention_2，没装就退回 sdpa。
  这样卸载 flash-attn 不会把已有工作流跑挂。

幂等：重复执行只改需要改的。原值不是 sdpa 且不是 auto 时不碰。
"""
import json
import os
import shutil
import sys

TARGETS = [
    "h3_i2v_qwen35.json",
    "h3_ref2v_multi_qwen35.json",
    "prompt_enhancer_qwen35_text.json",
    "prompt_enhancer_qwen35_image.json",
]

# examples/ 下的同名副本也要保持一致
EXAMPLES_DIR = (
    r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer\examples"
)

ATTN_IDX = 4          # widgets 顺序：model_name / system_prompt / user_prompt /
                      # quantization / **attention** / enable_thinking / ...
USER_NODE = "Qwen35PromptEnhancer"


def patch(path, dry=False):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    hits = []
    for node in data.get("nodes", []):
        if node.get("type") != USER_NODE:
            continue
        w = node.get("widgets_values") or []
        if len(w) <= ATTN_IDX:
            hits.append((node.get("id"), "<不足 %d 项，跳过>" % (ATTN_IDX + 1)))
            continue
        before = w[ATTN_IDX]
        if before == "auto":
            hits.append((node.get("id"), "已是 auto，跳过"))
            continue
        if before != "sdpa":
            hits.append((node.get("id"), "原值 %r 非 sdpa，跳过" % (before,)))
            continue
        w[ATTN_IDX] = "auto"
        node["widgets_values"] = w
        hits.append((node.get("id"), "sdpa -> auto"))
    if not dry:
        shutil.copy2(path, path + ".bak_attention")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return hits


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    changed = 0
    for name in TARGETS:
        for path in (os.path.join(base, name), os.path.join(EXAMPLES_DIR, name)):
            if not os.path.isfile(path):
                print("  缺失  %s" % path)
                continue
            hits = patch(path)
            tag = "  ".join("node %s: %s" % h for h in hits) if hits else "无目标节点"
            did = any("sdpa -> auto" in h[1] for h in hits)
            changed += 1 if did else 0
            print("%-6s %s\n        %s" % ("改动" if did else "跳过", path, tag))
    print("\n共改动 %d 个文件（备份后缀 .bak_attention）" % changed)


if __name__ == "__main__":
    sys.exit(main())
