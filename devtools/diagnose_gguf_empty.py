# -*- coding: utf-8 -*-
"""诊断：GGUF 后端「输出为空」到底怎么来的。

复现 ComfyUI 里那次「优化已有打标 → 7 张全空」。嫌疑链条：

    enable_thinking=False 没有被 GGUF 里存的 chat template 认下
      → 模型照样按混合思考模式产出 think 块
      → 启动参数里的 --reasoning-format deepseek 把 think 分流进
        reasoning_content
      → content 只剩空串
      → 节点侧 strip_thinking("") 仍然为空 → RuntimeError: 输出为空

关键是要把「模板认不认」「content/reasoning 各多长」「finish_reason 是
length 还是 stop」三样都量出来，而不是靠猜。

做法：按日志里那轮的完全相同参数起一个 llama-server，先拉 /props 把
chat_template 原文取回来（直接 grep enable_thinking 是否出现在模板里），
再发四种组合对比：

    A 纯文本    + enable_thinking=False
    B 带图      + enable_thinking=False
    C 带图+初稿 + enable_thinking=False     <- 复现现场
    D 带图+初稿 + 根本不带 chat_template_kwargs  <- 对照

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_gguf_empty.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time

NODE_DIR = r"E:\AI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen35-Enhancer"
sys.path.insert(0, NODE_DIR)

import gguf_backend as gb          # noqa: E402
import requests                    # noqa: E402

IMG = r"E:\AI\ComfyUI-aki-v3\ComfyUI\output\Qwen_image_2.1_00249.png"

# 与 comfyui.log 20:10:17 那轮的启动参数逐项对齐
CTX = 8192
MAX_TOKENS = 1024

SYS = ("You are an image captioning assistant. Describe the character in the "
       "image in detail: hair style and colour, eye colour, expression, "
       "clothing, accessories, pose and background.")

USER_PLAIN = "Describe this image."

USER_REFINE = (
    "Here is an existing caption for this image. Rewrite it so it is more "
    "accurate and more detailed, keeping the same structure.\n\n"
    "EXISTING CAPTION:\n"
    "a girl with long pink hair and red eyes, wearing a white ruffled top, "
    "standing in a city street at night"
)


def post(port, payload, label):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=(20, 600))
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        print(f"  [{label}] HTTP {r.status_code}: {r.text[:300]}")
        return
    d = r.json()
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = (msg.get("content") or "")
    reason = (msg.get("reasoning_content") or "")
    usage = d.get("usage") or {}
    print(f"  [{label}] {dt:.1f}s  finish={ch.get('finish_reason')!r}")
    print(f"      completion_tokens={usage.get('completion_tokens')}"
          f"  prompt_tokens={usage.get('prompt_tokens')}")
    print(f"      len(content)={len(content)}  len(reasoning_content)={len(reason)}")
    print(f"      content[:180]    = {content[:180]!r}")
    print(f"      reasoning[:180]  = {reason[:180]!r}")
    print()


def main():
    exe = gb.find_llama_server()
    mains, projs = gb.list_gguf()
    if not exe or not mains:
        print("!! 找不到 llama-server 或 gguf")
        return 2
    model = list(mains.keys())[0]
    mmproj = list(projs.keys())[0] if projs else ""
    print(f"server : {exe}")
    print(f"model  : {model}")
    print(f"mmproj : {mmproj}\n")

    cfg = {
        "server_exe": exe,
        "model": mains[model],
        "mmproj": projs[mmproj] if mmproj else "",
        "context_size": CTX,
        "parallel": 1,
        "kv_cache_type": "q8_0",
        "flash_attn": True,
        "batch": 2048,
        "ubatch": 512,
        "reasoning_format": "deepseek",
        "n_gpu_layers": -1,
        "extra_args": "",
    }
    srv = gb.acquire(cfg)
    print("启动 llama-server…", flush=True)
    t0 = time.perf_counter()
    srv.ensure_ready(on_status=lambda m: print("   " + m, flush=True))
    print(f"就绪，用时 {time.perf_counter() - t0:.1f}s，端口 {srv.port}\n")

    # ---- ① 把 chat template 原文抠出来，看它认不认 enable_thinking ----
    print("=" * 74)
    print("① chat template 自查")
    print("=" * 74)
    try:
        props = requests.get(f"http://127.0.0.1:{srv.port}/props", timeout=30).json()
        tmpl = props.get("chat_template") or ""
        if not tmpl:
            # 有些版本把它放在 default_generation_settings 之外
            tmpl = json.dumps(props)[:2000]
        print(f"  template 长度: {len(tmpl)}")
        print(f"  含 'enable_thinking'      : {'enable_thinking' in tmpl}")
        print(f"  含 'thinking'             : {'thinking' in tmpl}")
        print(f"  含 '<|im_start|>'         : {'<|im_start|>' in tmpl}")
        i = tmpl.find("enable_thinking")
        if i >= 0:
            print("  --- 出现位置上下文 ---")
            print("  " + tmpl[max(0, i - 260):i + 260].replace("\n", "\n  "))
    except Exception as e:
        print(f"  !! 取 /props 失败：{e}")
    print()

    # ---- ② 四种组合 ----
    b64 = ""
    with open(IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    img_part = {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}}

    def payload(user, with_img, thinking_kwarg):
        content = ([{"type": "text", "text": user}] + [img_part]) if with_img else user
        p = {
            "model": "local",
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.05,
            "stream": False,
            "cache_prompt": True,
            "seed": 1234,
        }
        if thinking_kwarg:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        return p

    print("=" * 74)
    print("② 四种组合对比")
    print("=" * 74)
    post(srv.port, payload(USER_PLAIN, False, True), "A 纯文本 +no_think")
    post(srv.port, payload(USER_PLAIN, True, True), "B 带图   +no_think")
    post(srv.port, payload(USER_REFINE, True, True), "C 带图初稿+no_think")
    post(srv.port, payload(USER_REFINE, True, False), "D 带图初稿(无kwarg)")

    gb.release(force=True)
    print("server 已关闭")
    return 0


if __name__ == "__main__":
    sys.exit(main())
