# -*- coding: utf-8 -*-
"""因果验证：把 ComfyUI 进程钉到 P-core / E-core，看 tok/s 怎么变。

为什么做这个：HF 解码线程独占 1 个核却只跑出 8 tok/s，独立进程同一个核跑 20.5。
本机 CPU = i7-14700KF（8 P-core 逻辑 0-15 + 12 E-core 逻辑 16-27）。
如果「钉到 P-core 就变快、钉到 E-core 就变慢」成立，
那 2.56 倍就是**线程被调度到 E-core**造成的，与模型/代码/量化全都无关。

做法：`psutil.Process.cpu_affinity()`（Windows = SetProcessAffinityMask），
可逆、无需重启、不碰任何用户数据。跑完**一定**恢复原亲和性。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_core_pinning.py --images 2
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe diagnose_core_pinning.py --images 2 --restore-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

import psutil

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
ALL = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png"]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"

P_CORES = list(range(16))        # i7-14700KF：逻辑 0-15
E_CORES = list(range(16, 28))    # 逻辑 16-27


def find_comfyui():
    for p in psutil.process_iter(["pid", "cmdline"]):
        s = " ".join(str(a) for a in (p.info["cmdline"] or []))
        if "main.py" in s and "ComfyUI" in s:
            return psutil.Process(p.info["pid"])
    return None


def stage(n):
    tmp = tempfile.mkdtemp(prefix="qwen35_aff_")
    for x in ALL[:n]:
        shutil.copyfile(os.path.join(SRC, x), os.path.join(tmp, x))
    return tmp


def run_job(tmp, label):
    inputs = {
        "model_name": MODEL, "folder_path": tmp,
        "system_preset": "character", "system_prompt": "",
        "user_prompt": "给这张图打标。", "quantization": "none", "attention": "auto",
        "enable_thinking": False, "recursive": False, "overwrite": "overwrite",
        "image_exts": ".png,.jpg,.jpeg,.webp,.bmp", "output_suffix": "",
        "output_encoding": "utf-8", "output_format": "raw",
        "max_new_tokens": 784, "temperature": 0.2, "seed": 42,
        "max_image_side": 1536, "limit": 0, "dry_run": False,
        "keep_model_loaded": False, "unload_other_models": True,
        "custom_model_path": "", "show_progress": False, "progress_interval": 2.0,
        "bilingual": "off", "max_output_chars": 0, "caption_mode": "off",
        "desc_length": "extra_long", "desc_words": "", "bilingual_sync": "auto",
        "verify": "off",
    }
    print()
    print("=" * 90)
    print(f"{label}")
    print(f"  folder_path = {tmp}   ← 每轮换新目录，避免被 ComfyUI 的节点输出缓存命中")
    print("=" * 90)
    req = urllib.request.Request(
        API + "/prompt",
        data=json.dumps({"prompt": {"1": {
            "class_type": "Qwen35BatchImageTagger", "inputs": inputs}}}).encode(),
        headers={"Content-Type": "application/json"})
    before = _history_keys()
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        jid = json.loads(r.read())["prompt_id"]
    while time.perf_counter() - t0 < 420:
        time.sleep(3)
        try:
            with urllib.request.urlopen(f"{API}/history/{jid}", timeout=10) as r:
                h = json.loads(r.read())
        except Exception:
            continue
        if jid in before:                      # 绝不该发生，留个保险
            print("  !! 提交前该 id 就在 history 里了，数据不可信")
            return None, ""
        if jid in h and (h[jid].get("outputs")
                         or h[jid].get("status", {}).get("status_str") == "error"):
            wall = time.perf_counter() - t0
            rep_txt = ""
            for _n, o in (h[jid].get("outputs") or {}).items():
                for t in (o.get("text") or []):
                    rep_txt += str(t) + "\n"
            for ln in rep_txt.splitlines():
                if any(k in ln for k in ("打标耗时", "输出 token", "合计", "加载模型",
                                         "待处理", "没有需要处理")):
                    print("  | " + ln.strip())
            print(f"  墙钟总耗时 {wall:.1f}s")
            return wall, rep_txt
    print("  !! 超时")
    return None, ""


def _history_keys():
    try:
        with urllib.request.urlopen(f"{API}/history", timeout=10) as r:
            return set(json.loads(r.read()).keys())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--arms", default="all,p,e",
                    help="要测的亲和性组合，逗号分隔：all / p / e")
    ap.add_argument("--restore-only", action="store_true")
    a = ap.parse_args()

    proc = find_comfyui()
    if proc is None:
        print("!! 没找到 ComfyUI 进程")
        return
    orig = proc.cpu_affinity()
    orig_prio = proc.nice()
    print(f"ComfyUI PID {proc.pid}")
    print(f"原始 CPU 亲和性: {len(orig)} 个核 {orig}")
    print(f"本机 {psutil.cpu_count(logical=True)} 逻辑核，"
          f"P-core = 逻辑 0-15（{len(P_CORES)} 个），E-core = 逻辑 16-27（{len(E_CORES)} 个）")

    if a.restore_only:
        proc.cpu_affinity(orig)
        try:
            proc.nice(orig_prio)
        except Exception:
            pass
        print(f"已恢复为 {len(proc.cpu_affinity())} 个核，优先级 {proc.nice()}")
        return

    arms = {
        "all": ("A  不限核（原始状态：28 个逻辑核）", orig, None),
        "p": ("B  只给 P-core（逻辑 0-15）", P_CORES, None),
        "e": ("C  只给 E-core（逻辑 16-27）", E_CORES, None),
        "hi": ("D  不限核 + 进程优先级 HIGH", orig, "high"),
        "above": ("E  不限核 + 进程优先级 ABOVE_NORMAL", orig, "above"),
        "below": ("F  不限核 + 进程优先级 BELOW_NORMAL", orig, "below"),
    }
    prio_map = {
        "high": psutil.HIGH_PRIORITY_CLASS,
        "above": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
        "below": psutil.BELOW_NORMAL_PRIORITY_CLASS,
    }
    print(f"原始进程优先级: {orig_prio}")
    tmp = stage(a.images)
    print(f"首轮临时目录 {tmp}   每轮 {a.images} 张")

    results = []
    try:
        for n, key in enumerate(k.strip() for k in a.arms.split(",") if k.strip()):
            label, mask, prio = arms[key]
            # 每轮换新目录：ComfyUI 的节点输出缓存按 inputs 命中，
            # 不换目录的话第二轮会 0.02s 直接复用上一轮结果（踩过）。
            tmp = stage(a.images) if n else tmp
            try:
                proc.cpu_affinity(mask)
                if prio:
                    proc.nice(prio_map[prio])
                else:
                    proc.nice(orig_prio)
            except Exception as e:
                print(f"!! 设置失败 {key}: {e}")
                continue
            time.sleep(1)
            print(f"\n>>> 已设亲和性 = {len(proc.cpu_affinity())} 个核，"
                  f"优先级 = {proc.nice()}")
            wall, rep = run_job(tmp, label)
            tok = dec = None
            for ln in rep.splitlines():
                s = ln.strip()
                if s.startswith("输出 token"):
                    try:
                        tok = float(s.split(":")[1].split("（")[0].strip())
                    except Exception:
                        pass
                elif s.startswith("打标耗时"):
                    try:
                        dec = float(s.split(":")[1].strip().split("（")[0].strip().rstrip("s"))
                    except Exception:
                        pass
            results.append((label, wall, tok, dec))
    finally:
        proc.cpu_affinity(orig)
        try:
            proc.nice(orig_prio)
        except Exception:
            pass
        print()
        print(f"*** 已恢复原始 CPU 亲和性：{len(proc.cpu_affinity())} 个核，"
              f"优先级 {proc.nice()} ***")

    print()
    print("=" * 90)
    print("汇总")
    print("=" * 90)
    print(f"  {'亲和性':<34}{'墙钟s':>9}{'输出tok':>10}{'解码s':>9}{'tok/s':>9}")
    for label, wall, tok, dec in results:
        sp = f"{tok/dec:.1f}" if (tok and dec) else "-"
        print(f"  {label:<34}"
              f"{(f'{wall:.1f}' if wall else '-'):>9}"
              f"{(f'{tok:.0f}' if tok else '-'):>10}"
              f"{(f'{dec:.1f}' if dec else '-'):>9}{sp:>9}")
    if len(results) >= 2:
        base = next((r for r in results if r[0].startswith("A")), None)
        for r in results:
            if base and r[3] and base[3] and r is not base:
                b = base[2] / base[3]
                c = r[2] / r[3]
                print(f"  {r[0][:1]} 相对 A 的倍数: {c/b:.2f}x")


if __name__ == "__main__":
    main()
