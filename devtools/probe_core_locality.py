# -*- coding: utf-8 -*-
"""HF 解码是单核受限 —— 那这个核是 P-core 还是 E-core？

背景（已确证）：
    · ComfyUI 内 HF 打标：解码线程 tid3076 独占 85~96% 的核，整进程也是 ~1 个核
    · 独立进程 同代码同模型：48.8 ms/token（20.5 tok/s）
      ComfyUI 内：125 ms/token（8.0 tok/s）
      -> **同一段代码每个 token 的 CPU 开销差 2.56 倍**
    · 本机 CPU = i7-14700KF：8 P-core（逻辑 0-15，带 HT）+ 12 E-core（逻辑 16-27）
      P-core 单线程性能约为 E-core 的 1.6~2.5 倍（Python 这种指针追逐型负载差得更狠）
    · Windows Thread Director 会主动把「后台的、持续吃 CPU 的」线程赶到 E-core

判据：
    解码期间繁忙的逻辑核落在 0-15   -> P-core，正常
    解码期间繁忙的逻辑核落在 16-27  -> E-core，**这就是 2.56 倍的那 2.5 倍**

做法：`psutil.cpu_times(percpu=True)` 的增量是系统级、无采样偏差的地面真值；
配合 `Process.cpu_num()` 高频抽样看进程当前落在哪个核。

用法：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_core_locality.py --mode comfy
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_core_locality.py --mode standalone
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import urllib.request

import psutil

CV = r"E:\AI\ComfyUI-aki-v3\ComfyUI"
NODE_DIR = os.path.join(CV, "custom_nodes", "ComfyUI-Qwen35-Enhancer")
SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
ALL = ["CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png"]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"

P_CORES = set(range(16))        # i7-14700KF：逻辑 0-15 = 8 个 P-core × HT
E_CORES = set(range(16, 28))    # 逻辑 16-27 = 12 个 E-core


class CoreSampler:
    """后台线程：累计每个逻辑核的 CPU 时间增量。"""

    def __init__(self, pid=None, interval=0.5):
        self.interval = interval
        self.stop = threading.Event()
        self.per_core = [0.0] * psutil.cpu_count(logical=True)
        self.cpu_num_hist = collections.Counter()
        self.proc = None
        if pid:
            try:
                self.proc = psutil.Process(pid)
            except Exception:
                self.proc = None
        self._t = None

    def _run(self):
        prev = psutil.cpu_times(percpu=True)
        t_prev = time.perf_counter()
        while not self.stop.is_set():
            time.sleep(self.interval)
            cur = psutil.cpu_times(percpu=True)
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            # 忙时 = 墙钟 - idle 增量（只算一次，别和 user+system 重复计）
            for i in range(len(self.per_core)):
                idle_delta = cur[i].idle - prev[i].idle
                self.per_core[i] += max(0.0, dt - idle_delta)
            prev = cur
            if self.proc is not None:
                for _ in range(4):
                    try:
                        self.cpu_num_hist[self.proc.cpu_num()] += 1
                    except Exception:
                        break
                    time.sleep(0.02)

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def report(self, wall):
        self.stop.set()
        if self._t:
            self._t.join(timeout=3)
        total = sum(self.per_core) or 1.0
        p = sum(c for i, c in enumerate(self.per_core) if i in P_CORES)
        e = sum(c for i, c in enumerate(self.per_core) if i in E_CORES)
        print()
        print("=" * 92)
        print(f"观察窗口 {wall:.1f}s   （psutil.cpu_times(percpu=True) 增量，系统级地面真值）")
        print("=" * 92)
        print(f"  系统总 CPU 时间      : {total:8.1f} s")
        print(f"  P-core (逻辑 0-15)   : {p:8.1f} s   {p/total*100:5.1f}%")
        print(f"  E-core (逻辑 16-27)  : {e:8.1f} s   {e/total*100:5.1f}%")
        print()
        print("  最忙的 8 个逻辑核：")
        rank = sorted(enumerate(self.per_core), key=lambda x: -x[1])[:8]
        for i, c in rank:
            kind = "P" if i in P_CORES else "E"
            bar = "#" * int(c / max(self.per_core) * 40) if max(self.per_core) else ""
            print(f"    cpu{i:<3}({kind}) {c:8.1f} s   {bar}")
        if self.cpu_num_hist:
            print()
            print("  Process.cpu_num() 抽样直方图（进程当前落在哪个逻辑核）：")
            for i, n in self.cpu_num_hist.most_common(8):
                kind = "P" if i in P_CORES else "E"
                print(f"    cpu{i:<3}({kind})  {n} 次")
        print()
        print("  判读：最忙核在 0-15 = P-core（正常）；在 16-27 = E-core（就是它慢 2.5 倍）")


# ----------------------------------------------------------------------
def build_stub():
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(CV, "models")
    fp.get_filename_list = lambda *a, **k: []
    fp.get_input_directory = lambda: os.environ.get("TEMP", ".")
    fp.get_output_directory = lambda: os.environ.get("TEMP", ".")
    sys.modules["folder_paths"] = fp

    class _PB:
        def __init__(self, *a, **k):
            pass

        def update_absolute(self, *a, **k):
            pass

    cu = types.ModuleType("comfy.utils")
    cu.ProgressBar = _PB
    comfy = types.ModuleType("comfy")
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = cu

    class _IPE(BaseException):
        pass

    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda *a, **k: None
    mm.free_memory = lambda *a, **k: None
    mm.soft_empty_cache = lambda *a, **k: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.InterruptProcessingException = _IPE
    comfy.model_management = mm
    sys.modules["comfy.model_management"] = mm


def stage(n=3):
    tmp = tempfile.mkdtemp(prefix="qwen35_core_")
    for x in ALL[:n]:
        shutil.copyfile(os.path.join(SRC, x), os.path.join(tmp, x))
    return tmp


def run_comfy(n=3):
    tmp = stage(n)
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
        "custom_model_path": "", "show_progress": True, "progress_interval": 2.0,
        "bilingual": "off", "max_output_chars": 0, "caption_mode": "off",
        "desc_length": "extra_long", "desc_words": "", "bilingual_sync": "auto",
        "verify": "off",
    }
    pid = None
    for p in psutil.process_iter(["pid", "cmdline"]):
        s = " ".join(str(a) for a in (p.info["cmdline"] or []))
        if "main.py" in s and "ComfyUI" in s:
            pid = p.info["pid"]
            break
    print(f"ComfyUI PID {pid}")
    req = urllib.request.Request(
        API + "/prompt",
        data=json.dumps({"prompt": {"1": {
            "class_type": "Qwen35BatchImageTagger", "inputs": inputs}}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        jid = json.loads(r.read())["prompt_id"]
    print(f"已提交 pid={jid}，等 18s 让模型加载完再开始采样…")
    time.sleep(18)
    s = CoreSampler(pid=pid)
    s.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 240:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{API}/history/{jid}", timeout=10) as r:
                h = json.loads(r.read())
            if jid in h and (h[jid].get("outputs")
                             or h[jid].get("status", {}).get("status_str") == "error"):
                break
        except Exception:
            pass
    wall = time.perf_counter() - t0
    s.report(wall)


def run_standalone(n=3):
    build_stub()
    sys.path.insert(0, NODE_DIR)
    import nodes as QM
    tmp = stage(n)
    print(f"独立进程 PID {os.getpid()}（自带 stub，不碰 ComfyUI）")
    print("先跑一遍预热 + 加载模型（约 20s），再开始采样…")
    node = QM.Qwen35BatchImageTagger()
    kw = dict(model_name="huihui-ai_Huihui-Qwen3.5-9B-abliterated",
              folder_path=tmp, system_preset="character", system_prompt="",
              user_prompt=QM.DEFAULT_TAGGER_USER_PROMPT, quantization="none",
              attention="auto", enable_thinking=False, overwrite="overwrite",
              output_format="raw", max_new_tokens=784, temperature=0.2, seed=42,
              max_image_side=1536, keep_model_loaded=True,
              unload_other_models=True, show_progress=False, bilingual="off",
              caption_mode="off", desc_length="extra_long", verify="off")
    t0 = time.perf_counter()
    node.tag_folder(**kw)
    print(f"  预热/加载完成 {time.perf_counter()-t0:.1f}s（模型已常驻）")
    time.sleep(1)
    print("开始采样（模型已常驻，纯解码）…")
    s = CoreSampler(pid=os.getpid())
    s.start()
    t0 = time.perf_counter()
    r = node.tag_folder(**kw)
    wall = time.perf_counter() - t0
    rep = r["result"][0]
    for ln in rep.splitlines():
        if any(k in ln for k in ("打标耗时", "输出 token", "合计")):
            print("  | " + ln.strip())
    s.report(wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["comfy", "standalone"], default="comfy")
    ap.add_argument("--images", type=int, default=3)
    a = ap.parse_args()
    print(f"CPU: {psutil.cpu_count(logical=True)} 逻辑 / "
          f"{psutil.cpu_count(logical=False)} 物理   "
          f"P-core=逻辑0-15  E-core=逻辑16-27")
    print()
    if a.mode == "comfy":
        run_comfy(a.images)
    else:
        run_standalone(a.images)


if __name__ == "__main__":
    main()
