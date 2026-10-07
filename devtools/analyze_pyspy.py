# -*- coding: utf-8 -*-
"""把 py-spy 的 speedscope 输出聚合成人看的排行榜。

**栈方向（实测确认）**：`samples[i]` = [root ... leaf]，即 `s[0]` 是线程入口、
`s[-1]` 是当前正在执行的帧。所以：
  * `--leaf`（独占时间）用 `s[-1]`
  * inclusive 统计（关键子串那一节）遍历整条栈，与方向无关

⚠ 只把它当**线索**。py-spy 在 Windows 上 `--native` 采样会出现明显的线程偏斜
（本项目实测：把一个 CPU 只占 1~5% 的线程报成 52.7% 的样本）。
凡涉及"谁的 CPU"这类归属，一律以 `probe_thread_cpu.py` 的每线程 CPU 计数为准。

用法：
    python analyze_pyspy.py _pyspy_native.json [--top 40] [--leaf] [--thread 正则]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=35)
    ap.add_argument("--thread", default=None, help="只看名字匹配该正则的线程")
    ap.add_argument("--leaf", action="store_true", help="按最内层帧(独占时间)聚合")
    a = ap.parse_args()

    d = load(a.path)
    frames = d["shared"]["frames"]

    def fname(i):
        fr = frames[i]
        n = fr.get("name") or "?"
        fl = (fr.get("file") or "").replace("\\", "/")
        if "/" in fl:
            fl = fl.rsplit("/", 1)[-1]
        return (n, fl)

    profiles = d.get("profiles") or []
    print(f"profile 数（=线程数）: {len(profiles)}")
    for p in profiles:
        print(f"  {p.get('name','?'):<40} samples={len(p.get('samples') or [])}")

    # 方向确认：leaf = s[-1]，root = s[0]（实测结论，见模块 docstring）
    leaf_dist = collections.Counter()
    root_dist = collections.Counter()
    for p in profiles:
        if a.thread and not re.search(a.thread, p.get("name") or ""):
            continue
        for s, w in zip(p.get("samples") or [], p.get("weights") or []):
            if not s:
                continue
            leaf_dist[fname(s[-1])] += w
            root_dist[fname(s[0])] += w

    print()
    print("栈方向确认 —— 最内层帧（s[-1]，真正的热点）分布 top5：")
    for (n, f), c in leaf_dist.most_common(5):
        print(f"   {c:9.1f}  {n}   ({f})")
    print("最外层帧（s[0]，应是线程入口）分布 top5：")
    for (n, f), c in root_dist.most_common(5):
        print(f"   {c:9.1f}  {n}   ({f})")

    # 正式聚合
    agg = collections.Counter()
    for p in profiles:
        if a.thread and not re.search(a.thread, p.get("name") or ""):
            continue
        for s, w in zip(p.get("samples") or [], p.get("weights") or []):
            if not s:
                continue
            key = fname(s[-1] if a.leaf else s[0])
            agg[key] += w
    tot = sum(agg.values()) or 1.0
    print()
    print("=" * 100)
    print(f"按{'最内层帧(独占)' if a.leaf else '栈底第0帧'}聚合  top {a.top}"
          f"   总权重 {tot:.0f}")
    print("=" * 100)
    print(f"{'权重':>10}{'占比':>8}  帧")
    for (n, f), c in agg.most_common(a.top):
        print(f"{c:10.1f}{c/tot*100:7.1f}%  {n}   ({f})")

    # 指定关键字的 inclusive 占比
    keys = ["cudaLaunch", "LaunchKernel", "cudaMalloc", "cudaFree", "Malloc",
            "free", "alloc", "malloc", "memcpy", "Memcpy", "conv3d", "convolution",
            "linear", "matmul", "mm(", "bmm", "attention", "sdpa", "flash",
            "synchronize", "empty_cache", "nvml", "aimdo", "emptyCache",
            "aten", "torch", "qwen3_5", "gated_delta", "recurrent", "chunk",
            "get_free_memory", "throw_exception", "ProgressBar", "send_sync"]
    print()
    print("=" * 100)
    print("关键子串的 inclusive 命中（含该帧的样本占比）")
    print("=" * 100)
    inc = collections.Counter()
    for p in profiles:
        for s, w in zip(p.get("samples") or [], p.get("weights") or []):
            seen = set()
            for i in s:
                n, f = fname(i)
                tag = f"{n} ({f})"
                if tag not in seen:
                    seen.add(tag)
                    inc[tag] += w
    for k in keys:
        hits = [(tag, c) for tag, c in inc.items() if k.lower() in tag.lower()]
        if not hits:
            continue
        hits.sort(key=lambda x: -x[1])
        tag, c = hits[0]
        print(f"  {k:<20} {c/tot*100:6.1f}%   {tag}")
        for tag2, c2 in hits[1:4]:
            print(f"  {'':<20} {c2/tot*100:6.1f}%   {tag2}")


if __name__ == "__main__":
    sys.exit(main())
