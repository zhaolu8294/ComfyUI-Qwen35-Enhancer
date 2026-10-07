# -*- coding: utf-8 -*-
"""按线程打印「众数调用栈」—— 看清楚每个线程到底卡在哪。

**栈方向（2026-10-07 实测确认）**：py-spy 的 speedscope 输出里
`samples[i]` 是 **[最外层(root) ... 最内层(leaf)]**，
所以 `s[0]` 是线程入口（如 `_bootstrap`）、**`s[-1]` 才是当前正在执行的帧**。
本脚本的「独占时间」用 `s[-1]`；反过来取会把 `_bootstrap` 当成热点（踩过）。

用法：
    python pyspy_thread_stacks.py _pyspy_native.json
    python pyspy_thread_stacks.py _pyspy_native.json --only monitor
"""
from __future__ import annotations

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--only", default=None, help="只打印名字含该子串的线程")
    ap.add_argument("--depth", type=int, default=40)
    a = ap.parse_args()

    with open(a.path, "r", encoding="utf-8") as f:
        d = json.load(f)
    frames = d["shared"]["frames"]

    def fname(i):
        fr = frames[i]
        n = fr.get("name") or "?"
        fl = (fr.get("file") or "").replace("\\", "/")
        ln = fr.get("line")
        if "/" in fl:
            fl = fl.rsplit("/", 1)[-1]
        return f"{n}  ({fl}:{ln})"

    grand = 0
    for p in d.get("profiles") or []:
        nm = p.get("name") or "?"
        if a.only and a.only.lower() not in nm.lower():
            continue
        stacks = p.get("samples") or []
        weights = p.get("weights") or [1.0] * len(stacks)
        tot = sum(weights)
        grand += tot
        print("=" * 100)
        print(f"{nm}    样本权重合计 {tot:.1f}   不同栈 {len(stacks)}")
        print("=" * 100)

        # 独占时间 = 最内层帧（s[-1]）；s[0] 是 root，别搞反
        leaf = collections.Counter()
        for s, w in zip(stacks, weights):
            if s:
                leaf[fname(s[-1])] += w
        print("  -- 独占时间 top 8（最内层帧）--")
        for k, c in leaf.most_common(8):
            print(f"     {c:8.1f} {c/tot*100:6.1f}%   {k}")

        # 众数栈
        modal = collections.Counter()
        for s, w in zip(stacks, weights):
            modal[tuple(s)] += w
        print("  -- 众数调用栈 top 3（元素顺序 = root → leaf） --")
        for stack, c in modal.most_common(3):
            print(f"   * 权重 {c:.1f} ({c/tot*100:.1f}%)")
            for i in list(stack)[:a.depth]:
                print(f"        {fname(i)}")
        print()


if __name__ == "__main__":
    main()
