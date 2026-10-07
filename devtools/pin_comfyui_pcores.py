# -*- coding: utf-8 -*-
"""把 ComfyUI 钉到 P-core 上 —— 解决「HF 打标 8 tok/s」的根因。

## 结论（2026-10-07 真机 A/B/C 实测，i7-14700KF，8 P-core + 12 E-core）

| 亲和性 | 解码 tok/s | 相对 |
|---|---|---|
| 不限核（28 逻辑核，原始状态） | **7.1** | 1.00x |
| 只给 P-core（逻辑 0-15） | **18.8** | **2.66x** |
| 只给 E-core（逻辑 16-27） | 7.4 | 1.05x |
| 不限核 + 进程优先级 HIGH | 7.9 | 1.11x |

「不限核」与「只给 E-core」几乎一样 → **Windows 平时就把这个线程丢在 E-core 上**。
本机单核 P/E 性能比：Python 紧循环 2.05×、小算子派发 3.40×、小分配 4.39× ——
HF 解码正是「单核 + 逐个小算子派发」型负载，2.66× 完全对得上。

## 为什么只有 HF 路径中招

HF 打标是**进程内、单核、Python 逐算子发射**的（实测 3365 个 CUDA kernel/输出 token），
所以它跑在 P-core 还是 E-core 上差别巨大。
GGUF 后端是**独立进程 llama-server**，吞吐由 GPU 决定 → 不受影响（同机 33.9~37.1 tok/s）。

## 用法

    # 只看诊断（默认，不改任何东西）
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe pin_comfyui_pcores.py

    # 立刻钉到 P-core
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe pin_comfyui_pcores.py --set

    # 恢复原始亲和性
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe pin_comfyui_pcores.py --restore

    # 常驻守着：ComfyUI 每次重启后自动重新钉上（Ctrl-C 退出）
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe pin_comfyui_pcores.py --watch

## 代价

只能让 ComfyUI 用 16 个逻辑核（8 个物理 P-core）而不是 28 个。
出图这类 GPU 主导的任务几乎无感；真需要 28 核的纯 CPU 任务会慢一点。
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import psutil

K32 = ctypes.WinDLL("kernel32", use_last_error=True)

RelationProcessorCore = 0


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t),          # KAFFINITY = ULONG_PTR
                ("Group", wintypes.WORD),
                ("Reserved", wintypes.WORD * 3)]


class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    _fields_ = [("Flags", ctypes.c_ubyte),
                ("EfficiencyClass", ctypes.c_ubyte),
                ("Reserved", ctypes.c_ubyte * 20),
                ("GroupCount", wintypes.WORD),
                ("GroupMask", GROUP_AFFINITY * 1)]


class SLPI_HEADER(ctypes.Structure):
    """每条记录的头 8 字节。只读它来步进，避免越界读整条记录。"""
    _fields_ = [("Relationship", wintypes.DWORD),
                ("Size", wintypes.DWORD)]


def topology():
    """返回 (p_cores, e_cores, detail)

    用 Windows 官方 `GetLogicalProcessorInformationEx(RelationProcessorCore)` 读
    每个**物理核**的 EfficiencyClass。Intel 混合架构上该值大的那颗性能更高
    （Windows 的约定：值越大越快），即 P-core。
    """
    buf_len = wintypes.DWORD(0)
    K32.GetLogicalProcessorInformationEx(RelationProcessorCore, None,
                                         ctypes.byref(buf_len))
    if buf_len.value == 0:
        return None, None, f"取缓冲区长度失败 err={ctypes.get_last_error()}"
    buf = ctypes.create_string_buffer(buf_len.value)
    if not K32.GetLogicalProcessorInformationEx(RelationProcessorCore, buf,
                                                ctypes.byref(buf_len)):
        return None, None, f"调用失败 err={ctypes.get_last_error()}"

    total = buf_len.value
    per_class = {}
    off = 0
    gm_off = PROCESSOR_RELATIONSHIP.GroupMask.offset      # = 24
    while off + ctypes.sizeof(SLPI_HEADER) <= total:
        hdr = SLPI_HEADER.from_buffer(buf, off)
        if hdr.Size == 0:
            break
        if hdr.Relationship == RelationProcessorCore and \
                off + 8 + ctypes.sizeof(PROCESSOR_RELATIONSHIP) <= total:
            pr = PROCESSOR_RELATIONSHIP.from_buffer(buf, off + 8)
            base = off + 8 + gm_off
            for g in range(pr.GroupCount):
                if base + (g + 1) * ctypes.sizeof(GROUP_AFFINITY) > total:
                    break
                ga = GROUP_AFFINITY.from_buffer(buf, base + g * 16)
                m = ga.Mask
                idx = 0
                while m:
                    if m & 1:
                        per_class.setdefault(pr.EfficiencyClass, []).append(idx)
                    m >>= 1
                    idx += 1
        off += hdr.Size

    if len(per_class) < 2:
        return None, None, (f"只读到 {len(per_class)} 种 EfficiencyClass "
                            f"{sorted(per_class)} —— 可能是非混合架构的 CPU")
    classes = sorted(per_class, reverse=True)              # 大 = 更快
    p = sorted(per_class[classes[0]])
    e = sorted(x for c in classes[1:] for x in per_class[c])
    detail = "  ".join(f"EfficiencyClass {c}: {len(per_class[c])} 个逻辑核"
                       for c in sorted(per_class))
    return p, e, detail


def find_comfyui():
    hits = []
    for pr in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            s = " ".join(str(a) for a in (pr.info["cmdline"] or []))
        except Exception:
            continue
        if "main.py" in s and "ComfyUI" in s:
            hits.append(psutil.Process(pr.info["pid"]))
    return hits


def sig(p):
    try:
        return (p.pid, p.create_time())
    except Exception:
        return None


def report(p_cores, e_cores, detail, procs):
    print("CPU 拓扑（来自 Windows GetLogicalProcessorInformationEx）")
    print(f"  逻辑核总数 : {psutil.cpu_count(logical=True)}"
          f"   物理核 {psutil.cpu_count(logical=False)}")
    print(f"  {detail}")
    print(f"  P-core 逻辑核（EfficiencyClass 最高）: {p_cores}")
    print(f"  E-core 逻辑核                       : {e_cores}")
    print()
    if not procs:
        print("ComfyUI 进程：未找到（先启动 ComfyUI 再 --set）")
        return
    for p in procs:
        try:
            aff = p.cpu_affinity()
        except Exception as ex:
            print(f"  PID {p.pid}: 读亲和性失败 {ex}")
            continue
        n_e = len([c for c in aff if c in set(e_cores or [])])
        n_p = len([c for c in aff if c in set(p_cores or [])])
        verdict = ("已钉在 P-core ✓" if n_e == 0 and n_p
                   else "**包含 E-core —— HF 路径会慢约 2.6 倍**")
        print(f"  PID {p.pid}  亲和性 {len(aff)} 个核 "
              f"(P {n_p} / E {n_e})  优先级 {p.nice()}   -> {verdict}")


def set_aff(p, p_cores):
    p.cpu_affinity(p_cores)
    return p.cpu_affinity()


def main():
    ap = argparse.ArgumentParser(description="把 ComfyUI 钉到 P-core")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--set", action="store_true", help="立刻钉到 P-core")
    g.add_argument("--restore", action="store_true", help="恢复全部逻辑核")
    g.add_argument("--watch", action="store_true", help="常驻：ComfyUI 重启后自动重钉")
    ap.add_argument("--interval", type=float, default=5.0, help="--watch 的轮询间隔秒")
    a = ap.parse_args()

    p_cores, e_cores, detail = topology()
    if not p_cores:
        print(f"!! 拿不到混合架构拓扑：{detail}")
        print("   （非 Intel 混合架构 / 老系统 / 只读到一类核，本工具不适用）")
        print("   退路：手动指定，例如 --set 后自己 psutil 设 [0..15]")
        return 2

    procs = find_comfyui()

    if not (a.set or a.restore or a.watch):
        report(p_cores, e_cores, detail, procs)
        print()
        print("建议：")
        print(f"  python {sys.argv[0]} --set      立刻钉到 P-core（HF 打标约快 2.6 倍）")
        print(f"  python {sys.argv[0]} --watch    常驻守着，ComfyUI 重启后自动重钉")
        return 0

    if a.restore:
        for p in procs:
            try:
                p.cpu_affinity(list(range(psutil.cpu_count(logical=True))))
                print(f"PID {p.pid} 已恢复：{len(p.cpu_affinity())} 个核")
            except Exception as ex:
                print(f"PID {p.pid} 恢复失败：{ex}")
        return 0

    if a.set:
        if not procs:
            print("!! 没找到 ComfyUI 进程")
            return 1
        for p in procs:
            try:
                got = set_aff(p, p_cores)
                print(f"PID {p.pid} 已钉到 P-core：{len(got)} 个核 {got}")
            except Exception as ex:
                print(f"PID {p.pid} 设置失败：{ex}")
        print()
        report(p_cores, e_cores, detail, find_comfyui())
        return 0

    # --watch
    print(f"监听中（每 {a.interval}s 一次）：ComfyUI 新进程出现就自动钉到 P-core")
    print(f"P-core 逻辑核 = {p_cores}")
    print("Ctrl-C 退出（退出时不会恢复，若要还原请跑 --restore）")
    seen = {}
    try:
        while True:
            for p in find_comfyui():
                k = sig(p)
                if k is None or k in seen:
                    continue
                try:
                    got = p.cpu_affinity()
                    n_e = len([c for c in got if c in set(e_cores)])
                    if n_e:
                        set_aff(p, p_cores)
                        print(f"[{time.strftime('%H:%M:%S')}] PID {p.pid} "
                              f"新进程 -> 已钉到 {len(p.cpu_affinity())} 个 P-core")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] PID {p.pid} "
                              f"已在 P-core 上，跳过")
                except Exception as ex:
                    print(f"[{time.strftime('%H:%M:%S')}] PID {p.pid} 处理失败：{ex}")
                seen[k] = True
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n退出监听（亲和性保持现状，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
