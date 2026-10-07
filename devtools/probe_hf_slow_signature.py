# -*- coding: utf-8 -*-
"""ComfyUI 内 HF 打标从 22 tok/s 掉到 8 tok/s —— 抓「资源指纹」把假设一次切开。

已确证（不要再重复验证）：
    · 同一进程、同一次运行、7 张图：先 44ms/token，中途翻到 130ms/token，之后不再恢复
      （prev.log 21:47 那轮：前 5 张 22.4 tok/s，第 6/7 张 8.7/7.6）
    · 反方向的翻转也发生过（prev2.log 21:03：前 2 张 7.6，之后 21~22.6）
    · ComfyUI 内 Qwen3-VL-8B（纯注意力）同样 8.0 tok/s → 与"线性注意力"无关
    · 独立进程同代码同模型 = 20.5 tok/s（native）/ 19.4（cudaMallocAsync）→ 分配器只值 5%
    · GGUF（llama-server 独立进程）在 ComfyUI 里 33.9~37.1 tok/s，完全不受影响
    · 当前机器没有别的进程抢 CPU；28 逻辑核，内存 40 GiB 可用

只在 ComfyUI 进程里存在、独立进程没有的东西：
    comfy_aimdo 0.5.5（cuda-detour.c 装 6 个 CUDA hook, NVML pressure）
    --async-offload（默认 2 streams） / --disable-pinned-memory
    --fast-disk（disk-backed dynamic loading and offload） / --disable-smart-memory
    --cuda-malloc（本机 argv 里确实有）
    main.py: comfy.memory_management.aimdo_enabled = True

本探针采集（1s 一次）：
    nvidia-smi : GPU% / 显存 used / SM 时钟 / 显存时钟 / P-State / 功耗 / PCIe 链路
    psutil     : ComfyUI 进程 CPU% / WorkingSet / 页文件提交量 / 页错误数
                 系统磁盘读-写字节增量 / 内存可用 / 页文件使用
    pynvml     : PCIe 吞吐（rx/tx KB/s，Windows 上不一定支持，能读就读）

判据：
    显存 used 明显 < 权重（17.5 GiB 主干 + 0.87 GiB 视觉塔）
        -> 权重被换出，问题在显存管理（aimdo / DynamicVRAM）
    解码期间磁盘读持续几十~几百 MB/s  -> 磁盘兜底在跑
    解码期间 PCIe 吞吐持续 GB/s 级     -> 权重每 token 走 PCIe 重搬
    SM 时钟掉档                      -> GPU 降频（与 CPU 侧无关）
    显存 used 正常 + 时钟正常 + GPU% 低 -> CPU 发射侧受限（要看是谁在耗 CPU）

用法（用 ComfyUI 自带的 python，因为它有 psutil）：
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_hf_slow_signature.py
    E:\\AI\\ComfyUI-aki-v3\\python\\python.exe probe_hf_slow_signature.py --images 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import psutil

SRC = r"J:\资料\训练数据集\二次元\漆原new\H3-261004\test"
ALL_NAMES = [
    "CellWorks39.jpg", "CellWorks42.jpg", "Qwen_image_2.1_00026.png",
    "Qwen_image_2.1_00027.png", "Qwen_image_2.1_00028.png",
]
API = "http://127.0.0.1:8188"
MODEL = "huihui-ai_Huihui-Qwen3.5-9B-abliterated  [Qwen3_5ForConditionalGeneration]"

WEIGHTS_GIB = 17.5 + 0.85          # 主干 + 视觉塔（日志实测）
SMI_FIELDS = ("utilization.gpu,memory.used,clocks.sm,clocks.mem,pstate,"
              "power.draw,pcie.link.gen.current,pcie.link.width.current")


# --------------------------------------------------------------------------
def get_json(path, timeout=30):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(path, obj, timeout=60):
    req = urllib.request.Request(
        API + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def smi():
    """返回 (gpu%, mem_used_MB, sm_mhz, mem_mhz, pstate, watt, pcie_gen, pcie_w)"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_FIELDS}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 8:
            return None
        g = lambda i: float(parts[i]) if parts[i] not in ("N/A", "[N/A]") else -1.0
        return dict(gpu=g(0), mem_mb=g(1), sm=g(2), memclk=g(3),
                    pstate=parts[4], watt=g(5), pcie_gen=g(6), pcie_w=g(7))
    except Exception:
        return None


def try_pynvml():
    try:
        import pynvml                                   # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml, h
    except Exception:
        return None, None


def pcie_throughput(pynvml, handle):
    """KB/s，Windows 上常抛 NotSupported。"""
    if pynvml is None:
        return None
    try:
        rx = pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        tx = pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        return float(rx), float(tx)
    except Exception:
        return None


def find_comfyui():
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        cl = p.info["cmdline"] or []
        s = " ".join(str(a) for a in cl)
        if "main.py" in s and "ComfyUI" in s:
            try:
                return psutil.Process(p.info["pid"])
            except Exception:
                pass
    return None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    names = ALL_NAMES[:max(1, args.images)]
    tmp = tempfile.mkdtemp(prefix="qwen35_sig_")
    for n in names:
        shutil.copyfile(os.path.join(SRC, n), os.path.join(tmp, n))

    print("=" * 108)
    print("ComfyUI 内 HF 打标 —— 资源指纹探针")
    print("=" * 108)
    st = get_json("/system_stats")
    print(f"ComfyUI argv    : {st['system']['argv']}")
    print(f"版本            : {st['system'].get('comfyui_version')}"
          f"   python {st['system'].get('python_version','')[:22]}")
    for dv in st.get("devices", []):
        print(f"设备            : {dv.get('name')}  "
              f"显存 {dv.get('vram_free',0)/2**30:.2f} GiB 可用 / "
              f"{dv.get('vram_total',0)/2**30:.2f} GiB")
    print(f"临时目录        : {tmp}")
    print(f"图片            : {len(names)} 张   {names}")
    print(f"模型权重参考    : 主干 {WEIGHTS_GIB-0.85:.1f} GiB + 视觉塔 0.85 GiB"
          f" = {WEIGHTS_GIB:.2f} GiB（若显存 used 明显低于此值即为换出）")

    comfy = find_comfyui()
    print(f"ComfyUI 进程    : PID {comfy.pid if comfy else '未找到'}")
    if comfy is None:
        print("!! 没找到进程，后面的进程级指标会是 0")
    print()

    pynvml, h = try_pynvml()
    print(f"pynvml          : {'可用（能读 PCIe 吞吐）' if pynvml else '不可用/不支持'}")
    s0 = smi()
    if s0:
        print(f"静止基线        : GPU {s0['gpu']:.0f}%  显存 {s0['mem_mb']/1024:.2f} GiB"
              f"  SM {s0['sm']:.0f}MHz  {s0['pstate']}  {s0['watt']:.0f}W"
              f"  PCIe gen{s0['pcie_gen']:.0f} x{s0['pcie_w']:.0f}")
    print()

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
    pid = post_json("/prompt", {"prompt": {"1": {
        "class_type": "Qwen35BatchImageTagger", "inputs": inputs}}})["prompt_id"]
    print(f"已提交 prompt_id = {pid}")
    print()

    hdr = (f"{'t':>5}{'GPU%':>6}{'显存GiB':>9}{'SM MHz':>8}{'PState':>8}{'W':>6}"
           f"{'PCIe':>8}{'PCIeRx':>10}"
           f"{'进程CPU%':>10}{'WS GiB':>8}{'页文件GiB':>10}{'页错误/s':>10}"
           f"{'磁盘读MB/s':>11}{'磁盘写MB/s':>11}{'内存可用':>9}")
    print(hdr)
    print("-" * len(hdr))

    if comfy:
        comfy.cpu_percent(None)
        try:
            mi = comfy.memory_info()
            last_pf = mi.num_page_faults
        except Exception:
            mi, last_pf = None, 0
    else:
        mi, last_pf = None, 0
    d0 = psutil.disk_io_counters()
    last_d = d0
    t0 = time.perf_counter()
    last = t0
    peak = dict(gpu=0.0, cpu=0.0, mem_mb=0.0, diskr=0.0, pcierx=0.0, watt=0.0)
    min_sm = 1e9
    rows = 0
    done = False
    while time.perf_counter() - t0 < 260:
        time.sleep(args.interval)
        now = time.perf_counter()
        dt = now - last
        last = now
        el = now - t0

        s = smi() or {}
        u = s.get("gpu", -1.0)
        mem_mb = s.get("mem_mb", -1.0)
        sm = s.get("sm", -1.0)
        if sm > 0:
            min_sm = min(min_sm, sm)

        cpu = 0.0
        ws_gib = pf_gib = pfps = 0.0
        if comfy:
            try:
                cpu = comfy.cpu_percent(None)
                mi = comfy.memory_info()
                ws_gib = mi.rss / 2**30
                pf_gib = getattr(mi, "pagefile", 0) / 2**30
                if dt > 0:
                    pfps = (mi.num_page_faults - last_pf) / dt
                last_pf = mi.num_page_faults
            except Exception:
                pass

        d1 = psutil.disk_io_counters()
        dr = (d1.read_bytes - last_d.read_bytes) / dt / 2**20 if dt > 0 else 0.0
        dw = (d1.write_bytes - last_d.write_bytes) / dt / 2**20 if dt > 0 else 0.0
        last_d = d1

        pc = pcie_throughput(pynvml, h)
        pc_txt = f"gen{s.get('pcie_gen',-1):.0f}x{s.get('pcie_w',-1):.0f}"
        pc_rx = f"{pc[0]/1024:.1f}MB/s" if pc else "n/a"

        avail = psutil.virtual_memory().available / 2**30

        peak["gpu"] = max(peak["gpu"], u)
        peak["cpu"] = max(peak["cpu"], cpu)
        peak["mem_mb"] = max(peak["mem_mb"], mem_mb)
        peak["diskr"] = max(peak["diskr"], dr)
        peak["watt"] = max(peak["watt"], s.get("watt", 0.0))
        if pc:
            peak["pcierx"] = max(peak["pcierx"], pc[0])

        print(f"{el:>5.1f}{u:>6.0f}{mem_mb/1024:>9.2f}{sm:>8.0f}"
              f"{s.get('pstate','?'):>8}{s.get('watt',-1):>6.0f}{pc_txt:>8}{pc_rx:>10}"
              f"{cpu:>10.1f}{ws_gib:>8.2f}{pf_gib:>10.2f}{pfps:>10.0f}"
              f"{dr:>11.1f}{dw:>11.1f}{avail:>9.1f}")
        rows += 1

        try:
            hist = get_json(f"/history/{pid}", timeout=10)
            if pid in hist:
                e = hist[pid]
                if e.get("outputs") or e.get("status", {}).get("status_str") == "error":
                    done = True
                    break
        except Exception:
            pass

    print()
    print("=" * 108)
    print(f"采样 {rows} 行   完成={done}")
    print(f"峰值 GPU {peak['gpu']:.0f}% | 峰值显存 {peak['mem_mb']/1024:.2f} GiB"
          f" | 最低 SM 时钟 {min_sm if min_sm < 1e9 else -1:.0f} MHz"
          f" | 峰值功耗 {peak['watt']:.0f} W")
    print(f"峰值 进程CPU {peak['cpu']:.0f}% | 峰值系统磁盘读 {peak['diskr']:.1f} MB/s"
          f" | 峰值 PCIeRx {peak['pcierx']/1024:.1f} MB/s")
    print()
    print("怎么读：")
    print(f"  显存峰值 ≈ {WEIGHTS_GIB:.2f} GiB  -> 权重全驻留，排除换出")
    print(f"  显存峰值 << {WEIGHTS_GIB:.2f} GiB  -> 权重被换出（aimdo / DynamicVRAM）")
    print("  磁盘读持续高                      -> 磁盘兜底在跑（--fast-disk）")
    print("  PCIeRx 持续 GB/s                  -> 权重每 token 走 PCIe 重搬")
    print("  SM 时钟掉档（如 <1500MHz）        -> GPU 降频")
    print("  显存/时钟都正常 + GPU% 低          -> CPU 发射侧受限")

    # 打印 ComfyUI 那轮自己的报告
    try:
        hist = get_json(f"/history/{pid}")
        if pid in hist:
            print()
            print("--- ComfyUI 节点返回的报告 ---")
            for _nid, o in (hist[pid].get("outputs") or {}).items():
                for t in (o.get("text") or []):
                    for ln in str(t).splitlines():
                        if ln.strip():
                            print("  | " + ln)
    except Exception:
        pass


if __name__ == "__main__":
    main()
