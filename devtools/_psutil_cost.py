import time, statistics
import psutil

def bench(fn, n=30, label=""):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    print(f"{label:<26} n={n:<4} 中位 {statistics.median(ts):8.3f} ms   最小 {ts[0]:8.3f}   最大 {ts[-1]:9.3f}")
    return statistics.median(ts)

print("=== psutil 各类调用在本机的单次耗时 ===")
t_vm  = bench(psutil.virtual_memory,   label="virtual_memory()")
t_swp = bench(psutil.swap_memory,      label="swap_memory()")
t_cpu = bench(lambda: psutil.cpu_percent(None), label="cpu_percent(None)")
t_ci  = bench(psutil.cpu_times,        label="cpu_times()")
t_pm  = bench(psutil.process_iter,     label="process_iter()")
try:
    import ctypes
    class P(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("f", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in
                    ("TotalPhys","AvailPhys","TotalPageFile","AvailPageFile",
                     "TotalVirtual","AvailVirtual","AvailExtendedVirtual")]
    def gms():
        s = P(); s.cb = ctypes.sizeof(P)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    bench(gms, label="GlobalMemoryStatusEx (裸)")
    def gpi():
        # GetPerformanceInfo
        class PI(ctypes.Structure):
            _fields_ = [(n, ctypes.c_size_t) for n in
                        ("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal",
                         "PhysicalAvailable","SystemCache","KernelTotal","KernelPaged",
                         "KernelNonpaged","PageSize","HandleCount","ProcessCount","ThreadCount")]
        p = PI(); ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), ctypes.sizeof(PI))
    bench(gpi, label="GetPerformanceInfo (裸)")
except Exception as e:
    print("ctypes 探测失败:", e)

print()
vm = psutil.virtual_memory()
print(f"物理内存 total {vm.total/2**30:.1f} GiB  available {vm.available/2**30:.1f} GiB  used {vm.percent}%")
sw = psutil.swap_memory()
print(f"页文件 total {sw.total/2**30:.1f} GiB  used {sw.used/2**30:.1f} GiB  {sw.percent}%")
