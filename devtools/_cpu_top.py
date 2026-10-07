import time, sys
try:
    import psutil
except ImportError:
    print("NO psutil"); sys.exit(1)

procs = {}
for p in psutil.process_iter():
    try:
        procs[p.pid] = p
    except Exception:
        pass

def snap():
    for p in procs.values():
        try:
            p.cpu_percent(None)
        except Exception:
            pass

snap()
time.sleep(5.0)

rows = []
for p in procs.values():
    try:
        c = p.cpu_percent(None)
        if c <= 0.5:
            continue
        rss = p.memory_info().rss / 2**30
        rows.append((c, p.pid, p.name(), rss))
    except Exception:
        pass
rows.sort(reverse=True)
print(f"{'CPU%':>7}  {'PID':>7}  {'RSS GiB':>8}  NAME")
for c, pid, name, rss in rows[:25]:
    print(f"{c:7.1f}  {pid:7d}  {rss:8.2f}  {name}")
print()
print("logical CPUs:", psutil.cpu_count(logical=True), " physical:", psutil.cpu_count(logical=False))
print("cpu freq:", psutil.cpu_freq())
print("total RAM GiB: %.1f  available: %.1f  used%%: %.1f" % (
    psutil.virtual_memory().total/2**30, psutil.virtual_memory().available/2**30,
    psutil.virtual_memory().percent))
