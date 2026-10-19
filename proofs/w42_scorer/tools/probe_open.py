import sys, tracemalloc, gc
from pathlib import Path
import netCDF4
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1])//1024
files = sorted(Path(sys.argv[1]).glob("wrfout_d02_*"))[:5]
netCDF4.set_chunk_cache(0, 0, 0.75)
r0 = rss(); tracemalloc.start()
hs = [netCDF4.Dataset(f, "r") for f in files]
cur, peak = tracemalloc.get_traced_memory()
print("open5 rss_delta_MiB", rss()-r0, "python_traced_MiB", round(cur/2**20,1))
for h in hs: h.close()
del hs; gc.collect()
print("after close rss", rss()-r0, "traced", round(tracemalloc.get_traced_memory()[0]/2**20,1))
hs = [netCDF4.Dataset(f, "r") for f in files]
print("reopen5 rss_delta", rss()-r0)
