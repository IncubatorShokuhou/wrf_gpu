"""Memory anatomy of held netCDF handles (w42-scorer): open every CPU+GPU wrfout of one domain, then read every common
variable from every file in the comparator's variable-major order (values discarded), printing VmRSS / VmHWM.
usage: python probe_handles.py CPU_DIR GPU_DIR DOMAIN FRAMES CACHE_BYTES [close_each]
CACHE_BYTES -1 = netCDF-C default (no set_chunk_cache call). close_each=1: open/close per read (control, no held handles)."""
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np


def mem():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            out[k] = int(v.split()[0]) // 1024
    return out


cpu_dir, gpu_dir, domain, frames, cache = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
close_each = len(sys.argv) > 6 and sys.argv[6] == "1"
if cache >= 0:
    netCDF4.set_chunk_cache(cache, 521 if cache else 0, 0.75)
print("chunk_cache", netCDF4.get_chunk_cache(), "start", mem(), flush=True)
cpu = sorted(cpu_dir.glob(f"wrfout_{domain}_*"))[:frames]
gpu = [gpu_dir / p.name for p in cpu]
t0 = time.process_time()
if close_each:
    handles = None
else:
    handles = [(netCDF4.Dataset(c, "r"), netCDF4.Dataset(g, "r")) for c, g in zip(cpu, gpu)]
    print("opened", 2 * len(handles), "files", mem(), "cpu_s", round(time.process_time() - t0, 1), flush=True)
with netCDF4.Dataset(cpu[0]) as c0, netCDF4.Dataset(gpu[0]) as g0:
    names = sorted(set(c0.variables) & set(g0.variables))
for i, name in enumerate(names, 1):
    for c, g in (handles if handles is not None else zip(cpu, gpu)):
        for ds in (c, g):
            if close_each:
                with netCDF4.Dataset(ds, "r") as d:
                    v = d.variables[name]
                    np.asarray(v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:])
            else:
                v = ds.variables[name]
                np.asarray(v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:])
    if i % 50 == 0 or i == len(names):
        print(i, name, mem(), "cpu_s", round(time.process_time() - t0, 1), flush=True)
if handles:
    for c, g in handles:
        c.close(); g.close()
print("closed", mem(), "cpu_s", round(time.process_time() - t0, 1), flush=True)
