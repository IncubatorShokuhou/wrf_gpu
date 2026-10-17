"""CPU-only allocation/timing receipt; no forecast or GPU speed claim.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c25
nice -n19 timeout60 python proofs/u1_app/benchmark.py output.json
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from gpuwrf.post.apps.wind import rotor_wind
from gpuwrf.post.apps.fire import daily_fwi
from gpuwrf.post.apps.hazards import visibility

assert os.environ["JAX_PLATFORMS"] == "cpu"
assert os.environ["GPUWRF_JAX_CACHE"] == "0"
shape=(117,267,44)
z=np.broadcast_to(np.linspace(20,18000,44),shape)
u=np.broadcast_to(np.linspace(4,30,44),shape)
v=np.full(shape,2.)
# Match the profile's near-surface coverage to the chosen rotor.
results={}
selected = set(sys.argv[2:] or ["rotor_wind", "daily_fwi", "visibility"])
assert selected <= {"rotor_wind", "daily_fwi", "visibility"}
for name, call in (
    ("rotor_wind",lambda:rotor_wind(z,u,v,hub_height=100,rotor_diameter=100,samples=9)),
    ("daily_fwi",lambda:daily_fwi(np.full(shape[:2],25),35,25,0,latitude=28,month=7,ffmc_previous=85,dmc_previous=6,dc_previous=15)),
    ("visibility",lambda:visibility(np.full(shape[:2],270),95000,.004,.0002,.0003,.0001,.0004)),
):
    if name not in selected:
        continue
    times=[]
    for _ in range(5):
        before=time.perf_counter();result=call();times.append(time.perf_counter()-before)
        leaves=result.values() if isinstance(result,dict) else result if isinstance(result,tuple) else [result]
        assert all(np.isfinite(a).all() for a in leaves)
    results[name]={"seconds":times,"median_s":float(np.median(times))}
receipt={"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         "work":"CPU25 NI19 one process, shape117x267x44; 5 repetitions; offline diagnostics only, concurrent host load possible",
         "jax_platforms":os.environ["JAX_PLATFORMS"],"cache":"NONE",
         "results":results,"process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "model_gpu_launches_added":0,"model_compiles_added":0,
         "profile_inputs":"broadcast view z/u, allocated v (~10.5MiB), NumPy float64 diagnostics"}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,sort_keys=True))
