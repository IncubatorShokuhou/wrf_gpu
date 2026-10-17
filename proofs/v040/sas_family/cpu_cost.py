"""Changed cu94 artifact CPU compile/buffer/timing diagnostic; never a GPU claim."""
from __future__ import annotations

import functools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64",True)
from gpuwrf.physics.cumulus_sas94 import cu_sas94_columns

ROOT = Path(__file__).resolve().parents[3]
assert jax.devices()[0].platform == "cpu"
inp = np.load(ROOT/"proofs/v034/scalesas/savepoints/columns_inputs.npz")
args = [inp[k] for k in ("T","QV","QC","QI","P","PI","RHO","DZ","U","V","W")]
args += [inp["P8W"][:,0],inp["XLAND"]]
fn = jax.jit(functools.partial(cu_sas94_columns,dt=18.,stepcu=1))
start = time.monotonic()
compiled = fn.lower(*args).compile()
compile_s = time.monotonic()-start
jax.block_until_ready(compiled(*args))
samples=[]
for _ in range(7):
    start = time.monotonic()
    jax.block_until_ready(compiled(*args))
    samples.append(time.monotonic()-start)
stats = compiled.memory_analysis()
fields = ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes","generated_code_size_in_bytes")
report = {"source":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    "backend":"cpu","work":{"columns":455,"levels":44,"dt":18.,"stepcu":1,"shallow":True},
    "compile_s":compile_s,"warm_s":samples,"median_warm_s":statistics.median(samples),
    "memory":{field:getattr(stats,field,None) for field in fields},
    "gpu_launches":"UNKNOWN","gpu_vram":"UNKNOWN","gpu_compile_readiness":"UNKNOWN"}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
