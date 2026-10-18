"""Run scripts/wave_cubin_gate.py unchanged, but let cuobjdump take up to 1200 s instead of 180 s.

The shared scanner (scripts/e41/cubin_scan.py) times cuobjdump -sass out at 180 s; on this host at load 55-80
the WN3 d01 step cubins exceeded it (gpusmoke1 arms A and B, 2026-10-17). Only the timeout changes; every
check and threshold of the gate is the product one.
usage: JAX_PLATFORMS=cpu python proofs/w41_optimize/e41_gate_slowhost.py CACHE OUT
"""
import runpy
import subprocess
import sys
from pathlib import Path

_run = subprocess.run


def _patient(*args, **kwargs):
    if kwargs.get("timeout") == 180:
        kwargs["timeout"] = 1200
    return _run(*args, **kwargs)


subprocess.run = _patient
sys.argv = ["wave_cubin_gate.py", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parents[2]/"scripts/wave_cubin_gate.py"), run_name="__main__")
