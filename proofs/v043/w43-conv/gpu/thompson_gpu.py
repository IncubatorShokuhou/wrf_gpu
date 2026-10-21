"""CG01: the CT02 port arm on the asserted GPU (Triton-compiled full_column, interpret=False, release fast defaults) on the
same 586 convective columns; WRF side = the CT02 pristine outputs (CPU Fortran, unchanged). Score afterwards on CPU with
analysis/thompson/score.py (frozen CT03 gate).
usage: thompson_gpu.py <columns.npz> <ct02_oracle.npz> <out.npz>"""
import os, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thompson"))
assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1", "GPU arm must run under scripts/with_gpu_lock.sh"
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import jax
assert jax.devices()[0].platform == "gpu", jax.devices()
import run_oracle as ro
z, o = dict(np.load(sys.argv[1])), dict(np.load(sys.argv[2]))
res = {k: v for k, v in o.items() if k.startswith("wrf_")}
for dt in sorted(set(z["dt"].tolist())):
    idx = np.flatnonzero(z["dt"] == dt)
    t0 = time.time()
    for k, v in ro.run_port(z, idx, dt, interpret=False, platform="gpu").items():
        res.setdefault(f"port_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
    print("gpu dt", dt, len(idx), "%.1fs" % (time.time() - t0), flush=True)
res["env_applied"] = np.array(["gpu", str(jax.devices()[0])])
np.savez_compressed(sys.argv[3], **res)
