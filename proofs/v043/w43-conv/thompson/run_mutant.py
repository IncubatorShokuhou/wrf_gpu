"""CT04: deletion-sensitivity (E202) of the CT02/CT03 gate on the same 586 convective columns: port-only re-run with ONE
WRF element removed, scored against the CT02 pristine outputs with the frozen CT03 gate (fresh process per mutant).
Mutants: mp_off (GPUWRF_THOMPSON_MIXED_PHASE_WRF=0, staged pre-B55 mixed-phase path), const_off
(GPUWRF_THOMPSON_WRF_CONSTANTS=0), nstep1 (_nstep_per_column -> 1: no adaptive sedimentation substeps).
usage: run_mutant.py <columns.npz> <ct02_oracle.npz> <mutant> <out.npz>"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
mut = sys.argv[3]
if mut == "mp_off":
    os.environ["GPUWRF_THOMPSON_MIXED_PHASE_WRF"] = "0"
elif mut == "const_off":
    os.environ["GPUWRF_THOMPSON_WRF_CONSTANTS"] = "0"
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import run_oracle as ro
if mut == "nstep1":
    import jax.numpy as jnp
    from gpuwrf.physics import thompson_column as tc
    tc._nstep_per_column = lambda vt_a, vt_b, dz, dt: jnp.ones(jnp.shape(vt_a)[:-1], jnp.float32)
z, o = dict(np.load(sys.argv[1])), dict(np.load(sys.argv[2]))
res = {k: v for k, v in o.items() if k.startswith("wrf_")}
for dt in sorted(set(z["dt"].tolist())):
    idx = np.flatnonzero(z["dt"] == dt)
    for k, v in ro.run_port(z, idx, dt).items():
        res.setdefault(f"port_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
    print(mut, "dt", dt, flush=True)
res["env_applied"] = np.array([mut])
np.savez_compressed(sys.argv[4], **res)
