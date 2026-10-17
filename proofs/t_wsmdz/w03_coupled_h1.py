"""t-wsmdz W03: one-hour CPU-port coupled run of a Swiss composite vs ORIGINAL CPU-WRF at h1.

Same product entry as t-mprain RA06 (_load_domains + _advance_chunk n_steps=1, CPU-portable
dispatch, release defaults). Scores the h1 port state and held radiation diagnostics
against the CPU-WRF h1 history: OLR (lw_toa_up), per-level interior THM/QSNOW/U.
CPU causal evidence only; not GPU qualification, not a timing benchmark.
Usage: python w03_coupled_h1.py <src_root> <case_dir> <out.json>
"""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

root, case, output = (Path(p).resolve() for p in sys.argv[1:4])
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path.insert(0, str(root / "src"))
import jax  # noqa: E402
import numpy as np  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402
from gpuwrf.physics.rrtmg_mp_re import mp_re_active  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(str(root)), om.__file__
source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
assert not dirty, dirty
work = output.parent / (output.stem + "_work")
cfg = NestedPipelineConfig(case / "input", work / "out", work / "proof", hours=1, max_dom=1)
t0 = time.perf_counter()
handles, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
ns, carry = bundles["d01"].namelist, carries["d01"]
clock = om.build_clock_base(ns)
radius = bool(mp_re_active(ns))
print("LOADED", time.perf_counter() - t0, "mp", ns.mp_physics, "mp_re_active", radius,
      "re_cloud", carry.state.re_cloud is not None, flush=True)
first = None
for step in range(1, 201):
    while Path("/tmp/wrf_gpu2_quiet").exists():
        print("QUIET_WAIT", step, flush=True)
        time.sleep(10)
    s = time.perf_counter()
    carry = om._advance_chunk(carry, ns, step, clock, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(carry)
    if step == 1:
        first = time.perf_counter() - s
        print("FIRST_STEP", first, flush=True)
    if step % 50 == 0:
        print("STEP", step, time.perf_counter() - t0, flush=True)
leaves = [np.asarray(a) for a in jax.tree.leaves(carry)]
finite = all(np.isfinite(a).all() for a in leaves if np.issubdtype(a.dtype, np.floating))
st = carry.state
port = {
    "THM": np.asarray(st.theta, np.float64) - 300.0,
    "QSNOW": np.asarray(st.qs, np.float64),
    "QICE": np.asarray(st.qi, np.float64),
    "QVAPOR": np.asarray(st.qv, np.float64),
    "U": np.asarray(st.u, np.float64),
    "OLR": np.asarray(carry.radiation_diagnostics.lw_toa_up, np.float64),
}
ref_path = sorted((case / "cpu").glob("wrfout_d01_*"))[1]
gpu_path = sorted((case / "gpu").glob("wrfout_d01_*"))[1]
res = {"schema": "t-wsmdz-w03-v1", "source": source, "case": str(case), "reference": str(ref_path),
       "mp_physics": int(ns.mp_physics), "mp_re_active": radius, "finite": bool(finite),
       "first_step_s": first, "wall_s": time.perf_counter() - t0,
       "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "fields": {}}
with Dataset(ref_path) as ref, Dataset(gpu_path) as gpu:
    for name, a in port.items():
        r = np.asarray(ref.variables[name][0], np.float64)
        g = np.asarray(gpu.variables[name][0], np.float64)
        rec = {}
        for tag, x in (("port_cpu", a), ("paid_gpu", g)):
            d = x - r
            rec[tag] = {"rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean())}
            if d.ndim == 3:
                inner = d[:, 5:-5, 5:-5]
                rec[tag]["level_rms_k0_14"] = [float(v) for v in np.sqrt((inner ** 2).mean(axis=(1, 2)))[:15]]
                rec[tag]["level_bias_k0_14"] = [float(v) for v in inner.mean(axis=(1, 2))[:15]]
        res["fields"][name] = rec
        print(name, {t: (round(v["rmse"], 5), round(v["bias"], 5)) for t, v in rec.items()}, flush=True)
output.write_text(json.dumps(res, indent=1))
print("DONE", output, flush=True)
