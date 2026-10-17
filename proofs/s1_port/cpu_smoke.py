"""Two genuine release-default CPU steps, without patching model/JAX dispatch.

Usage: python cpu_smoke.py SOURCE_ROOT CASE OUTPUT_JSON
CPU execution smoke only: this does not compare a forecast with CPU-WRF.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import warnings

root, case, output = (Path(v).resolve() for v in sys.argv[1:4])
sys.path.insert(0, str(root / "src"))
source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
source_tree = subprocess.check_output(["git", "rev-parse", "HEAD:src/gpuwrf"], cwd=root, text=True).strip()
original_run = subprocess.run


def no_nvidia_tools(command, *args, **kwargs):
    words = command if isinstance(command, (list, tuple)) else [command]
    assert not any(Path(str(w)).name in {"nvidia-smi", "ptxas", "cuobjdump", "ncu", "nsys"}
                   for w in words), "portable CPU run invoked NVIDIA tooling"
    return original_run(command, *args, **kwargs)


subprocess.run = no_nvidia_tools
warning_receipts = []
original_warning = warnings.showwarning


def record_warning(message, category, filename, lineno, *args, **kwargs):
    warning_receipts.append({"message": str(message), "category": category.__name__,
                             "file": filename, "line": lineno})
    return original_warning(message, category, filename, lineno, *args, **kwargs)


warnings.showwarning = record_warning
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(str(root))
t0 = time.perf_counter()
cfg = NestedPipelineConfig(case, output.parent / "smoke_out", output.parent / "smoke_proof",
                           hours=1, max_dom=1)
handles, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
print("CPU_SMOKE_LOADED", time.perf_counter() - t0, flush=True)
ns, carry = bundles["d01"].namelist, carries["d01"]
om._resolve_operational_suite(ns)
clock = om.build_clock_base(ns)
loaded = time.perf_counter()
result = carry
step_seconds = []
for step in (1, 2):
    started = time.perf_counter()
    result = om._advance_chunk(result, ns, step, clock, n_steps=1,
                               cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(result)
    step_seconds.append(time.perf_counter() - started)
    print("CPU_SMOKE_STEP", step, step_seconds[-1], flush=True)
finished = time.perf_counter()
leaves = [np.asarray(x) for x in jax.tree.leaves(result)]
floating = [x for x in leaves if np.issubdtype(x.dtype, np.floating)]
nonfinite = sum(int((~np.isfinite(x)).sum()) for x in floating)
census = getattr(result, "census", None)
guards = getattr(census, "guards", None)
receipt = {"source": source, "source_tree": source_tree, "platform": "cpu", "gpu_executed": False,
           "case": str(case), "steps": 2, "dt_s": float(ns.dt_s),
           "load_s": loaded - t0, "step_including_cold_compile_s": finished - loaded,
           "step_seconds_including_first_compile": step_seconds,
           "maxrss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "floating_leaves": len(floating), "nonfinite": nonfinite,
           "census_guard_total": None if guards is None else int(np.asarray(guards).sum()),
           "nvidia_tools_called": False,
           "jax_version": jax.__version__, "numpy_version": np.__version__,
           "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"),
           "namelist_sha256": hashlib.sha256((case / "namelist.input").read_bytes()).hexdigest(),
           "theta_shape": list(result.state.theta.shape),
           "work": {name: getattr(ns, name) for name in
                    ("mp_physics", "bl_pbl_physics", "sf_surface_physics", "ra_sw_physics",
                     "ra_lw_physics", "cu_physics", "radiation_cadence_steps",
                     "cumulus_cadence_steps", "force_fp64")},
           "warnings": warning_receipts,
           "scope": "unpatched operational CPU execution; no CPU-WRF trajectory/fidelity or GPU speed claim"}
output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
assert nonfinite == 0, receipt
print(json.dumps(receipt, default=str))
