"""Exercise the real CLI loader and two production SSiB steps on Swiss input.

CPU plumbing smoke only: private wrfinput soil profiles are interpolated from
the Swiss Noah input to SSiB levels. This is NOT real.exe or CPU-WRF forecast
fidelity evidence. Model-init/diagnostic fidelity comes from the pristine oracle.
The product loader itself never performs this test-only conversion.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, default=Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--levels", type=int, choices=(3,), default=3)
ap.add_argument("--steps", type=int, default=2)
args = ap.parse_args()
assert os.environ["JAX_PLATFORMS"] == "cpu"
assert os.environ["GPUWRF_JAX_CACHE"] == "0"
assert os.environ["GPUWRF_FAST_DEFAULTS"] == "0"
assert args.out.is_relative_to(Path("<USER_HOME>/wrf_gpu2_lanes/s2-ssib"))
if Path("/tmp/wrf_gpu2_quiet").exists():
    raise SystemExit("QUIET")
args.out.mkdir(parents=True, exist_ok=True)
case = args.out / "case"
case.mkdir()

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
import jax
assert jax.devices()[0].platform == "cpu"
import gpuwrf
from gpuwrf.contracts import state as state_contract
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "o1_nlbind"))
import cpu_safe
cpu_safe.install()  # E105: CPU test harness, not a product solver change.
SSiB_SOIL_LEVELS = {3: (0., .05, 1.05)}

for source in args.input.iterdir():
    if source.name not in ("namelist.input", "wrfinput_d01"):
        (case / source.name).symlink_to(source)
text = (args.input / "namelist.input").read_text()
for key, val in {"sf_surface_physics": 8, "num_soil_layers": args.levels,
                 "sf_sfclay_physics": 1, "bl_pbl_physics": 1, "ra_sw_physics": 1, "ra_lw_physics": 4, "fractional_seaice": 1, "mp_physics": 8, "cu_physics": 0}.items():
    pattern = re.compile(rf"^\s*{key}\s*=.*$", re.M | re.I)
    line = f" {key} = {val},"
    if pattern.search(text):
        text = pattern.sub(line, text)
    else:
        text = re.sub(r"(&physics\s*\n)", lambda m: m.group(1) + line + "\n", text, count=1, flags=re.I)
(case / "namelist.input").write_text(text)
zs = np.asarray(SSiB_SOIL_LEVELS[args.levels], np.float32)
with Dataset(args.input / "wrfinput_d01") as source, Dataset(case / "wrfinput_d01", "w") as target:
    target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
    target.SF_SURFACE_PHYSICS = 8
    for name, dim in source.dimensions.items():
        target.createDimension(name, args.levels if name == "soil_layers_stag" else len(dim))
    old_zs = np.asarray(source["ZS"][0])
    for name, var in source.variables.items():
        out = target.createVariable(name, var.dtype, var.dimensions,
                                    **({"fill_value": var._FillValue} if "_FillValue" in var.ncattrs() else {}))
        out.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
        if name == "ZS":
            out[:] = zs
        elif name == "DZS":
            dzs = np.asarray([.05, 1., 0.], np.float32)
            out[:] = dzs
        elif "soil_layers_stag" in var.dimensions:
            data = np.asarray(var[0])
            result = np.empty((args.levels,) + data.shape[1:], np.float32)
            top = np.asarray(source["TSK"][0]) if name == "TSLB" else data[0]
            deep = np.asarray(source["TMN"][0]) if name == "TSLB" else data[-1]
            for j in range(data.shape[1]):
                for i in range(data.shape[2]):
                    result[:, j, i] = np.interp(zs, np.r_[0., old_zs, 3.], np.r_[top[j,i], data[:,j,i], deep[j,i]])
            out[0] = result
        else:
            out[:] = var[:]

from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
cfg = NestedPipelineConfig(case, args.out / "output", args.out / "proof", hours=1, max_dom=1)
start = time.perf_counter()
hierarchy, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
load_s = time.perf_counter() - start
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
assert bundles["d01"].namelist.sf_surface_physics == 8
assert carries["d01"].history_diagnostics is not None
assert carries["d01"].ssib_land is not None
print(f"CPU load: {load_s:.2f} s", flush=True)
own = {"d01": 0}
times = []
for _ in range(args.steps):
    while Path("/tmp/wrf_gpu2_quiet").exists():
        time.sleep(1)
    start = time.perf_counter()
    result = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=own,
                                         block_between=False, root_sync_cadence=0)
    jax.block_until_ready(result.carries)
    carries, own = result.carries, dict(result.own_steps)
    times.append(time.perf_counter() - start)
    print(f"CPU step {own['d01']}: {times[-1]:.2f} s", flush=True)
carry = carries["d01"]
bad = [jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_leaves_with_path(carry)
       if np.asarray(leaf).dtype.kind == "f" and not np.isfinite(np.asarray(leaf)).all()]
guards = None if carry.census is None else int(np.asarray(carry.census.guards).sum())
record = {"platform": "cpu", "source_tree": str(Path(gpuwrf.__file__).resolve().parents[2]),
          "input": str(args.input), "levels": args.levels, "steps": args.steps,
          "load_s": load_s, "step_s": times, "nonfinite_leaves": bad, "guard_events": guards,
          "history_ranges": {name: [float(np.asarray(getattr(carry.history_diagnostics, name)).min()),
                                     float(np.asarray(getattr(carry.history_diagnostics, name)).max())]
                             for name in ("t2", "q2", "lh", "psfc")},
          "scope": "CLI/production CPU plumbing smoke; interpolated Swiss soil and original-WRF LANDUSE startup, not WRF forecast fidelity"}
(args.out / "smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
assert not bad and guards == 0, record
