"""HVT's PHENOLOGY consumer on a real grass state with declared snow cover.

Uses original WRF objects through the existing full-step wrapper. This checks
the static parameter route independently of ENERGY's pre-gathered HVT field.
"""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--mutant", action="store_true", help="test-only omission of PHENOLOGY's controlled HVT")
a = p.parse_args()
sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
import jax
from columns import build
import oracle as wrapper
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
from gpuwrf.surface_params import controls as sc
assert jax.devices()[0].platform == "cpu"
source = json.loads((a.repo / "tests/v025/b_noahmp/fixtures/savepoints_energy_frozen.json").read_text())
row = copy.deepcopy(source["columns"][3])
assert row["vegtyp"] == 10
row.update(name="real_frozen_grass_with_structured_snow", zsoil=[-.05, -.25, -.7, -1.5],
           dt=90., dx=1000., lat_rad=float(np.deg2rad(28.)), shdmax=row["shdfac"], tbot=285.)
row["state_in"].update(isnow=-1, snowh=.3, sneqv=75., sneqvo=75., albold=.75,
                        snice=[0., 0., 70.], snliq=[0., 0., 5.])
row["state_in"]["stc"][2] = 268.
row["state_in"]["zsnso"] = [0., 0., -.3, -.35, -.55, -1., -1.8]
land, forcing, static, ep, rp, nl, _ = build(rows=[row])
build_record = json.loads((a.build / "build.json").read_text())
executable = a.build / "w2_driver.exe"
assert wrapper.sha(executable) == build_record["executable_sha256"]
a.out.mkdir(parents=True, exist_ok=True)
references = []
parser = wrapper.module("height_wrf_parser", a.repo / "proofs/noahmp/build_noahmp_savepoints.py")
for factor in (1., 1.1):
    cwd = a.out / str(factor)
    cwd.mkdir(parents=True, exist_ok=True)
    wrapper.edited_tables(a.repo, Path("<USER_HOME>/src/wrf_pristine/WRF"), cwd, "hvt", factor)
    wrapper.write_columns(cwd / "noahmp_columns.in", [row])
    env = dict(os.environ, LD_LIBRARY_PATH="<USER_HOME>/miniconda3/envs/wrfbuild/lib",
               NOAHMP_WIRE_SOIL="1", NOAHMP_WIRE_FICEOLD="1", NOAHMP_NSTEPS="1", W2_OPT_CRS="1")
    result = subprocess.run([str(executable)], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "NOAHMP_OFFLINE_OK" in result.stdout, result.stdout
    parser.HERE = cwd
    references.append(parser._parse_savepoints()[0]["phen_out"])
traces = []


def program(values):
    traces.append(1)
    s, _, _ = sc.apply_surface_controls(static, ep, rp, values)
    if a.mutant:
        s = s.replace(parameters=s.parameters._replace(hvt=static.parameters.hvt))
    return noahmp_phenology_table(land, forcing, s)


fn = jax.jit(program)
outputs = []
for factor, reference in zip((1., 1.1), references, strict=True):
    values = sc.expand_controls(nl, {"hvt": factor})
    result = jax.device_get(fn(values))
    outputs.append({name: float(np.asarray(getattr(result, name)).item()) for name in ("lai", "sai", "fveg")})
assert references[0]["fveg"] == 0. and references[1]["fveg"] > .05
receipt = {"source": str(a.repo), "platform": "cpu", "gpu_executed": False, "mutant": a.mutant,
           "input": row, "references": references, "native": outputs, "traces": len(traces),
           "jit_cache_size": fn._cache_size(), "tolerance_source": "phenology_savepoint_gate.TOL", "atol": 1.e-5, "rtol": 1.e-5}
(a.out / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "input"}), flush=True)
for actual, reference in zip(outputs, references, strict=True):
    for name in ("lai", "sai", "fveg"):
        assert np.isfinite(actual[name]) and np.isfinite(reference[name])
        assert abs(actual[name] - reference[name]) <= 1.e-5 + 1.e-5 * abs(reference[name]), "HVT PHENOLOGY consumer differs from pristine WRF"
assert len(traces) == 1 and fn._cache_size() == 1
