"""Conditional three-step column sensitivity, not forecast skill or WRF parity.

Only the named constants are changed in this process; clear all JIT caches per
arm. Surface forcing stays fixed. The entrainment packing control must recover
baseline bytes before its perturbation can be interpreted.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.kernels import phys_mynn_plume as plume

assert jax.devices()[0].platform == "cpu"
assert os.environ.get("GPUWRF_MYNN_FP32_COLUMNS") == "1"
fixture = args.source / "tests/v025/b_phys/fixtures/mynn_predict_saw_wrf.json"
data = json.loads(fixture.read_text())
original = {name: getattr(P, name) for name in ("NL_ALP1", "NL_ELT_MAX", "NL_ELT_MAX_WATER", "CPHM_UNST")}
original_stack, original_phim = plume._scalar_stack, P._phim_puhales
arms = {
    "baseline": {}, "alp1_up10": {"NL_ALP1": .23 * 1.1}, "alp1_down10": {"NL_ALP1": .23 * .9},
    "caps_up10": {"NL_ELT_MAX": 440., "NL_ELT_MAX_WATER": 385.},
    "caps_half": {"NL_ELT_MAX": 200., "NL_ELT_MAX_WATER": 175.},
    "cphm_unst_up10": {"CPHM_UNST": 17.6},
    "am_st_control": {"am_st": 6.1},
    "am_st_up10": {"am_st": 6.71},
    "entrainment_pack_control": {"entrainment": .33},
    "entrainment_up10": {"entrainment": .363},
}
fields = ("tke", "km", "kh", "el", "u", "v", "theta", "qv", "qc", "qi")
report = {"source": str(args.source.resolve()), "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
          "platform": "cpu", "gpu_executed": False, "steps": 3,
          "scope": "real column inputs, fixed surface forcing, static constant perturbations; NOT forecast sensitivity/skill or WRF oracle", "arms": {}}
args.out.parent.mkdir(parents=True, exist_ok=True)
references = {}


def set_arm(values):
    for name, value in original.items():
        setattr(P, name, values.get(name, value))
    plume._scalar_stack, P._phim_puhales = original_stack, original_phim
    if "entrainment" in values:
        value = values["entrainment"]
        plume._scalar_stack = lambda: original_stack().at[plume._C033].set(jnp.float32(value))
    if "am_st" in values:
        text = inspect.getsource(original_phim)
        assert text.count("am_st = 6.1") == 1
        text = text.replace("am_st = 6.1", "am_st = " + repr(values["am_st"]))
        namespace = dict(P.__dict__)
        exec(compile(text, P.__file__, "exec"), namespace)
        P._phim_puhales = namespace["_phim_puhales"]
    jax.clear_caches()


for label, values in arms.items():
    set_arm(values)
    result = {}
    t0 = time.monotonic()
    for row in data["rows"]:
        array = lambda value: None if value is None else jnp.asarray(value, jnp.float32)
        state = P.MynnPBLColumnState(**{name: array(value) for name, value in row["state"].items()})
        surface = SurfaceFluxes(**{name: array(value) for name, value in row["flux"].items()})
        snapshots = []
        with jax.enable_x64(False):
            for step in range(3):
                state, _ = P.step_mynn_pbl_column_with_pblh(state, row["dt"], surface=surface, edmf=True, dx=row["dx"])
                if step in (0, 2):
                    snapshots.append({name: np.asarray(getattr(state, name)).copy() for name in fields})
        assert all(np.isfinite(value).all() for snap in snapshots for value in snap.values())
        if label == "baseline":
            references[row["label"]] = snapshots
        changes = []
        for step, (base, actual) in zip((1, 3), zip(references[row["label"]], snapshots)):
            metrics = {}
            for name in fields:
                before, after = base[name], actual[name]
                delta = after.astype(float) - before.astype(float)
                metrics[name] = {"changed_cells": int(np.count_nonzero(before != after)),
                    "rms_change": float(np.sqrt(np.mean(delta**2))), "max_abs_change": float(np.max(np.abs(delta))),
                    "reference_rms": float(np.sqrt(np.mean(before.astype(float)**2)))}
            changes.append({"step": step, "fields": metrics})
        if label in ("entrainment_pack_control", "am_st_control"):
            assert all(np.array_equal(a[name], b[name]) for a, b in zip(references[row["label"]], snapshots) for name in fields)
        result[row["label"]] = {"dt": row["dt"], "ncol": int(state.u.shape[0]), "changes": changes}
    report["arms"][label] = {"changes": values, "cohorts": result, "wall_s": time.monotonic() - t0}
    args.out.write_text(json.dumps(report, indent=2))
    print(label, round(report["arms"][label]["wall_s"], 2), flush=True)
set_arm({})
