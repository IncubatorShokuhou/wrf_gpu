"""Checkpoint porosity must use retained soil before parameter-only early return."""
from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w44_override"))
from test_override import make_input
from gpuwrf.config.paths import wrf_run_dir
from gpuwrf.contracts.noahmp_state import NoahMPStatic
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
from gpuwrf.io.noahmp_land_init import build_noahmp_params
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.checkpoint_consistency import checkpoint_validator, STATE_TO_WRF
from gpuwrf.runtime.checkpoint_branch import apply_overrides
from gpuwrf.surface_params import controls as sc


@pytest.fixture
def branch(tmp_path):
    path = make_input(tmp_path / "wrfinput_d01")
    with Dataset(path) as ds:
        f = {name: np.asarray(value[0]).copy() for name, value in ds.variables.items() if name != "Times"}
    full = lambda value: jnp.full((2, 3), value, jnp.float32)
    static = NoahMPStatic(jnp.asarray(f["IVGTYP"]), jnp.asarray(f["ISLTYP"]), jnp.asarray(f["XLAND"]),
        jnp.asarray(f["LANDMASK"]), full(0), jnp.asarray(f["LU_INDEX"]), full(280),
        jnp.asarray([.1, .3, .6, 1.]), jnp.asarray([-.1, -.4, -1., -2.]), full(28), 9000.,
        load_noahmp_parameters(wrf_run_dir()), shdmax=full(.5), shdfac=full(.5))
    ep, rp, _ = build_noahmp_params(static)
    nl = OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True, sf_surface_physics=4,
        noahmp_static=static, noahmp_energy_params=ep, noahmp_rad_params=rp)
    state = {leaf: f[key] for leaf, key in STATE_TO_WRF.items() if key in f}
    state["theta"] = ((f["T"].astype(np.float64) + 300) * (1 + (461.6 / 287.) * state["qv"])).astype(np.float32)
    for leaf, perturbation, base in (("p_total", "P", "PB"), ("ph_total", "PH", "PHB"), ("mu_total", "MU", "MUB")):
        state[leaf] = f[perturbation] + f[base]
    soil = {"smois": f["SMOIS"], "sh2o": f["SMOIS"].copy(), "tslb": f["TSLB"]}
    base = SimpleNamespace(pb=f["PB"], phb=f["PHB"], mub=f["MUB"])
    payload = {"state_fields": state, "noahmp_land_fields": soil, "extra_fields": {"base_state": base}, "scratch_fields": {}}
    snapshot = {"carries": {"d01": payload}, "own_steps": {"d01": 100},
                "driver_state": {"outputs": {"d01": []}, "forecast_hours": 3}}
    return tmp_path, nl, snapshot


def requested(nl, snapshot, cap):
    initial = SimpleNamespace(smois=np.full((4, 2, 3), .2, np.float32), sh2o=np.full((4, 2, 3), .2, np.float32))
    controls = sc.prior_controls(nl, ["smcmax"], device=False, land_state=initial)
    controls["smcmax"][:, np.asarray(nl.noahmp_static.landmask) == 1] = np.asarray(cap, controls["smcmax"].dtype)
    # Branch start: the loader binds with cold_start=False (SH2O is restored, NOAHMP_INIT does not rerun).
    return sc.bind_domain_controls(nl, {"d01": controls}, "d01", land_state=initial, cold_start=False)


def branch_outcome(root, nl, snapshot, changes):
    validate = checkpoint_validator(root, {"d01": SimpleNamespace(namelist=nl)})
    try:
        value = apply_overrides(snapshot, changes, validate=validate)
        return {"passed": True, "result": value}
    except ValueError as exc:
        return {"passed": False, "message": str(exc)}


def test_parameter_only_branch_refuses_retained_water_above_lower_cap(branch, research_w2_policy):
    root, nl, snapshot = branch
    snapshot["carries"]["d01"]["noahmp_land_fields"]["smois"][:, 0, 0] = .32
    before = snapshot["carries"]["d01"]["noahmp_land_fields"]["smois"].tobytes()
    bound = requested(nl, snapshot, .3)
    changes = {"schema_version": 1, "parameters": {"surface_controls": {"d01": sc.restart_controls_digest(bound)}}}
    result = branch_outcome(root, bound, snapshot, changes)
    assert not result["passed"], "skipping validate_retained_porosity would admit a lowered cap below retained SMOIS"
    assert "soil_porosity" in result["message"]
    assert snapshot["carries"]["d01"]["noahmp_land_fields"]["smois"].tobytes() == before
    assert bound.noahmp_land is None


def test_controlled_soil_only_check_preserves_retained_atmospheric_supersaturation(branch, research_w2_policy):
    root, nl, snapshot = branch
    snapshot["carries"]["d01"]["state_fields"]["qv"][:] = .1
    before = snapshot["carries"]["d01"]["state_fields"]["qv"].tobytes()
    bound = requested(nl, snapshot, .3)
    result = branch_outcome(root, bound, snapshot, {"schema_version": 1})
    assert result["passed"]
    new, reports = result["result"]
    assert reports["consistency"]["d01"]["controlled_soil_porosity"]["passed"]
    assert new["carries"]["d01"]["state_fields"]["qv"].tobytes() == before


def test_joint_state_and_porosity_edit_uses_effective_cap(branch, research_w2_policy):
    root, nl, snapshot = branch
    bound = requested(nl, snapshot, .4)
    increment = np.broadcast_to(np.asarray(nl.noahmp_static.landmask) * .18, (4, 2, 3))
    changes = {"schema_version": 1, "domains": {"d01": {"land": {"smois": {"add": increment}}}}}
    result = branch_outcome(root, bound, snapshot, changes)
    assert result["passed"], result.get("message")
    _, reports = result["result"]
    assert reports["consistency"]["d01"]["soil_cap_source"] == "explicit effective SMCMAX"
