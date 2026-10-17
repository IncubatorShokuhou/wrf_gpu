"""Optional held state is exact, or refused with E78; never cold-start active physics."""
from dataclasses import replace
import json
import pickle

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.io import restart as rp
from gpuwrf.io import wrfrst_netcdf as nc
from gpuwrf.io.restart_schema import CAMUW_FIELDS, EFFECTIVE_RADIUS_FIELDS
from gpuwrf.physics.bl_camuw import CamUwCarry
from gpuwrf.runtime import checkpoint as cp
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state

TIMES = dict(valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00", step_index=2)


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", False)
    grid = make_ideal_grid(4, 3, 5)
    state = _state(grid)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics)
    return grid, nl, initial_operational_carry(state)


def exact(left, right):
    a, at = jax.tree.flatten(left)
    b, bt = jax.tree.flatten(right)
    assert at == bt
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert x.tobytes() == y.tobytes()


def populated(case, scheme):
    grid, nl, carry = case
    if scheme == "camuw":
        nz, ny, nx = carry.state.theta.shape
        values = [jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 32 + i
                  for i, shape in enumerate([(nz+1, ny, nx)] * 2 + [(ny, nx)] * 2 + [(nz+1, ny, nx)], 1)]
        return grid, replace(nl, bl_pbl_physics=9), carry.replace(camuw_pbl=CamUwCarry(*values))
    state = carry.state.ensure_conditional_leaves(mp_physics=40).replace(**{
        name: jnp.arange(carry.state.qc.size, dtype=jnp.float32).reshape(carry.state.qc.shape) / 2**20 + i / 2**12
        for i, name in enumerate(EFFECTIVE_RADIUS_FIELDS, 1)})
    return grid, replace(nl, mp_physics=40), carry.replace(state=state)


def save(path, fmt, grid, nl, carry):
    if fmt == "pickle":
        rp.write_restart(carry, nl, grid, 2, path)
    else:
        nc.write_wrfrst_carry(carry, grid, nl, path, **TIMES)


def read(path, fmt):
    return (rp.read_restart(path) if fmt == "pickle" else nc.read_wrfrst_carry(path))[0]


@pytest.mark.parametrize("scheme", ["camuw", "morrison"])
@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_active_carry_roundtrips_every_byte(case, tmp_path, scheme, fmt):
    grid, nl, carry = populated(case, scheme)
    path = tmp_path / fmt
    save(path, fmt, grid, nl, carry)
    exact(carry, read(path, fmt))
    if scheme == "camuw":
        assert tuple(carry.camuw_pbl._fields) == CAMUW_FIELDS
    else:
        path = tmp_path / "state.pkl"
        cp.write_checkpoint(carry.state, nl, grid, 2, path)
        exact(carry.state, cp.read_checkpoint(path)[0])
    store = RestartStore(tmp_path / "store", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {"scheme": scheme})
    exact(carry, store.read(generation)[0]["carries"]["d01"])


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("damage", ["marker", "group", "leaf"])
def test_active_camuw_pre_schema_or_missing_group_refused(case, tmp_path, fmt, damage):
    grid, nl, carry = populated(case, "camuw")
    path = tmp_path / fmt
    save(path, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        c = payload["carry"]
        if damage == "marker":
            c.pop("camuw_pbl_schema_version")
        elif damage == "group":
            c["extra_fields"].pop("camuw_pbl")
            c["carry_field_order"].remove("camuw_pbl")
        else:
            c["extra_fields"]["camuw_pbl"] = c["extra_fields"]["camuw_pbl"]._replace(kvm3d=None)
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            if damage == "marker":
                ds.delncattr("GPUWRF_CAMUW_PBL_SCHEMA_VERSION")
            elif damage == "group":
                for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", "GPUWRF_OPTIONAL_CARRY_KIND"):
                    manifest = json.loads(ds.getncattr(attr))
                    manifest.pop("camuw_pbl")
                    ds.setncattr(attr, json.dumps(manifest))
            else:
                # Exact variable deletion, keeping the declared manifest.
                ds.renameVariable("GPUWRF_CAMUW_PBL_LEAF_000", "REMOVED_LEAF")
    with pytest.raises(ValueError, match="CAM-UW.*E78|missing exact optional carry variables"):
        read(path, fmt)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_inactive_legacy_camuw_group_remains_readable(case, tmp_path, fmt):
    grid, nl, carry = case
    path = tmp_path / fmt
    save(path, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        payload["carry"].pop("camuw_pbl_schema_version")
        payload["carry"]["extra_fields"].pop("camuw_pbl")
        payload["carry"]["carry_field_order"].remove("camuw_pbl")
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            ds.delncattr("GPUWRF_CAMUW_PBL_SCHEMA_VERSION")
            ds.delncattr("GPUWRF_CAMUW_PBL_CONFIG")
            for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", "GPUWRF_OPTIONAL_CARRY_KIND"):
                manifest = json.loads(ds.getncattr(attr))
                manifest.pop("camuw_pbl")
                ds.setncattr(attr, json.dumps(manifest))
    if fmt == "netcdf":
        with pytest.raises(ValueError, match="CAM-UW selection is unknown.*E78"):
            read(path, fmt)
        exact(carry, nc.read_wrfrst_carry(path, namelist=nl)[0])
        with pytest.raises(ValueError, match="CAM-UW.*E78"):
            nc.read_wrfrst_carry(path, namelist=replace(nl, bl_pbl_physics=9))
        # WRF mp40/aercu_opt=0 uses default radii. Its allocated Registry
        # diagnostics do not make the optional radiation carry mandatory.
        exact(carry, nc.read_wrfrst_carry(path, namelist=replace(nl, mp_physics=40))[0])
    else:
        exact(carry, read(path, fmt))


@pytest.mark.parametrize("missing", EFFECTIVE_RADIUS_FIELDS)
@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "state"])
def test_morrison_missing_restart_radius_refused(case, tmp_path, missing, fmt):
    grid, nl, carry = populated(case, "morrison")
    path = tmp_path / fmt
    if fmt == "state":
        cp.write_checkpoint(carry.state, nl, grid, 2, path)
    else:
        save(path, fmt, grid, nl, carry)
    if fmt in ("pickle", "state"):
        payload = pickle.loads(path.read_bytes())
        state_payload = payload if fmt == "state" else payload["carry"]
        state_payload["state_field_order"].remove(missing)
        state_payload["state_fields"].pop(missing)
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            order = json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
            order.remove(missing)
            ds.GPUWRF_STATE_FIELD_ORDER = json.dumps(order)
    with pytest.raises(ValueError, match="effective radii.*E78"):
        cp.read_checkpoint(path) if fmt == "state" else read(path, fmt)


@pytest.mark.parametrize("scheme,fmt", [("camuw", "pickle"), ("camuw", "netcdf"),
                                       ("morrison", "pickle"), ("morrison", "netcdf"), ("morrison", "state")])
def test_active_missing_write_preserves_previous_file(case, tmp_path, fmt, scheme):
    grid, nl, carry = populated(case, scheme)
    path = tmp_path / fmt
    if fmt == "state":
        cp.write_checkpoint(carry.state, nl, grid, 2, path)
    else:
        save(path, fmt, grid, nl, carry)
    previous = path.read_bytes()
    damaged = (carry.replace(camuw_pbl=None) if scheme == "camuw" else
               carry.replace(state=carry.state.replace(re_ice=None)))
    with pytest.raises(ValueError, match="E78"):
        if fmt == "state":
            cp.write_checkpoint(damaged.state, nl, grid, 3, path)
        else:
            save(path, fmt, grid, nl, damaged)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("name", ["sfc_wspd", *EFFECTIVE_RADIUS_FIELDS])
def test_state_only_optional_fields_exact_or_active_refusal(case, monkeypatch, tmp_path, name):
    grid, nl, carry = case
    state = carry.state
    if name == "sfc_wspd":
        monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "1")
        state = state.replace(sfc_wspd=jnp.full_like(state.hfx, 7.25))
    else:
        from gpuwrf.physics import rrtmg_mp_re
        monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", True)
        state = state.ensure_conditional_leaves(mp_physics=8)
    path = tmp_path / "state.pkl"
    cp.write_checkpoint(state, nl, grid, 2, path)
    exact(state, cp.read_checkpoint(path)[0])
    payload = pickle.loads(path.read_bytes())
    payload["state_field_order"].remove(name)
    payload["state_fields"].pop(name)
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="E78"):
        cp.read_checkpoint(path)
