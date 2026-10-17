"""Structural restart evidence; synthetic scalars are not WRF physics evidence."""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.runtime import checkpoint as cp
from gpuwrf.tracers.config import Emission, TracerSpecies, UserTracerConfig
from gpuwrf.tracers.restart import configuration, validate_restart
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
from test_online_diagnostics_restart_store import identity, rewrite


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", False)
    grid = make_ideal_grid(5, 4, 4)
    config = UserTracerConfig((TracerSpecies("SMOKE", .01, .02, .03),),
                              (Emission(0, 0., 100., 0, 1, 0, 1, 2., 0., True),), "a" * 64)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
                             tracer_opt=2, tracer_config=config)
    tracers = np.arange(9 * 5 * 4 * 4, dtype=np.float32).reshape(9, 5, 4, 4) / np.float32(1024)
    state = _state(grid).replace(tracers=tracers, tracer_budget=np.arange(36, dtype=np.float32).reshape(9, 4))
    carry = initial_operational_carry(state).replace(tracer_tendencies=tracers * np.float32(.125))
    return grid, nl, carry


def bound_identity(nl):
    return {**identity(nl), "tracer_config": {"d01": configuration(nl)}}


def write(path, fmt, grid, nl, carry):
    if fmt == "pickle":
        rp.write_restart(carry, nl, grid, 17, path)
    elif fmt == "netcdf":
        start = datetime(2026, 6, 3)
        nc.write_wrfrst_carry(carry, grid, nl, path, valid_time=start + timedelta(seconds=17 * nl.dt_s),
                             run_start=start, step_index=17)
    elif fmt == "checkpoint":
        cp.write_checkpoint(carry.state, nl, grid, 17, path, runtime_state=carry)
    else:
        store = RestartStore(path, 10**7, 2, 0)
        generation, _ = store.save({"d01": carry}, {"d01": 17}, {"d01": nl.dt_s}, bound_identity(nl))
        return generation
    return path


def read(path, fmt, nl=None):
    if fmt == "pickle":
        return rp.read_restart(path, namelist=nl)[0]
    if fmt == "netcdf":
        return nc.read_wrfrst_carry(path, namelist=nl)[0]
    if fmt == "checkpoint":
        state, _, _, _, carry = cp.read_checkpoint_with_runtime_state(path, namelist=nl)
        exact(state, carry.state)
        return carry
    return RestartStore(path.parent, 10**7, 2, 0).read(path,
        expected_identity=None if nl is None else bound_identity(nl))[0]["carries"]["d01"]


def exact(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "native", "checkpoint"])
@pytest.mark.parametrize("nested", [False, True])
def test_full_tracer_carry_byte_exact(case, tmp_path, fmt, nested):
    grid, nl, carry = case
    if nested:
        boundary = replace(nl.boundary_config, nested_frozen_wrf_boundary_bundle=True, force_geopotential=False)
        nl = replace(nl, boundary_config=boundary)
        bdy = np.arange(9 * 2 * 4 * 1 * 5 * 5, dtype=np.float32).reshape(9, 2, 4, 1, 5, 5)
        carry = carry.replace(state=carry.state.replace(tracer_bdy=bdy))
    path = write(tmp_path / fmt, fmt, grid, nl, carry)
    restored = read(path, fmt, nl)
    exact(carry, restored)
    # Advancing the same held-source recurrence after restore is exact IO
    # continuation evidence, not a substitute for a coupled forecast gate.
    exact(carry.state.tracers + np.float32(18) * carry.tracer_tendencies,
          restored.state.tracers + np.float32(18) * restored.tracer_tendencies)


def damage_payload(payload, kind):
    if kind == "marker":
        payload.pop("tracer_schema_version")
    elif kind == "config":
        payload.pop("tracer_config", None)
    elif kind == "tendency":
        payload["extra_fields"]["tracer_tendencies"] = None
    elif kind in ("tracers", "tracer_budget", "tracer_bdy"):
        payload["state_fields"].pop(kind, None)
        if kind in payload["state_field_order"]:
            payload["state_field_order"].remove(kind)
    elif kind == "nonfinite":
        payload["state_fields"]["tracers"].flat[0] = np.nan
    elif kind == "precision":
        payload["state_fields"]["tracers"] = payload["state_fields"]["tracers"].astype(np.float64)
    elif kind == "shape":
        payload["state_fields"]["tracers"] = payload["state_fields"]["tracers"][..., :-1]


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "native", "checkpoint"])
@pytest.mark.parametrize("kind", ["marker", "config", "tendency", "tracers", "tracer_budget", "nonfinite", "precision", "shape"])
def test_corrupt_active_restart_refused(case, tmp_path, fmt, kind):
    grid, nl, carry = case
    path = write(tmp_path / fmt, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        damage_payload(payload["carry"], kind)
        path.write_bytes(pickle.dumps(payload))
    elif fmt == "checkpoint":
        payload = pickle.loads(path.read_bytes())
        if kind == "tendency":
            payload["runtime_state"] = payload["runtime_state"].replace(tracer_tendencies=None)
        else:
            damage_payload(payload, kind)
        path.write_bytes(pickle.dumps(payload))
    elif fmt == "native":
        def change(payload, manifest):
            if kind == "config":
                manifest["identity"].pop("tracer_config")
            else:
                damage_payload(payload, kind)
        rewrite(path, change)
    else:
        with Dataset(path, "a") as ds:
            if kind in ("marker", "config"):
                ds.delncattr("GPUWRF_TRACER_SCHEMA_VERSION" if kind == "marker" else "GPUWRF_TRACER_CONFIG")
            elif kind == "tendency":
                order = json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER)
                order["tracer_tendencies"] = []
                kinds = json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND)
                kinds["tracer_tendencies"] = "none"
                ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER = json.dumps(order)
                ds.GPUWRF_OPTIONAL_CARRY_KIND = json.dumps(kinds)
            elif kind in ("tracers", "tracer_budget"):
                order = json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
                order.remove(kind)
                ds.GPUWRF_STATE_FIELD_ORDER = json.dumps(order)
            elif kind == "nonfinite":
                ds.variables[nc.state_extension_name("tracers")][0, 0, 0, 0, 0] = np.nan
            else:
                name = nc.state_extension_name("tracers")
                old = ds.variables[name]
                data, dims = np.asarray(old[...]), old.dimensions
                ds.renameVariable(name, name + "_CORRUPT_ORIGINAL")
                if kind == "shape":
                    ds.createDimension("damaged_tracer_x", 3)
                    dims = (*dims[:-1], "damaged_tracer_x")
                    data = data[..., :-1]
                ds.createVariable(name, "f8" if kind == "precision" else "f4", dims)[...] = data
    with pytest.raises(ValueError):
        read(path, fmt)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "native", "checkpoint"])
@pytest.mark.parametrize("change", ["source", "species", "decay", "deposition", "scavenging", "emission", "transport", "dt", "off"])
def test_requested_configuration_change_refused(case, tmp_path, fmt, change):
    grid, nl, carry = case
    path = write(tmp_path / fmt, fmt, grid, nl, carry)
    cfg = nl.tracer_config
    if change == "source":
        cfg = replace(cfg, source_sha256="b" * 64)
    elif change == "species":
        cfg = replace(cfg, species=(replace(cfg.species[0], name="POLLEN"),))
    elif change in ("decay", "deposition", "scavenging"):
        key = {"decay": "decay_rate_s", "deposition": "deposition_velocity_m_s", "scavenging": "wet_scavenging_per_mm"}[change]
        cfg = replace(cfg, species=(replace(cfg.species[0], **{key: .9}),))
    elif change == "emission":
        cfg = replace(cfg, emissions=(replace(cfg.emissions[0], rate_kg_s=3.),))
    requested = replace(nl, tracer_config=cfg)
    if change == "transport":
        requested = replace(requested, tracer_mix2_off=True)
    if change == "dt":
        requested = replace(requested, dt_s=nl.dt_s * 2)
    if change == "off":
        requested = replace(requested, tracer_opt=0, tracer_config=None)
    with pytest.raises(ValueError, match="tracer|identity"):
        read(path, fmt, requested)


def test_configuration_flat_grouped_and_content_bound(case):
    _, nl, _ = case
    expected = configuration(nl)
    assert configuration(expected) == expected
    assert configuration({"dynamics": {"tracer_opt": [2]}, "domains": {"time_step": nl.dt_s},
                          "tracer_config": nl.tracer_config}) == expected
    assert configuration(replace(nl, gpuwrf_tracer_file="other/path.json")) == expected
    nested = SimpleNamespace(nested_frozen_wrf_boundary_bundle=True, force_geopotential=False)
    assert configuration({"tracer_opt": 2, "dt_s": nl.dt_s, "boundary_config": nested})["nested"]
    assert not configuration({"tracer_opt": 2, "dt_s": nl.dt_s, "boundary_config": nested, "run_boundary": False})["nested"]


@pytest.mark.filterwarnings("error:overflow encountered:RuntimeWarning")
def test_restart_configuration_refuses_real32_overflow(case):
    _, nl, _ = case
    with pytest.raises(ValueError, match="tracer"):
        configuration(replace(nl, dt_s=1e300))
    for key in ("decay_rate_s", "deposition_velocity_m_s", "wet_scavenging_per_mm"):
        species = replace(nl.tracer_config.species[0], **{key: 1e300})
        with pytest.raises(ValueError, match="tracer"):
            configuration(replace(nl, tracer_config=replace(nl.tracer_config, species=(species,))))
    for key in ("end_s", "rate_kg_s", "flux_kg_m2_s"):
        source = replace(nl.tracer_config.emissions[0], **{key: 1e300})
        with pytest.raises(ValueError, match="tracer"):
            configuration(replace(nl, tracer_config=replace(nl.tracer_config, emissions=(source,))))


def test_state_only_active_restart_refused(case, tmp_path):
    grid, nl, carry = case
    with pytest.raises(ValueError, match="tracer_tendencies"):
        nc.write_wrfrst_state(carry.state, grid, nl, tmp_path / "state.nc", valid_time="2026-06-03_00:00:00",
                             run_start="2026-06-03_00:00:00", step_index=0)


def test_off_refuses_populated_state(case):
    _, _, carry = case
    fields = {name: getattr(carry.state, name) for name in carry.state.active_field_names()}
    with pytest.raises(ValueError, match="OFF"):
        validate_restart(fields, carry.tracer_tendencies, configuration(None), 1, carry.state.qv.shape)


def test_native_complete_tracer_deletion_refused(case, tmp_path):
    grid, nl, carry = case
    path = write(tmp_path / "native", "native", grid, nl, carry)
    def delete(payload, manifest):
        manifest["identity"].pop("tracer_config")
        payload.pop("tracer_schema_version")
        for key in ("tracers", "tracer_budget"):
            damage_payload(payload, key)
        payload["extra_fields"]["tracer_tendencies"] = None
    rewrite(path, delete)
    with pytest.raises(ValueError, match="tracer"):
        read(path, "native")


@pytest.mark.parametrize("reader", [cp.read_checkpoint, cp.read_checkpoint_with_land_state])
def test_checkpoint_reader_cannot_discard_runtime_carry(case, tmp_path, reader):
    grid, nl, carry = case
    path = write(tmp_path / "checkpoint", "checkpoint", grid, nl, carry)
    with pytest.raises(ValueError, match="tracer.*runtime_state"):
        reader(path)


def test_checkpoint_active_state_requires_runtime_carry(case, tmp_path):
    grid, nl, carry = case
    with pytest.raises(ValueError, match="tracer_tendencies"):
        cp.write_checkpoint(carry.state, nl, grid, 0, tmp_path / "checkpoint")


@pytest.mark.parametrize("damage", ["runtime", "embedded_state"])
def test_checkpoint_missing_or_mixed_runtime_carry_refused(case, tmp_path, damage):
    grid, nl, carry = case
    path = write(tmp_path / "checkpoint", "checkpoint", grid, nl, carry)
    payload = pickle.loads(path.read_bytes())
    if damage == "runtime":
        payload["runtime_state"] = None
    else:
        runtime = payload["runtime_state"]
        payload["runtime_state"] = runtime.replace(state=runtime.state.replace(tracers=runtime.state.tracers + np.float32(1)))
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="tracer"):
        read(path, "checkpoint")


def test_checkpoint_legacy_off_remains_readable(case, tmp_path):
    grid, nl, carry = case
    state = carry.state.replace(tracers=None, tracer_budget=None)
    nl = replace(nl, tracer_opt=0, tracer_config=None)
    path = tmp_path / "legacy-off"
    cp.write_checkpoint(state, nl, grid, 0, path)
    payload = pickle.loads(path.read_bytes())
    payload.pop("tracer_config")
    payload.pop("tracer_schema_version")
    path.write_bytes(pickle.dumps(payload))
    restored = cp.read_checkpoint(path)[0]
    exact(state, restored)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "native", "checkpoint"])
def test_actual_source_sink_continuation_uses_saved_step_clock(case, tmp_path, fmt):
    """Exercise actual sources, all sinks and restored step, not a fake recurrence.

    This is source-step/IO continuation evidence; the full coupled WRF forecast
    and its restart remain separate gates. The emitted kg check is independent
    of the product's overlap calculation.
    """
    from gpuwrf.tracers.runtime import sources_and_sinks

    grid, nl, carry = case
    # At dt=2, checkpoint17 lands at34s. One completed emission would replay
    # after a clock reset; the other straddles the checkpoint and must run once
    # on each side. All three existing loss coefficients remain nonzero.
    cfg = replace(nl.tracer_config, emissions=(
        Emission(0, 1., 3., 0, 1, 0, 1, 4., 0., True),
        Emission(0, 33., 37., 0, 1, 0, 1, 2., 0., True)))
    nl = replace(nl, dt_s=2., tracer_config=cfg)
    state = carry.state.replace(
        tracers=jnp.asarray(carry.state.tracers), tracer_budget=jnp.zeros((9, 4), jnp.float32),
        mu_total=jnp.full_like(carry.state.mu_total, 90000.),
        ph_total=jnp.broadcast_to(jnp.arange(6, dtype=jnp.float32)[:, None, None] * jnp.float32(981),
                                 carry.state.ph_total.shape),
        rain_acc=jnp.zeros_like(carry.state.rain_acc), rainc_acc=jnp.zeros_like(carry.state.rainc_acc))
    carry = carry.replace(state=state)

    def advance_sources(current, selected, step):
        before = current.state
        weather = before.replace(rain_acc=before.rain_acc + jnp.float32(.125))
        out = sources_and_sinks(weather, before, selected, (step - 1) * selected.dt_s)
        return current.replace(state=out)

    for step in range(1, 18):
        carry = advance_sources(carry, nl, step)
    path = write(tmp_path / fmt, fmt, grid, nl, carry)
    if fmt == "pickle":
        restored, restored_nl, _, saved_step = rp.read_restart(path, namelist=nl)
    elif fmt == "netcdf":
        restored, metadata = nc.read_wrfrst_carry(path, namelist=nl)
        saved_step, restored_nl = metadata["step_index"], nl
    elif fmt == "checkpoint":
        _, restored_nl, _, saved_step, restored = cp.read_checkpoint_with_runtime_state(path, namelist=nl)
    else:
        snapshot, _ = RestartStore(path.parent, 10**7, 2, 0).read(path, expected_identity=bound_identity(nl))
        restored, saved_step, restored_nl = snapshot["carries"]["d01"], snapshot["own_steps"]["d01"], nl
    assert saved_step == 17
    exact(carry, restored)
    uninterrupted = carry
    for step in range(18, 23):
        uninterrupted = advance_sources(uninterrupted, nl, step)
    resumed = restored
    for step in range(saved_step + 1, 23):
        resumed = advance_sources(resumed, restored_nl, step)
    exact(uninterrupted, resumed)
    budget = np.asarray(resumed.state.tracer_budget)
    # Two seconds at4kg/s plus four seconds at2kg/s =16kg, independent of dt.
    assert budget[8, 0] == np.float32(16)
    assert (budget[8, 1:] > 0).all(), "decay, dry deposition and wet scavenging must all execute"
    np.testing.assert_array_equal(budget[:8], 0)
    # Deletion-sensitive clock control: restarting from step1 replays the
    # completed first emission and skips the straddling emission's remainder.
    wrong_clock = restored
    for step in range(1, 6):
        wrong_clock = advance_sources(wrong_clock, restored_nl, step)
    assert not np.array_equal(wrong_clock.state.tracer_budget, resumed.state.tracer_budget)
