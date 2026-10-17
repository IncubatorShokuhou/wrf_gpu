"""Pristine WRF's mp17 alias: admission, caller configuration and restart.

No new microphysics arithmetic: all physical gates reuse the qualified mp18
source after proving exact normalization to the same static configuration.
"""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    spec = importlib.util.spec_from_file_location("legacy17_state", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_wrf_normalization_receipt():
    receipt = json.loads((ROOT / "proofs/s2_mpmore/normalization.json").read_text())
    assert hashlib.sha256(receipt["body"].encode()).hexdigest() == receipt["body_sha256"]
    rows = receipt["rows"]
    assert [r["resolved"] for r in rows[:3]] == [[18, 1, 1, 1, 2, 0]] * 3
    assert rows[0]["input_mp"] == 17 and rows[1]["input_mp"] == 18
    assert rows[2]["input_mp"] == 17 and rows[2]["conflicting_flags"]
    # The other legacy variants are scientifically different, and stay refused.
    assert all(r["resolved"] != rows[0]["resolved"] for r in rows[3:])


@pytest.mark.parametrize("replay", [False, True])
def test_shared_pipeline_binding_selects_original_wrf_alias(replay):
    from gpuwrf.io.namelist_binding import bind_namelist_options, require_namelist_honoured
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme
    f = fixture()
    grid = f._grid(ny=3, nx=4, nz=8)
    base = f._namelist(grid, ra_lw_physics=0, ra_sw_physics=0)
    raw = {"physics": {"mp_physics": [17], "cu_physics": [0], "ra_lw_physics": [0], "ra_sw_physics": [0]}}
    validate_operational_namelist(raw)
    require_namelist_honoured(raw, ("d01",), replay=replay)
    aliased = bind_namelist_options(base, raw, "d01")
    canonical = bind_namelist_options(base, {"physics": {**raw["physics"], "mp_physics": [18]}}, "d01")
    assert aliased.mp_physics == canonical.mp_physics == 18
    assert aliased == canonical
    assert classify_scheme("mp_physics", 17).status == SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified" in classify_scheme("mp_physics", 17).reason


def test_api_normalizes_before_all_operator_selection():
    from gpuwrf.contracts.physics_registry import MP_MOIST_MEMBERS, MP_NUMBER_MEMBERS
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS
    from gpuwrf.coupling.physics_dispatch import scheme_entry
    from gpuwrf.runtime import operational_mode as op
    f = fixture()
    grid = f._grid(ny=3, nx=4, nz=8)
    nml = dataclasses.replace(f._namelist(grid), mp_physics=17, moist_adv_opt=1, scalar_adv_opt=1)
    assert nml.mp_physics == 18  # Genuine deletion-sensitive normalization check.
    assert MP_SCAN_ADAPTERS[17] is MP_SCAN_ADAPTERS[18]
    entry = scheme_entry("microphysics", 17)
    canonical_entry = scheme_entry("microphysics", 18)
    assert entry.owner_module == canonical_entry.owner_module and entry.entrypoint == canonical_entry.entrypoint
    assert MP_MOIST_MEMBERS[17] == MP_MOIST_MEMBERS[18]
    assert MP_NUMBER_MEMBERS[17] == MP_NUMBER_MEMBERS[18]
    assert op._root_flow_only(nml) == op._root_flow_only(dataclasses.replace(nml, mp_physics=18))
    assert op._nested_number_scalar_species(nml) == op._nested_number_scalar_species(dataclasses.replace(nml, mp_physics=18))


def test_forced_legacy_flags_follow_original_wrf_precedence():
    from gpuwrf.io.namelist_binding import require_namelist_honoured, NamelistNotHonouredError
    raw = {"physics": {"mp_physics": [17], "nssl_2moment_on": 0, "nssl_hail_on": 0,
                       "nssl_ccn_on": 0, "nssl_density_on": 0}}
    require_namelist_honoured(raw, ("d01",))
    # mp18 does not have mp17's unconditional overrides; refuse unsupported controls.
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({"physics": {**raw["physics"], "mp_physics": [18]}}, ("d01",))
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({"physics": {"mp_physics": [17], "nssl_3moment": 1}}, ("d01",))


@pytest.mark.parametrize("mp", [17, 18])
def test_internal_nssl_parameter_overrides_are_refused(mp):
    from gpuwrf.io.namelist_binding import require_namelist_honoured, NamelistNotHonouredError
    raw = {"physics": {"mp_physics": [mp]}, "nssl_mp_params": {"irenuc": 2}}
    with pytest.raises(NamelistNotHonouredError, match="nssl_mp_params"):
        require_namelist_honoured(raw, ("d01",))
    require_namelist_honoured({**raw, "nssl_mp_params": {}}, ("d01",))
    # WRF never reads NSSL's internal group when another scheme is selected.
    require_namelist_honoured({**raw, "physics": {"mp_physics": [8]}}, ("d01",))


def test_alias_cannot_bypass_nested_or_extra_scalar_diffusion_refusal():
    from gpuwrf.io.namelist_check import refuse_single_domain_mp_nested, NotOperationallyWiredError, validate_operational_namelist, UnsupportedSchemeError
    from gpuwrf.io.namelist_binding import require_namelist_honoured, NamelistNotHonouredError
    raw = {"physics": {"mp_physics": [17, 17]}}
    with pytest.raises(NotOperationallyWiredError):
        refuse_single_domain_mp_nested(raw, 2)
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({**raw, "dynamics": {"diff_opt": 2, "km_opt": 2}}, ("d01",))
    for mp in (19, 21):
        validate_operational_namelist({"physics": {"mp_physics": [mp]}})


def test_alias_checkpoint_preserves_existing_nssl_schema(tmp_path, monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state, read_wrfrst_state
    monkeypatch.setattr(sc, "_gpu_device", lambda: jax.devices("cpu")[0])
    f = fixture()
    grid = f._grid(ny=3, nx=4, nz=8)
    nml = dataclasses.replace(f._namelist(grid), mp_physics=17)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=nml.mp_physics)
    extra = ("qh", "Nh", "Nc", "Nn", "Ns", "Ng", "qvolg", "qvolh", "hail_acc")
    state = state.replace(_cast=False, **{name: jnp.full_like(getattr(state, name), (i + 1) / 128) for i, name in enumerate(extra)})
    checkpoint = tmp_path / "alias.pkl"
    write_checkpoint(state, nml, grid, 2, checkpoint)
    restored, restored_nml, _, step = read_checkpoint(checkpoint)
    assert restored_nml.mp_physics == 18 and step == 2
    nc = tmp_path / "alias.nc"
    write_wrfrst_state(state, grid, nml, nc, valid_time="2026-07-25_18:00:36",
                       run_start="2026-07-25_18:00:00", step_index=2)
    restored_nc, _ = read_wrfrst_state(nc)
    for result in (restored, restored_nc):
        for name in extra:
            a, b = np.asarray(getattr(result, name)), np.asarray(getattr(state, name))
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
