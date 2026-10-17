"""Jensen ISHMAEL (mp_physics=55) REFL_10CM: WRF DIAG_DBZ3D=refl_10cm of every MP call.

WRF: microphysics_driver passes diag_dbz3d=refl_10cm unconditionally (no diagflag, no
do_radar_ref); mp_jensen_ishmael stores max(dbz1d,-35) on every call
(module_mp_jensen_ishmael.F:411, dbz1d from :2094/:2636-2649 before sedimentation). The
jensen_ishmael Registry package allocates refl_10cm ('hdu', not restart); REFD_MAX exists only
under compute_radar_ref (do_radar_ref=1, check_a_mundo:3495). Oracle: the unchanged module on
128 real c87 CPU-history columns (proofs/t_classd/jensen_refl_oracle).
"""
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "proofs/t_classd/jensen_refl_oracle/fixtures"
JENSEN = ROOT / "proofs/s2_mpmore/jensen55"
START = datetime(2023, 1, 15, tzinfo=timezone.utc)
# CPU-WRF c87 REFL_10CM attributes (v-prep/c87/cpu, every frame).
REFL_ATTRS = dict(FieldType=104, MemoryOrder="XYZ", description="Radar reflectivity (lamda = 10 cm)",
                  units="dBZ", stagger="", coordinates="XLONG XLAT XTIME")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle():
    manifest = json.loads((ORACLE / "manifest.json").read_text())
    assert hashlib.sha256((ORACLE / "oracle.npz").read_bytes()).hexdigest() == manifest["npz_sha256"]
    data = np.load(ORACLE / "oracle.npz")
    nz = manifest["nz"]
    fields = {n: data["inputs"][:, i * nz:(i + 1) * nz] for i, n in enumerate(manifest["input_names"])}
    dbz = manifest["output_names"].index("dbz")
    return manifest, fields, data["outputs"][:, dbz * nz:(dbz + 1) * nz]


def test_oracle_is_real_and_non_vacuous():
    manifest, fields, dbz = _oracle()
    assert manifest["columns"] == 128 and len(manifest["frames"]) == 4
    assert manifest["source_sha256"] == hashlib.sha256(
        (ROOT / "data/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F").read_bytes()).hexdigest()
    live = dbz > -35
    assert live.sum() == manifest["live_dbz_values"] >= 1000 and dbz.min() == np.float32(-35.)
    rain = (fields["qr"] > 1e-9).any(axis=1)
    ice = (fields["qi1"] + fields["qi2"] + fields["qi3"] > 1e-9).any(axis=1)
    echo = live.any(axis=1)
    # Rain-only, ice-only and mixed echo columns plus echo-free columns.
    assert (echo & rain & ice).any() and (echo & ice & ~rain).any() and (~echo).sum() >= 32


def test_literal_port_column_dbz_is_pristine_byte(tmp_path):
    from gpuwrf.physics.jensen_ishmael.column import mp_jensen_ishmael
    manifest, fields, dbz = _oracle()
    libm = _module("jensen_radar_libm", JENSEN / "libm_numpy.py")
    tables = _module("jensen_radar_tables", JENSEN / "reference_tables.py")
    with np.errstate(all="ignore"):
        got, _ = mp_jensen_ishmael(fields, tables.load(ROOT, "fp32"), np.float32(manifest["dt"]), manifest["it"],
                                   xp=libm.aggregation_backend(tmp_path))
    assert np.asarray(got["dbz"]).tobytes() == dbz.tobytes()


def test_operational_xla_column_dbz_against_pristine():
    """XLA transcendentals (log10/pow/gamma) are not glibc's: bounded, same echo mask."""
    from gpuwrf.physics.jensen_ishmael.column import step
    manifest, fields, dbz = _oracle()
    got, _ = jax.jit(lambda f: step(f, jnp.float32(manifest["dt"]), manifest["it"]))(
        {k: jnp.asarray(v) for k, v in fields.items()})
    got = np.asarray(got["dbz"])
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got > -35, dbz > -35)
    assert np.abs(got.astype(np.float64) - dbz).max() <= 1e-3


def _fixture():
    return _module("jensen_radar_rrtm", ROOT / "tests/test_rrtm_lw_operational_wiring.py")


def _state(monkeypatch, ny=4, nx=8):
    """Real c87 frame columns (the first ny*nx oracle columns) in a State."""
    from gpuwrf.contracts import state as sc
    monkeypatch.setattr(sc, "_gpu_device", lambda: jax.devices("cpu")[0])
    manifest, fields, _ = _oracle()
    f = _fixture()
    grid = f._grid(nz=manifest["nz"], ny=ny, nx=nx)
    state = f._state(grid).ensure_conditional_leaves(mp_physics=55)
    cube = lambda a: jnp.asarray(np.moveaxis(a[:ny * nx].reshape(ny, nx, -1), -1, 0))  # noqa: E731
    leaves = dict(qv="qv", qc="qc", qr="qr", Nr="nr", qi="qi1", Ni="ni1", qvoli="ai1", qaoli="ci1",
                  qi2="qi2", Ni2="ni2", qvoli2="ai2", qaoli2="ci2", qi3="qi3", Ni3="ni3", qvoli3="ai3", qaoli3="ci3")
    state = state.replace(theta=cube(fields["th"]) * (1 + np.float32(461.6 / 287.) * cube(fields["qv"])),
                          p=cube(fields["p"]), **{leaf: cube(fields[name]) for leaf, name in leaves.items()})
    return f, grid, state


def test_adapter_returns_the_kernel_dbz_in_state_layout(monkeypatch):
    from gpuwrf.physics.jensen_ishmael import adapter, column
    f, grid, state = _state(monkeypatch)
    seen = []
    real_step = column.step

    def spy(fields, dt, it):
        out = real_step(fields, dt, it)
        seen.append(out[0]["dbz"])
        return out

    monkeypatch.setattr(column, "step", spy)
    plain, water = adapter.jensen_adapter(state, 18., grid, timestep=5, return_precipitation=True)
    both, water2, radar = adapter.jensen_adapter(state, 18., grid, timestep=5, return_precipitation=True,
                                                 return_radar=True)
    only, radar2 = adapter.jensen_adapter(state, 18., grid, timestep=5, return_radar=True)
    for a, b in zip(jax.tree.leaves((plain, water)), jax.tree.leaves((both, water2)), strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert radar.dtype == jnp.float32 and radar.shape == state.qv.shape
    np.testing.assert_array_equal(radar, np.moveaxis(np.asarray(seen[1]), -1, 0))
    np.testing.assert_array_equal(radar, radar2)
    assert (np.asarray(radar) > -35).sum() > 100 and np.asarray(radar).min() == np.float32(-35.)


def test_native_kernel_is_asked_for_dbz_only_with_radar(monkeypatch):
    from gpuwrf.backends import pallas as backend
    from gpuwrf.kernels import phys_jensen
    from gpuwrf.physics.jensen_ishmael import adapter, column
    f, grid, state = _state(monkeypatch, ny=2, nx=2)
    asked = []

    def native(fields, dt, it, *, output_names=None, **kwargs):
        asked.append(tuple(output_names))
        out, precip = column.step(fields, dt, it)
        return {n: out[n] for n in output_names}, precip

    monkeypatch.setattr(backend, "cuda_native", lambda: True)
    monkeypatch.setattr(phys_jensen, "full_column", native)
    adapter.jensen_adapter(state, 18., grid, return_precipitation=True)
    _, _, radar = adapter.jensen_adapter(state, 18., grid, return_precipitation=True, return_radar=True)
    assert "dbz" not in asked[0] and asked[1] == (*asked[0], "dbz")
    assert radar.shape == state.qv.shape


def test_operational_seed_and_every_call_update(monkeypatch):
    from gpuwrf.diagnostics.online import configuration, radar_from_context
    from gpuwrf.runtime import operational_mode as op
    f, grid, state = _state(monkeypatch)
    for radar_ref in (0, 1):
        nl = dataclasses.replace(f._namelist(grid), mp_physics=55, run_physics=True, do_radar_ref=radar_ref,
                                 dt_s=18., diagnostic_history_interval=60.)
        assert configuration(nl)["do_radar_ref"] == 1
        carry = op._initial_carry_for_run(state, nl)
        assert carry.radar_reflectivity is not None, "mp55 must seed the held REFL_10CM"
        assert carry.radar_reflectivity.dtype == jnp.float32 and not np.asarray(carry.radar_reflectivity).any()
    updated, precip, context = op._apply_post_rk_microphysics(
        state, nl, return_precipitation=True, return_radar_context=True, step_index=jnp.int32(5))
    assert context.shape == state.qv.shape and (np.asarray(context) > -35).any()
    previous = jnp.full_like(context, 7.)
    # Step 5 is no history alarm: Jensen still overwrites refl_10cm on every call.
    np.testing.assert_array_equal(radar_from_context(previous, context, nl, jnp.int32(5), 0), context)
    held = np.asarray(radar_from_context(previous, context, nl, jnp.int32(5), 1))
    np.testing.assert_array_equal(held[:, 1:-1, 1:-1], np.asarray(context)[:, 1:-1, 1:-1])
    ring = np.ones(held.shape[1:], bool); ring[1:-1, 1:-1] = False
    assert (held[:, ring] == 7.).all()  # specified zone: WRF microphysics never runs there


def test_compiled_step_writes_the_call_reflectivity(monkeypatch):
    """Actual caller: one compiled _physics_boundary_step (RK stubbed) updates the radar carry."""
    from gpuwrf.physics.jensen_ishmael import adapter
    from gpuwrf.runtime import operational_mode as op
    f, grid, state = _state(monkeypatch, ny=3, nx=4)
    nl = dataclasses.replace(f._namelist(grid), mp_physics=55, run_physics=True, run_boundary=False,
                             disable_guards=True, dt_s=18., diagnostic_history_interval=60.,
                             ra_sw_physics=0, ra_lw_physics=0, bl_pbl_physics=0, cu_physics=0, sf_sfclay_physics=0)
    carry = op._initial_carry_for_run(state, nl)
    calls = []
    real = adapter.jensen_adapter

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append((kwargs, out))
        return out

    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    monkeypatch.setattr(adapter, "jensen_adapter", spy)
    monkeypatch.setattr(op, "_rk_scan_step", lambda c, *a, **kw: c)

    def run(c):
        after = op._physics_boundary_step(c, nl, jnp.int32(5), run_radiation=False)
        return after, calls[0][1]

    after, out = jax.jit(run)(carry)
    assert len(calls) == 1 and calls[0][0]["return_radar"] is True and calls[0][0]["return_precipitation"] is True
    radar = np.asarray(out[2])
    zone = op._microphysics_spec_zone(nl)
    expected = radar.copy()
    if zone:
        inner = np.zeros(radar.shape[1:], bool); inner[zone:-zone, zone:-zone] = True
        expected[:, ~inner] = np.asarray(carry.radar_reflectivity)[:, ~inner]
    np.testing.assert_array_equal(np.asarray(after.radar_reflectivity), expected)
    assert (np.asarray(after.radar_reflectivity) > -35).sum() > 50


def _writer(tmp_path, mp, *, diagnostics=None, subset=None, **namelist):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    f = _module("jensen_radar_writer", ROOT / "tests/test_m7_netcdf_writer.py")
    state, grid, nl = f.synthetic_case()
    nl.update(mp_physics=mp, nwp_diagnostics=0, **namelist)
    authority = f.writer_authority(grid, "d01")
    payload = prepare_wrfout_payload(state, grid, nl, tmp_path / f"h{mp}.nc", domain="d01",
        domain_authority=authority, valid_time=START, run_start=START, lead_hours=1.,
        full_variable_set=True, variable_subset=subset, diagnostics=diagnostics)
    write_prepared_wrfout(payload, expected_domain="d01", expected_domain_authority=authority,
                          variable_subset=subset)
    return state, payload.target


@pytest.mark.parametrize("radar_ref", (0, 1))
def test_writer_refl_from_carry_and_refd_max_only_under_do_radar_ref(tmp_path, radar_ref):
    from gpuwrf.io.netcdf_lock import Dataset
    shape = _module("jensen_radar_shape", ROOT / "tests/test_m7_netcdf_writer.py").synthetic_case()[0].qv.shape
    refl = np.linspace(-35, 40, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    _, target = _writer(tmp_path, 55, diagnostics={"REFL_10CM": refl}, do_radar_ref=radar_ref)
    with Dataset(target) as ds:
        np.testing.assert_array_equal(ds["REFL_10CM"][0], refl)
        assert {a: ds["REFL_10CM"].getncattr(a) for a in REFL_ATTRS} == REFL_ATTRS
        assert ("REFD_MAX" in ds.variables) == bool(radar_ref)
        if radar_ref:
            assert not np.asarray(ds["REFD_MAX"][:]).view(np.uint32).any()
    (tmp_path / "none").mkdir()
    _, target = _writer(tmp_path / "none", 55, do_radar_ref=radar_ref)
    with Dataset(target) as ds:
        assert "REFL_10CM" not in ds.variables  # never fabricated without the held source


def test_ntu_writes_prognostic_hail(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.config.paths import wrf_root
    import re
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    f = _module("jensen_radar_writer2", ROOT / "tests/test_m7_netcdf_writer.py")
    state, grid, nl = f.synthetic_case()
    qh = np.linspace(0, 1e-3, state.qv.size, dtype=np.float32).reshape(state.qv.shape)
    nl.update(mp_physics=56)
    state.qh = qh
    authority = f.writer_authority(grid, "d01")
    payload = prepare_wrfout_payload(state, grid, nl, tmp_path / "ntu.nc", domain="d01", domain_authority=authority,
        valid_time=START, run_start=START, lead_hours=1., full_variable_set=True)
    write_prepared_wrfout(payload, expected_domain="d01", expected_domain_authority=authority)
    original = (wrf_root() / "Registry/Registry.EM_COMMON").read_text()
    description, units = re.search(r'"QHAIL"\s+"([^"]*)"\s+"([^"]*)"', original).groups()
    with Dataset(payload.target) as ds:
        assert "QHAIL" in ds.variables, "NTU prognostic hail missing from history"
        np.testing.assert_array_equal(ds["QHAIL"][0], qh)
        assert ds["QHAIL"].description == description and ds["QHAIL"].units == units
        assert ds["QHAIL"].MemoryOrder == "XYZ" and ds["QHAIL"].dtype == np.float32
        assert "REFL_10CM" not in ds.variables  # mp_ntu has no reflectivity argument


def test_native_restart_requires_the_held_jensen_radar():
    from gpuwrf.diagnostics.online import configuration, validate_restart
    from types import SimpleNamespace
    cfg = configuration(SimpleNamespace(mp_physics=55, do_radar_ref=0, nwp_diagnostics=0,
                                        diagnostic_history_interval=60.))
    with pytest.raises(ValueError, match="E78"):
        validate_restart(None, None, cfg, 1, (4, 3, 2))
    validate_restart(None, jnp.zeros((4, 3, 2), jnp.float32), cfg, 1, (4, 3, 2))
