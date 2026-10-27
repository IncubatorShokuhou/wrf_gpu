"""RUC init/diagnostic pristine-oracle parity and CLI/history admission seams."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.physics.ruc_diagnostics import sfcdiags_ruclsm

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs/o2_ruccli/init_diag_savepoints.json"


@pytest.mark.parametrize("precision,rtol", [("fp64", 3.e-14), ("fp32", 3.e-6)])
def test_sfcdiags_pristine_all_regimes(precision, rtol):
    rows = np.asarray(json.loads(SAVE.read_text())[precision]["diagnostics"])
    names = ("hfx", "qfx", "tsk", "qsfc", "cqs", "cqs2", "chs2", "t1", "qv1", "rho1", "p1", "psfc")
    dtype = jnp.float64 if precision == "fp64" else jnp.float32
    inputs = {name: jnp.asarray(rows[:, k], dtype) for k, name in enumerate(names)}
    out = jax.jit(sfcdiags_ruclsm)(**inputs)
    for k, name in enumerate(("t2", "th2", "q2")):
        actual, expected = np.asarray(out[name]), rows[:, 12 + k]
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1.e-15)
        assert actual.dtype == np.dtype(precision.replace("fp", "float"))
    assert np.count_nonzero(rows[:, 0] > 0) and np.count_nonzero(rows[:, 0] < 0)
    assert np.count_nonzero(rows[:, 7] < 273.15) and np.count_nonzero(rows[:, 7] > 273.15)
    assert np.count_nonzero(rows[:, 5] < 1.e-5) and np.count_nonzero(rows[:, 6] < 1.e-5)


def test_ruc_diagnostics_has_no_fp64_in_real_program():
    row = np.asarray(json.loads(SAVE.read_text())["fp32"]["diagnostics"])[-1]
    names = ("hfx", "qfx", "tsk", "qsfc", "cqs", "cqs2", "chs2", "t1", "qv1", "rho1", "p1", "psfc")
    inputs = {name: jnp.asarray([row[k]], jnp.float32) for k, name in enumerate(names)}
    hlo = jax.jit(sfcdiags_ruclsm).lower(**inputs).compiler_ir("hlo").as_hlo_text()
    assert "f64[" not in hlo


@pytest.mark.parametrize("precision,rtol", [("fp64", 2.e-14), ("fp32", 2.e-6)])
@pytest.mark.parametrize("lu", [1, 2])
@pytest.mark.parametrize("nzs", [6, 9])
def test_ruclsminit_pristine(precision, rtol, lu, nzs):
    from gpuwrf.physics.ruclsm import load_ruc_tables, ruclsminit
    rows = np.asarray(json.loads(SAVE.read_text())[precision]["init"])
    rows = rows[(rows[:, 0] == lu) & (rows[:, 1] == nzs)].reshape(12, nzs, -1)
    tab = load_ruc_tables("USGS-RUC" if lu == 1 else "MODI-RUC")
    out = ruclsminit(tab, tslb=rows[:, :, 5], smois=rows[:, :, 6],
                     isltyp=rows[:, 0, 4], ivgtyp=np.full(12, 7), xice=rows[:, 0, 7],
                     dtype=precision.replace("fp", "float"))
    for name, expected in (("sh2o", rows[:, :, 8]), ("smfr3d", rows[:, :, 9]),
                           ("mavail", rows[:, 0, 10]), ("znt", rows[:, 0, 11])):
        assert np.all(np.isfinite(out[name])) and np.all(np.isfinite(expected))
        np.testing.assert_allclose(out[name], expected, rtol=rtol, atol=1.e-15)


def _input(path, nzs=6):
    """Small file for loader/schema tests, not WRF fidelity evidence."""
    from gpuwrf.io.ruc_land_init import RUC_SOIL_LEVELS
    with Dataset(path, "w") as ds:
        for name, size in (("Time", 1), ("soil_layers_stag", nzs), ("south_north", 2), ("west_east", 3), ("DateStrLen", 19)):
            ds.createDimension(name, size)
        ds.MMINLU = "USGS"; ds.ISWATER = 16; ds.ISICE = 24; ds.CEN_LAT = 45.; ds.SF_SURFACE_PHYSICS = 3
        ts = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        ts[0] = np.frombuffer(b"2023-01-15_00:00:00", "S1")
        dims = ("Time", "south_north", "west_east")
        for name, val in {"IVGTYP": 7, "LU_INDEX": 7, "ISLTYP": 3, "XLAND": 1., "TMN": 285.,
                          "SHDMIN": 5., "SHDMAX": 80., "ALBBCK": .18, "VEGFRA": 50.,
                          "TSK": 288., "SNOALB": .7, "SEAICE": 0., "LAKEMASK": 0.}.items():
            ds.createVariable(name, "f4", dims)[:] = val
        ds.createVariable("ZS", "f4", ("Time", "soil_layers_stag"))[:] = RUC_SOIL_LEVELS.get(nzs, (.05, .25, .70, 1.5))
        for name, val in (("TSLB", 286.), ("SMOIS", .25)):
            ds.createVariable(name, "f4", ("Time", "soil_layers_stag", "south_north", "west_east"))[:] = val


@pytest.mark.parametrize("nzs", [6, 9])
def test_loader_uses_prepared_ruc_soil_and_geometry(tmp_path, nzs):
    from gpuwrf.io.ruc_land_init import load_ruc_land, RUC_SOIL_LEVELS
    from gpuwrf.coupling.ruc_surface_hook import ruc_history_fields
    from gpuwrf.io.wrfout_writer import _dimension_sizes
    _input(tmp_path / "wrfinput_d01", nzs)
    static, land = load_ruc_land(tmp_path, "d01", num_soil_layers=nzs, dt=18., p1=np.full((2, 3), 90000.))
    assert land.tso.shape == (2, 3, nzs)
    np.testing.assert_array_equal(land.tso, np.full((2, 3, nzs), 286., np.float32))
    np.testing.assert_array_equal(land.qvg, np.zeros((2, 3), np.float32))
    fields = ruc_history_fields(land, static)
    np.testing.assert_array_equal(fields["ZS"], np.asarray(RUC_SOIL_LEVELS[nzs], np.float32))
    assert fields["TSLB"].shape == (nzs, 2, 3)
    nml = SimpleNamespace(sf_surface_physics=3, ruc_static=static)
    assert _dimension_sizes(nx=3, ny=2, nz=4, namelist=nml)["soil_layers_stag"] == nzs


@pytest.mark.parametrize("bad,pattern", [("levels", "num_soil_layers"), ("zs", "RUC ZS"),
                                          ("scheme", "SF_SURFACE_PHYSICS"), ("ice", "sea-ice"),
                                          ("lake", "lake"), ("category", "invalid RUC"),
                                          ("nan", "nonfinite")])
def test_loader_refuses_incompatible_input(tmp_path, bad, pattern):
    from gpuwrf.io.ruc_land_init import load_ruc_land
    p = tmp_path / "wrfinput_d01"
    _input(p)
    with Dataset(p, "a") as ds:
        if bad == "zs": ds["ZS"][0, 0] = .05
        if bad == "scheme": ds.SF_SURFACE_PHYSICS = 4
        if bad == "ice": ds["SEAICE"][0, 0, 0] = .3
        if bad == "lake": ds["LAKEMASK"][0, 0, 0] = 1.
        if bad == "category": ds["ISLTYP"][0, 0, 0] = 0
        if bad == "nan": ds["TSLB"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=pattern):
        load_ruc_land(tmp_path, "d01", num_soil_layers=4 if bad == "levels" else 6,
                       dt=18., p1=np.full((2, 3), 90000.))


def test_cli_honour_check_allows_ruc():
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.integration.nested_pipeline import _domain_sf_surface_physics
    nml = {"physics": {"sf_surface_physics": [3], "sf_sfclay_physics": [5], "mp_physics": [8],
                       "bl_pbl_physics": [5], "num_soil_layers": 6}, "domains": {"max_dom": 1}}
    require_namelist_honoured(nml, ("d01",))
    assert _domain_sf_surface_physics(SimpleNamespace(namelist=nml), "d01") == 3


def test_daily_keeps_carry_and_global_step(monkeypatch):
    from gpuwrf.integration.daily_pipeline import _RucDailyStepper
    import gpuwrf.integration.daily_pipeline as daily
    import gpuwrf.runtime.operational_mode as om
    from dataclasses import dataclass, replace
    @dataclass(frozen=True)
    class Carry:
        state: object
        land_steps: int = 0
        def replace(self, **kwargs): return replace(self, **kwargs)
    state = SimpleNamespace(theta=jnp.ones((1,)))
    calls = []
    monkeypatch.setattr(om, "_resolve_operational_suite", lambda n: None)
    monkeypatch.setattr(om, "_committed_initial_carry_for_run", lambda s, n: Carry(s))
    monkeypatch.setattr(om, "build_clock_base", lambda n: None)
    monkeypatch.setattr(daily, "_commit_to_operational_device", lambda s: s)
    def advance(c, nml, start, clock, *, n_steps, cadence):
        calls.append((int(start), n_steps, c.land_steps))
        return c.replace(land_steps=c.land_steps + n_steps)
    monkeypatch.setattr(om, "_advance_chunk", advance)
    nml = SimpleNamespace(dt_s=10., radiation_cadence_steps=2)
    fn = _RucDailyStepper(state, nml)
    fn(state, nml, 30. / 3600.)
    fn(state, nml, 20. / 3600.)
    assert calls == [(1, 2, 0), (3, 1, 2), (4, 2, 3)]
    assert fn.next_step == 6 and fn.carry.land_steps == 5


@pytest.mark.parametrize("nzs", [6, 9])
def test_ruc_writer_and_full_restart_keep_soil_and_held_diagnostics(tmp_path, nzs):
    from datetime import datetime, timezone
    from gpuwrf.io.ruc_land_init import load_ruc_land
    from gpuwrf.coupling.ruc_surface_hook import ruc_history_fields, RucRadiation
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.io.restart import write_restart, read_restart
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, bind_wrfout_domain_authority, write_prepared_wrfout
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    from test_m7_netcdf_writer import authenticated_source_grid
    _input(tmp_path / "wrfinput_d01", nzs)
    static, land = load_ruc_land(tmp_path, "d01", num_soil_layers=nzs, dt=18., p1=np.full((2, 3), 90000.))
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    state = build_neutral_state(grid)
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18.,
                              sf_surface_physics=3, ruc_static=static, ruc_land=land)
    # Product initializer seeding (operational_mode._initial_carry_for_run): static-scheme
    # conditional leaves, incl. the held MP_RE effective radii when mp_re_active(nml).
    state = state.ensure_conditional_leaves(mp_physics=int(nml.mp_physics), use_mp_re=int(nml.use_mp_re),
                                            ra_lw_physics=int(nml.ra_lw_physics),
                                            ra_sw_physics=int(nml.ra_sw_physics))
    carry = initial_operational_carry(state, ruc_land=land, ruc_rad=RucRadiation(jnp.ones((2, 3)), jnp.ones((2, 3))))
    history = seed_history_diagnostics(state)
    history = history.write({name: np.full((2, 3), i + 1.) for i, name in enumerate(history._fields)})
    carry = carry.replace(history_diagnostics=history)
    path = tmp_path / "restart.pkl"
    write_restart(carry, nml, grid, 17, path)
    restored, restored_nml, _, step = read_restart(path)
    assert step == 17 and restored_nml.ruc_static.cfg.nzs == nzs
    for group in ("ruc_land", "ruc_rad", "history_diagnostics"):
        a, b = getattr(carry, group), getattr(restored, group)
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            assert x.dtype == y.dtype
            np.testing.assert_array_equal(x, y)
    auth = bind_wrfout_domain_authority("d01", authenticated_source_grid(grid, "d01"), grid)
    clock = datetime(2023, 1, 15, tzinfo=timezone.utc)
    diagnostics = dict(ruc_history_fields(land, static), T2=np.full((2, 3), 271.), Q2=np.full((2, 3), .003),
                       HFX=np.zeros((2, 3)), LH=np.zeros((2, 3)))
    prepared = prepare_wrfout_payload(state, grid, nml, tmp_path / "wrfout", domain="d01", domain_authority=auth,
                                     valid_time=clock, run_start=clock, lead_hours=1., diagnostics=diagnostics,
                                     full_variable_set=True)
    write_prepared_wrfout(prepared, expected_domain="d01", expected_domain_authority=auth)
    with Dataset(tmp_path / "wrfout") as ds:
        assert len(ds.dimensions["soil_layers_stag"]) == nzs
        for name, value in diagnostics.items():
            np.testing.assert_array_equal(ds[name][0], value.astype(np.float32))
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    write_wrfrst_carry(carry, grid, nml, tmp_path / "wrfrst", valid_time=clock,
                       run_start=clock, step_index=17)
    restored_nc, _ = read_wrfrst_carry(tmp_path / "wrfrst")
    for group in ("ruc_land", "ruc_rad", "history_diagnostics"):
        for x, y in zip(jax.tree.leaves(getattr(carry, group)), jax.tree.leaves(getattr(restored_nc, group)), strict=True):
            assert x.dtype == y.dtype
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("nzs", [6, 9])
def test_written_ruc_th2_matches_pristine_sfcdiags(tmp_path, nzs):
    """Nonzero-lead NetCDF TH2 must use held T2 and WRF's PSFC2D, not P3D.

    The reference is the unmodified REAL=4 SFCDIAGS_RUCLSM output, covering
    every staged warm/cold and exchange-coefficient regime. A poisoned TH2
    diagnostic ensures this exercises the writer's reconstruction itself.
    """
    from datetime import datetime, timedelta, timezone
    from gpuwrf.io.ruc_land_init import RUC_SOIL_LEVELS
    from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority, write_wrfout_netcdf
    from gpuwrf.physics.ruclsm import RucConfig
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    from test_m7_netcdf_writer import authenticated_source_grid

    rows = np.asarray(json.loads(SAVE.read_text())["fp32"]["diagnostics"], np.float32)
    assert rows.shape == (12, 15)
    p1, psfc, t2, oracle_th2, q2 = (rows[:, k].reshape(3, 4) for k in (10, 11, 12, 13, 14))
    assert np.all(psfc != p1), "pressure-source mutant would be vacuous"
    grid = build_flat_grid(nx=4, ny=3, nz=3, dx_m=3000.)
    state = build_neutral_state(grid)
    history = seed_history_diagnostics(state).write(dict(
        t2=t2, q2=q2, psfc=psfc, u10=np.zeros((3, 4)), v10=np.zeros((3, 4)),
        lh=np.zeros((3, 4)), hfx=rows[:, 0].reshape(3, 4),
    ))
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None,
                              sf_surface_physics=3,
                              ruc_static=SimpleNamespace(cfg=RucConfig(dt=18., zs=RUC_SOIL_LEVELS[nzs])))
    auth = bind_wrfout_domain_authority("d01", authenticated_source_grid(grid, "d01"), grid)
    start = datetime(2023, 1, 15, tzinfo=timezone.utc)
    target = tmp_path / "wrfout_th2"
    write_wrfout_netcdf(
        state, grid, nml, target, domain="d01", domain_authority=auth,
        run_start=start, valid_time=start + timedelta(hours=1), lead_hours=1.,
        diagnostics=dict(T2=history.t2, Q2=history.q2, PSFC=history.psfc,
                         HFX=state.hfx, LH=history.lh, TH2=np.full((3, 4), -999., np.float32)),
    )
    with Dataset(target) as ds:
        written = np.asarray(ds["TH2"][0])
        assert written.dtype == np.float32 and np.all(np.isfinite(written))
        np.testing.assert_array_equal(ds["T2"][0], t2)
        np.testing.assert_array_equal(ds["PSFC"][0], psfc)
        # NumPy's REAL power and gfortran/libm may differ by a last bit.
        np.testing.assert_array_max_ulp(written, oracle_th2, maxulp=2)


def test_oracle_savepoint_loads_from_clean_git_archive(tmp_path):
    import io
    import subprocess
    import tarfile
    archive = subprocess.run(["git", "archive", "HEAD", "proofs/o2_ruccli/init_diag_savepoints.json"],
                             cwd=ROOT, check=True, capture_output=True, timeout=10).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path, filter="data")
    source = (tmp_path / "proofs/o2_ruccli/init_diag_savepoints.json").read_bytes()
    assert source == SAVE.read_bytes()
    data = json.loads(source)
    assert len(data["fp32"]["diagnostics"]) == len(data["fp64"]["diagnostics"]) == 12


@pytest.mark.parametrize("key,value", [("num_soil_layers", 4), ("mp_physics", 6),
                                       ("sf_sfclay_physics", 1), ("sf_lake_physics", 1)])
def test_cli_refuses_ruc_incompatible_options_before_jax(key, value):
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nml = {"physics": {"sf_surface_physics": [3], "sf_sfclay_physics": [5], "mp_physics": [8],
                       "num_soil_layers": 6}, "domains": {"max_dom": 1}}
    nml["physics"][key] = value
    errors, _ = namelist_honour_report(nml, ("d01",))
    assert any("RUC requires " + key in e for e in errors)


@pytest.mark.parametrize("pipeline", ["daily", "nested"])
def test_both_pipeline_loaders_bind_ruc_bundles(monkeypatch, tmp_path, pipeline):
    from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state, _zero_tendencies
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_m7_netcdf_writer import authenticated_source_grid
    import gpuwrf.io.lower_boundary as lower
    _input(tmp_path / "wrfinput_d01")
    (tmp_path / "namelist.input").write_text(
        "&domains\n max_dom=1, time_step=18,\n/\n&physics\n sf_surface_physics=3, "
        "num_soil_layers=6, sf_sfclay_physics=5, mp_physics=8, bl_pbl_physics=5, cu_physics=0,\n/\n")
    run = Gen2Run(tmp_path)
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    source = authenticated_source_grid(grid, "d01")
    run.grid = lambda domain: source
    state = build_neutral_state(grid)
    case = SimpleNamespace(run=run, grid=grid, metrics=grid.metrics, state=state,
                           tendencies=_zero_tendencies(grid),
                           metadata={"run_start_label": "2023-01-15_00:00:00", "standalone_native_init": True})
    mod = daily if pipeline == "daily" else nested
    monkeypatch.setattr(mod, "build_replay_case", lambda *a, **kw: case)
    monkeypatch.setattr(mod, "load_radiation_static", lambda *a, **kw: (None, {}))
    monkeypatch.setattr(lower, "load_lower_boundary", lambda *a, **kw: None)
    if pipeline == "daily":
        monkeypatch.setattr(daily, "dealias_state_buffers", lambda s: s)
        monkeypatch.setattr(daily, "_load_static_latlon_writer_diagnostics", lambda *a, **kw: (None, {}))
        loaded, _ = daily._build_real_case(daily.DailyPipelineConfig(run_id=str(tmp_path), domain="d01"))
        nml = loaded.namelist
    else:
        monkeypatch.setattr(nested, "_root_boundary_cadence_override", lambda n, *a: n)
        monkeypatch.setattr(nested, "_commit_to_operational_device", lambda c: c)
        monkeypatch.setattr(nested, "_initial_carry_for_run", lambda s, n: initial_operational_carry(s, ruc_land=n.ruc_land))
        cfg = nested.NestedPipelineConfig(tmp_path, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=1)
        _, bundles, _, _, _, carries = nested._load_domains(cfg, ("d01",))
        nml = bundles["d01"].namelist
        assert carries["d01"].ruc_land is not None
    assert nml.sf_surface_physics == 3 and not nml.use_noahmp
    assert nml.ruc_static.cfg.nzs == 6 and nml.ruc_land.tso.shape == (2, 3, 6)


def test_surface_history_uses_post_ruc_mavail_and_first_step(monkeypatch, tmp_path):
    from gpuwrf.io.ruc_land_init import load_ruc_land
    import gpuwrf.coupling.ruc_surface_hook as hook
    import importlib
    sl = importlib.import_module("gpuwrf.physics.surface_layer")
    import gpuwrf.physics.ruc_diagnostics as rd
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    _input(tmp_path / "wrfinput_d01")
    static, land = load_ruc_land(tmp_path, "d01", num_soil_layers=6, dt=18., p1=np.full((2, 3), 90000.))
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    state = build_neutral_state(grid)
    captured = {}
    real_surface = sl.surface_layer_with_exchange
    def surface(*a, **kw):
        out = real_surface(*a, **kw)
        captured["flqc"] = out[2]
        return out
    def lsm(st, forcing, statics, cfg, tables, ktau, **kw):
        captured["ktau"] = int(ktau)
        captured["rho"] = forcing["rho3d"].reshape((2, 3))
        out = dict(st)
        out["mavail"] = jnp.full_like(st["mavail"], .15)
        return out
    real_diag = rd.sfcdiags_ruclsm
    def diag(**kw):
        captured["cqs"] = kw["cqs"]
        return real_diag(**kw)
    monkeypatch.setattr(sl, "surface_layer_with_exchange", surface)
    monkeypatch.setattr(hook, "lsmruc_step", lsm)
    monkeypatch.setattr(rd, "sfcdiags_ruclsm", diag)
    # This fixture has an analytic State but no WRF phy_prep metric carry.
    # Exercise the column seam directly; real-case/grid coupling is the smoke.
    out, evolved, history = hook.ruc_surface_step(state, land, static, 18., None, first_timestep=True, history=True)
    assert captured["ktau"] == 1
    expected = np.asarray(captured["flqc"], np.float32) / (np.float32(.15) * np.asarray(captured["rho"], np.float32))
    np.testing.assert_allclose(captured["cqs"], expected, rtol=1.e-6)
    assert set(("t2", "th2", "q2", "u10", "v10", "psfc", "lh", "hfx")) <= history.keys()
    assert np.all(np.isfinite(history["t2"])) and np.all(np.isfinite(history["q2"]))


def test_ruc_radiation_uses_step_start_land_albedo_and_emissivity(monkeypatch):
    import gpuwrf.runtime.operational_mode as om
    from gpuwrf.coupling.physics_couplers import _surface_radiation_properties
    from gpuwrf.coupling.ruc_surface_hook import ruc_radiation_surface
    from dataclasses import dataclass, replace
    land = SimpleNamespace(alb=jnp.asarray([[.65, .11]], jnp.float32),
                           emiss=jnp.asarray([[.96, .98]], jnp.float32))
    state = SimpleNamespace(lu_index=jnp.ones((1, 2)), xland=jnp.ones((1, 2)), t_skin=jnp.ones((1, 2)))
    proxy = ruc_radiation_surface(land)
    assert proxy.albedo is land.alb and proxy.emiss is land.emiss
    a, e = _surface_radiation_properties(state, proxy)
    np.testing.assert_array_equal(a, land.alb)
    np.testing.assert_array_equal(e, land.emiss)
    @dataclass(frozen=True)
    class Carry:
        state: object
        ruc_land: object
        rthraten: object = None
        radiation_diagnostics: object = None
        census: object = None
        noahmp_land: object = None
        solar_interpolation: object = None
        def replace(self, **kw): return replace(self, **kw)
    captured = {}
    def radiation(state, grid, **kw):
        captured.update(kw)
        return None, None
    monkeypatch.setattr(om, "rrtmg_theta_tendency", radiation)
    nml = SimpleNamespace(sf_surface_physics=3, bl_pbl_physics=5, radiation_interval_s=1800.,
                          grid=None, time_utc=None, radiation_static=None, topo_shading=0,
                          slope_rad=0, topo_shadow_length_m=0., mp_physics=8, use_mp_re=0,
                          sf_urban_physics=0, cu_physics=0, ra_sw_physics=4, ra_lw_physics=4)
    om._refresh_rrtmg_driver(Carry(state, land), nml, 18., True, None)
    assert captured["land_state"].albedo is land.alb
    assert captured["land_state"].emiss is land.emiss
