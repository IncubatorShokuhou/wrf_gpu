"""Daily host-loop lifecycle regressions, not full forecast fidelity claims.

The recording chunk in the fast tests replaces model arithmetic only. The actual
daily loop, retained stepper, output routing and full-carry restart run unchanged.
The final test also runs the real observation driver on pristine Swiss operands.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation.obs_driver import initial_observation_carry, observation_driver
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.coupling.boundary_apply import BoundaryConfig, interpolate_boundary_leaf
from gpuwrf.integration import daily_pipeline as daily, nested_pipeline
from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
from gpuwrf.io import restart
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_m7_daily_pipeline import _synthetic_case
from test_m7_restart_checkpoint_roundtrip import _synthetic_state, _synthetic_tendencies
from test_obs_operational_io import _inputs
from test_obs_full_coupled import ROOT, prepared_inputs


def _output_hooks(monkeypatch):
    monkeypatch.setattr(daily, "write_wrfout_netcdf", lambda *a, **k: None)
    monkeypatch.setattr(daily, "prepare_wrfout_payload", lambda *a, **k: object())
    monkeypatch.setattr(daily, "_emit_auxhist_frame", lambda **k: k["output_dir"] / str(k["lead_minutes"]))
    monkeypatch.setattr(daily, "_commit_to_operational_device", lambda s: s)
    monkeypatch.setattr(om, "_resolve_operational_suite", lambda n: None)
    monkeypatch.setattr(om, "build_clock_base", lambda n: None)


def _same_leaves(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert (x.shape, x.dtype, x.tobytes()) == (y.shape, y.dtype, y.tobytes())


@pytest.mark.parametrize("aux_minutes", [None, 15])
@pytest.mark.parametrize("checkpoint_hour", [None, 1])
def test_daily_obs_carry_clock_outputs_and_real_checkpoint(monkeypatch, tmp_path, aux_minutes, checkpoint_hour):
    _output_hooks(monkeypatch)
    grid = GridSpec.canary_3km_template()
    # Nonlinear three-record forcing detects re-windowing with a global clock.
    # The leaf keeps the State boundary contract (time, side, bdy_width, z, side_len)
    # (contracts/state.py boundary_mass) that the native REAL boundary interpolation
    # (GPUWRF_BOUNDARY_FP32=1, kernels/dyn_boundary_fp32.interpolate_leaf) requires.
    state = _synthetic_state(grid)
    boundary = jnp.broadcast_to(jnp.asarray([0., 10., 40.], jnp.float32)[:, None, None, None, None],
                                (3, *state.u_bdy.shape[1:]))
    state = state.replace(u_bdy=boundary)
    inputs = replace(_inputs(), options=tuple(sorted(dict(_inputs().options, obs_ionf=137).items())))
    nl = om.OperationalNamelist(grid=grid, tendencies=_synthetic_tendencies(grid), metrics=grid.metrics,
        dt_s=18., radiation_cadence_steps=200, use_mp_re=0, sf_surface_physics=4,
        observation_nudging=inputs, boundary_config=BoundaryConfig(update_cadence_s=3600.))
    case = replace(_synthetic_case(tmp_path), state=state, grid=grid, namelist=nl,
                   metadata={"boundary": {"interval_seconds": 21600.}})
    calls, seeds, diagnostics, checkpoints = [], [], [], []

    def seed(s, n):
        seeds.append(1)
        return initial_operational_carry(s).replace(
            obs=initial_observation_carry(n.observation_nudging, s.theta.shape),
            rthraten=jnp.full_like(s.theta, .125), noahmp_rad=(jnp.asarray(123., jnp.float32),))

    def advance(c, n, start, clock, *, n_steps, cadence):
        start = int(start)
        calls.append((start, int(c.obs.read_step)))
        # The full time axis and original record cadence must survive every call.
        np.testing.assert_array_equal(np.asarray(c.state.u_bdy), np.asarray(boundary))
        assert n.boundary_config.update_cadence_s == 21600.
        lead = (start - 1) * n.dt_s
        expected = lead / 21600. * 10.
        np.testing.assert_allclose(interpolate_boundary_leaf(c.state.u_bdy, lead, 21600.), expected)
        read = ((start + n_steps - 2) // 137) * 137
        return c.replace(obs=c.obs.replace(read_step=jnp.int32(read), count=jnp.int32(1),
            varobs=jnp.full_like(c.obs.varobs, read + .25),
            rko=jnp.full_like(c.obs.rko, 2.5), savwt_t=jnp.full_like(c.obs.savwt_t, read + 1)))

    def history(s, n, run_start, **kw):
        # Actual daily diagnostics must reach the Noah-MP history path, with
        # the held land/radiation/history, rather than the RUC-only helper.
        diagnostics.append(kw)
        assert kw["noahmp_land"] is None
        np.testing.assert_array_equal(kw["noahmp_rad"][0], np.float32(123.))
        return {"T2": np.full((grid.ny, grid.nx), 289., np.float32)}

    write_real, read_real = restart.write_restart, restart.read_restart
    def write(c, n, g, step, path, **kw):
        write_real(c, n, g, step, path, **kw)
        restored, restored_n, _, restored_step = read_real(path)
        _same_leaves(c, restored)
        _same_leaves(n.observation_nudging, restored_n.observation_nudging)
        assert restored_step == step == 200
        assert int(restored.obs.read_step) == 137
        checkpoints.append(Path(path))

    monkeypatch.setattr(om, "_committed_initial_carry_for_run", seed)
    monkeypatch.setattr(om, "_advance_chunk", advance)
    monkeypatch.setattr(nested_pipeline, "_noahmp_surface_diagnostics_for_output", history)
    monkeypatch.setattr(restart, "write_restart", write)
    monkeypatch.setattr(daily, "write_checkpoint", lambda *a, **k: pytest.fail("State-only obs checkpoint"))
    cfg = daily.DailyPipelineConfig(hours=2, dt_s=18., async_output=False, score=False,
        refresh_land_state_hourly=False,
        auxhist=AuxhistStreamConfig(interval_minutes=aux_minutes) if aux_minutes else None)
    result = daily._run_forecast_sequence(cfg, output_dir=tmp_path / "out",
        checkpoint_at_hour=checkpoint_hour, forecast_fn=daily._segmented_forecast_fn,
        case_builder=lambda c: (case, tmp_path))
    stride = 50 if aux_minutes else 200
    assert [s for s, _ in calls] == list(range(1, 401, stride))
    assert calls[200 // stride] == (201, 137)  # second hour is an ionf HOLD
    assert seeds == [1]
    assert len(diagnostics) == 400 // stride
    assert len(checkpoints) == int(checkpoint_hour is not None)
    if checkpoint_hour:
        assert result.checkpoint["step_index"] == 200


def test_obs_daily_diagnostics_pass_noahmp_carry(monkeypatch):
    land, rad, radiation, history = object(), object(), object(), object()
    carry = SimpleNamespace(noahmp_land=land, noahmp_rad=rad,
                            radiation_diagnostics=radiation, history_diagnostics=history)
    recorded = []
    def output(*args, **kw):
        recorded.append(kw)
        return {"T2": np.ones((2, 3), np.float32)}
    monkeypatch.setattr(nested_pipeline, "_noahmp_surface_diagnostics_for_output", output)
    values = daily._surface_diagnostics_for_output(object(), SimpleNamespace(grid=object()), object(),
        lead_seconds=3600., operational_carry=carry)
    assert values["T2"].shape == (2, 3)
    assert recorded[0]["noahmp_land"] is land and recorded[0]["noahmp_rad"] is rad
    assert recorded[0]["radiation_diagnostics"] is radiation and recorded[0]["history_diagnostics"] is history


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("aux_minutes", [None, 15])
def test_daily_initialized_obs_writes_actual_output_consumer(monkeypatch, tmp_path, with_history, aux_minutes):
    """Real obs initialization, daily routing, consumers and NetCDF output.

    Only model stepping and expensive diagnostic arithmetic are stubs. The
    original crash must propagate through the unmodified output consumer here.
    Q2 comes from bulk diagnostics without history and held Q2 with history.
    """
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    from test_m7_netcdf_writer import authenticated_source_grid

    grid = GridSpec.canary_3km_template()
    state = _synthetic_state(grid)
    nl = om.OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=_synthetic_tendencies(grid),
        dt_s=18., radiation_cadence_steps=200, observation_nudging=_inputs(),
        use_mp_re=0, mp_physics=0, cu_physics=0, ra_sw_physics=0, ra_lw_physics=0,
        rad_rk_tendf=1, run_physics=True, sf_surface_physics=4, bl_pbl_physics=5, sf_sfclay_physics=5)
    case = replace(_synthetic_case(tmp_path), state=state, grid=grid, namelist=nl,
        writer_domain_authority=bind_wrfout_domain_authority("d02", authenticated_source_grid(grid, "d02"), grid))
    bulk_q2 = jnp.full_like(state.t_skin, .007, dtype=jnp.float32)
    held_q2 = jnp.full_like(state.t_skin, .013, dtype=jnp.float32)
    land = SimpleNamespace(soil_marker=jnp.asarray(17., jnp.float32))
    rad = tuple(jnp.asarray(x, jnp.float32) for x in (100., 200., .5))
    radiation = SimpleNamespace(marker=jnp.asarray(23., jnp.float32))
    calls, initializations, bulk_calls = [], [], []

    def advance(c, n, start, clock, *, n_steps, cadence):
        assert c.obs is not None  # actual _committed_initial_carry_for_run seeded it
        if int(start) == 1:
            assert c.history_diagnostics is None  # reproduce ordinary obs initialization
            initializations.append(1)
        history = seed_history_diagnostics(c.state)._replace(q2=held_q2) if with_history else None
        return c.replace(noahmp_land=land, noahmp_rad=rad, radiation_diagnostics=radiation,
                         history_diagnostics=history)

    def m9(s, n, lead, **kw):
        assert kw["noahmp_land"] is land and kw["noahmp_rad"] is rad
        assert kw["radiation_diagnostics"] is radiation
        assert (kw.get("history_diagnostics") is not None) == with_history
        calls.append(float(lead))
        return SimpleNamespace(**{attr: jnp.ones_like(s.t_skin) for _, attr in daily._M9_OUTPUT_FIELDS if attr},
                               hfx=jnp.ones_like(s.t_skin), lh=jnp.ones_like(s.t_skin))

    def bulk(s, g):
        bulk_calls.append(1)
        return SimpleNamespace(q2=bulk_q2)

    monkeypatch.setenv("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY", "0")
    monkeypatch.setattr(om, "_resolve_operational_suite", lambda n: None)
    monkeypatch.setattr(om, "_advance_chunk", advance)
    monkeypatch.setattr(om, "compute_m9_diagnostics", m9)
    monkeypatch.setattr(om, "surface_layer_diagnostics", bulk)
    cfg = daily.DailyPipelineConfig(hours=2, dt_s=18., refresh_land_state_hourly=False,
        async_output=False, score=False,
        auxhist=AuxhistStreamConfig(interval_minutes=aux_minutes) if aux_minutes else None)
    result = daily._run_forecast_sequence(cfg, output_dir=tmp_path / "actual-output",
        forecast_fn=daily._segmented_forecast_fn, case_builder=lambda c: (case, tmp_path))
    assert initializations == [1]
    assert calls == list(np.arange(900. if aux_minutes else 3600., 7201., 900. if aux_minutes else 3600.))
    assert len(bulk_calls) == (0 if with_history else len(calls))
    assert len(result.output_files) == 2
    assert len(result.auxhist_files) == (8 if aux_minutes else 0)
    for path in (*result.output_files, *result.auxhist_files):
        with Dataset(path) as ds:
            np.testing.assert_array_equal(ds.variables["Q2"][0], np.asarray(held_q2 if with_history else bulk_q2))
            assert np.isfinite(ds.variables["T2"][:]).all()


def test_initialized_obs_fingerprint_survives_full_carry_donation(monkeypatch):
    """Compose obs seeding with main's retained-seed ownership guard."""
    monkeypatch.setattr(om, "_CARRY_DONATE", True)
    grid = GridSpec.canary_3km_template()
    inputs = _inputs()
    nl = om.OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=_synthetic_tendencies(grid),
        observation_nudging=inputs, use_mp_re=0, mp_physics=0, cu_physics=0,
        ra_sw_physics=0, ra_lw_physics=0, rad_rk_tendf=1, run_physics=True,
        sf_surface_physics=4, bl_pbl_physics=5, sf_sfclay_physics=5)
    carry = om._initial_carry_for_run(_synthetic_state(grid), nl)
    assert carry.obs.fingerprint.unsafe_buffer_pointer() != inputs.fingerprint.unsafe_buffer_pointer()
    before = [np.asarray(x).copy() for x in jax.tree.leaves(inputs)]
    carry = om.unalias_donated_carry(carry, by_buffer=True)
    out = jax.block_until_ready(jax.jit(lambda c: c, donate_argnums=(0,))(carry))
    for x, expected in zip(jax.tree.leaves(inputs), before, strict=True):
        assert not x.is_deleted()
        np.testing.assert_array_equal(np.asarray(x), expected)
    np.testing.assert_array_equal(out.obs.fingerprint, inputs.fingerprint)


def test_daily_real_obs_driver_holds_and_expires_on_global_clock(monkeypatch, tmp_path):
    _output_hooks(monkeypatch)
    inputs, _ = prepared_inputs()
    inputs = replace(inputs, options=tuple(sorted(dict(inputs.options,
                     obs_ionf=137, obs_twindo=2., fdda_end=70.).items())))
    with np.load(ROOT / "obs_probe_000000.npz") as a:
        args = {k: jnp.asarray(a[k]) for k in ("u", "v", "tb", "qv", "pbase", "pp", "z",
            "uratx", "vratx", "tratx", "kpbl", "regime", "pblh", "terrain", "mut", "muu", "muv",
            "c1h", "c2h", "msfux", "msfuy", "msfvx", "msfvy", "msftx", "msfty")}
    driver = jax.jit(lambda c, k: observation_driver(c, inputs, ktau=k,
        xtime=k.astype(jnp.float32) * jnp.float32(.3), dt_s=18., dx=3000., **args))
    @dataclass(frozen=True)
    class Carry:
        state: object
        obs: object
        def replace(self, **kw):
            return replace(self, **kw)
    starts, sources_at_start, finals = [], [], []
    def seed(s, n):
        return Carry(s, initial_observation_carry(inputs, args["tb"].shape))
    def advance(c, n, start, clock, *, n_steps, cadence):
        starts.append((int(start), int(c.obs.read_step)))
        for k in range(int(start) - 1, int(start) - 1 + n_steps):
            previous = c.obs
            obs, sources = driver(previous, jnp.int32(k))
            if k == int(start) - 1:
                sources_at_start.append([np.count_nonzero(np.asarray(x)) for x in sources])
                if k == 200:  # hourly boundary does not reset held innovations
                    _same_leaves(previous, obs)
            c = c.replace(obs=obs)
        finals.append([np.count_nonzero(np.asarray(x)) for x in sources])
        return c
    monkeypatch.setattr(om, "_committed_initial_carry_for_run", seed)
    monkeypatch.setattr(om, "_advance_chunk", advance)
    monkeypatch.setattr(daily, "_surface_diagnostics_for_output", lambda *a, **k: {})
    nl = SimpleNamespace(rk_order=3, dt_s=18., radiation_cadence_steps=200,
                         sf_surface_physics=4, sf_lake_physics=0, observation_nudging=inputs)
    case = replace(_synthetic_case(tmp_path), namelist=nl)
    cfg = daily.DailyPipelineConfig(hours=2, dt_s=18., refresh_land_state_hourly=False,
                                    async_output=False, score=False)
    daily._run_forecast_sequence(cfg, output_dir=tmp_path / "numeric",
        forecast_fn=daily._segmented_forecast_fn, case_builder=lambda c: (case, tmp_path))
    assert starts == [(1, -1), (201, 137)]
    assert all(x > 0 for x in sources_at_start[0])
    assert all(x > 0 for x in sources_at_start[1])  # XTIME60, fdda_end70
    assert any(x > 0 for x in finals[0])
    assert finals[1] == [0, 0, 0, 0]  # XTIME119.7: WRF caller skips obs
