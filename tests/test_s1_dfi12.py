"""DFL/DDFI clocks and coupled-boundary rebasing versus independent WRF equations."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.dfi.control import dfi_control_from_namelist, dfi_refusals
from gpuwrf.io.namelist_check import _parse_wrf_namelist, validate_operational_namelist

SWISS = Path(__file__).parent / "fixtures/o2_dfi_swiss_tdfi.namelist"


def nml(opt):
    out = _parse_wrf_namelist(SWISS.read_text())
    out["dfi_control"]["dfi_opt"] = opt
    return out


@pytest.mark.parametrize("opt,fwd,n_bck,n_fwd,fst", [
    (1, datetime(2023, 1, 15), 0, 30, datetime(2023, 1, 15, 0, 4, 30)),
    (2, datetime(2023, 1, 14, 23, 42), 60, 90, datetime(2023, 1, 15)),
    (3, datetime(2023, 1, 14, 23, 51), 60, 60, datetime(2023, 1, 15)),
])
def test_set_timekeeping_windows(opt, fwd, n_bck, n_fwd, fst):
    ctl = dfi_control_from_namelist(nml(opt))
    assert (ctl.fwd_start, ctl.n_bck, ctl.n_fwd, ctl.forecast_start) == (fwd, n_bck, n_fwd, fst)
    assert dfi_refusals(nml(opt)) == []
    validate_operational_namelist(nml(opt))


def test_dfl_does_not_use_backward_window_and_has_integer_midpoint():
    data = nml(1)
    data["dfi_control"].update(dfi_bckstop_day=16, dfi_fwdstop_second=1)
    ctl = dfi_control_from_namelist(data)
    assert ctl.forecast_start == datetime(2023, 1, 15, 0, 4, 30)
    assert dfi_refusals(data) == []


def test_dfl_restores_dfi_diffusion_defaults_not_forecast_options():
    from gpuwrf.dfi.driver12 import _dfl_diffusion
    assert _dfl_diffusion(nml(1)) == {"diff_opt": 0, "km_opt": 1}
    data = nml(1)
    data["dynamics"].update(diff_opt_dfi=1, km_opt_dfi=4)
    assert _dfl_diffusion(data) == {"diff_opt": 1, "km_opt": 4}


@pytest.mark.parametrize("diffusion", [None, (1, 4)])
@pytest.mark.parametrize("phase", ["forward", "forecast"])
@pytest.mark.parametrize("omit_forecast_keys", [True, False])
def test_dfl_stage_writer_binds_diffusion_in_dynamics(tmp_path, diffusion, phase, omit_forecast_keys):
    import re
    from gpuwrf.dfi import driver, driver12
    from gpuwrf.integration.nested_pipeline import _domain_int
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.netcdf_lock import Dataset

    text = re.sub(r"(^\s*dfi_opt\s*=)\s*3", r"\g<1> 1", SWISS.read_text(), flags=re.M)
    text = text.replace("&dynamics", "&dynamics\n ! WRF diff_opt/km_opt restored by DFL", 1)
    if omit_forecast_keys:
        text = re.sub(r"^\s*(?:diff_opt|km_opt)\s*=.*\n", "", text, flags=re.M)
    if diffusion is not None:
        text = text.replace("&dynamics", f"&dynamics\n diff_opt_dfi = {diffusion[0]},\n km_opt_dfi = {diffusion[1]},", 1)
    data = _parse_wrf_namelist(text)
    assert dfi_refusals(data) == []
    src = tmp_path / "input"
    src.mkdir()
    (src / "namelist.input").write_text(text)
    with Dataset(src / "wrfinput_d01", "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.array(list("2023-01-15_00:00:00"), "S1")
    ctl = dfi_control_from_namelist(data)
    when = ctl.fwd_start if phase == "forward" else ctl.forecast_start
    edits = {**driver12._clock_edits(when), **driver12._dfl_diffusion(data)}
    if phase == "forward":
        edits.update(sst_update=0, dfi_opt=0)
    stage = driver12._write_dfl_stage_dir(src, tmp_path / phase, fields={}, when=when, namelist_edits=edits)
    actual = _parse_wrf_namelist((stage / "namelist.input").read_text())
    expected = diffusion or (0, 1)
    run = Gen2Run(stage)
    # These are the exact group/domain bindings used by _load_domains.
    for key, value in zip(("diff_opt", "km_opt"), expected):
        assert actual["dynamics"].get(key) == [value]
        assert key not in actual["physics"]
        assert _domain_int(run, "dynamics", key, "d01", 0) == value
    if not omit_forecast_keys:
        old = driver._write_stage_dir(src, tmp_path / "old", fields={}, when=when, namelist_edits=edits)
        assert (stage / "namelist.input").read_bytes() == (old / "namelist.input").read_bytes()


@pytest.mark.parametrize("opt", [1, 2, 3])
def test_catalog_reports_the_recorded_cpu_and_gpu_scope(opt):
    from gpuwrf.io.scheme_catalog import classify_control, SupportStatus
    support = classify_control("dfi_opt", opt)
    assert support.status is SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified;" in support.reason
    if opt in (1, 2):
        assert "GPU/coupled-forecast qualification pending" in support.reason
    else:
        # Existing s1-gpuqual TDFI3 receipt; DFL/DDFI are separate configurations.
        assert "GPU-qualified vs CPU-WRF Swiss 24 h" in support.reason
        assert "proofs/v040/gpuqual/dfi3" in support.reason


@pytest.mark.parametrize("key,value", [("diff_opt_dfi", 2), ("km_opt_dfi", 2)])
def test_dfl_refuses_unqualified_dfi_diffusion(key, value):
    data = nml(1)
    data["dynamics"][key] = value
    assert any(key in r for r in dfi_refusals(data))


def test_dfl_boundary_keeps_piecewise_coupled_tendencies_and_original(tmp_path):
    from netCDF4 import Dataset
    from gpuwrf.dfi.driver12 import rebase_dfl_boundary
    original = tmp_path / "original"
    with Dataset(original, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("x", 2)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for k, text in enumerate(("2023-01-15_00:00:00", "2023-01-15_00:02:00")):
            times[k] = np.array(list(text), "S1")
        ds.createVariable("U_BXS", "f4", ("Time", "x"))[:] = [[1, 10], [100, 1000]]
        ds.createVariable("U_BTXS", "f4", ("Time", "x"))[:] = [[.25, .5], [2, 4]]
    before = original.read_bytes()
    stage = tmp_path / "wrfbdy_d01"
    stage.symlink_to(original)
    assert rebase_dfl_boundary(stage, offset_s=30, forecast_s=180, interval_s=120) == 30
    assert original.read_bytes() == before and not stage.is_symlink()
    with Dataset(stage) as ds:
        expected = [[8.5, 25], [16, 40], [23.5, 55], [100, 1000], [160, 1120], [220, 1240]]
        np.testing.assert_array_equal(ds["U_BXS"][:], expected)
        np.testing.assert_array_equal(ds["U_BTXS"][:3], [[.25, .5]] * 3)
        np.testing.assert_array_equal(ds["U_BTXS"][3:], [[2, 4]] * 3)
        assert b"".join(ds["Times"][0].tolist()) == b"2023-01-15_00:00:30"


def test_dfl_midpoint_adjusts_forecast_end_on_initial_and_resume(tmp_path):
    import json
    from dataclasses import dataclass
    from gpuwrf.dfi.driver import _dfl_forecast_config, MARKER
    @dataclass
    class Config:
        hours: float
    (tmp_path / MARKER).write_text(json.dumps({"dfi_opt": 1, "forecast_offset_s": 540}))
    assert _dfl_forecast_config(Config(1), tmp_path).hours == .85
    assert _dfl_forecast_config(Config(2), tmp_path).hours == 1.85
    with pytest.raises(ValueError, match="midpoint"):
        _dfl_forecast_config(Config(.1), tmp_path)


def test_ddfi_uses_unfiltered_backward_endpoint(monkeypatch):
    from dataclasses import dataclass, replace
    from types import SimpleNamespace
    import jax.numpy as jnp
    from gpuwrf.dfi.driver12 import _backward_endpoint
    from gpuwrf.dfi import driver
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.kernels import dyn_carry_fp32

    @dataclass
    class State:
        theta: object
        u: object
        v: object
        w: object
        mu_perturbation: object
        ph_perturbation: object
        qv: object
        qc: object
        def replace(self, **kw):
            return replace(self, **kw)
    @dataclass
    class NL:
        dt_s: float = 18
        run_physics: bool = True
        diff_opt: int = 1
        km_opt: int = 4
        damp_opt: int = 3
        moist_adv_opt: int = 1
        lower_boundary: object = None
        use_noahmp: bool = True
        gwd_opt: int = 0
        cu_physics: int = 0
        acoustic_substeps: int = 4
        grid: object = None
        radiation_cadence_steps: int = 33
        boundary_config: object = None
    # Nonconstant sequence: endpoint differs strongly from a weighted mean.
    x = jnp.array([0.], jnp.float32)
    state = State(x+300, x, x, x, x, x, x+.01, x+.002)
    nl = NL(boundary_config=SimpleNamespace(update_cadence_s=10800))
    ctl = dfi_control_from_namelist(nml(2))
    calls = []
    def advance(carry, nl, step, clock, **kw):
        calls.append(step)
        assert nl.dt_s == -18 and not nl.run_physics and nl.diff_opt == 0
        assert nl.moist_adv_opt == 0 and nl.km_opt == 1
        assert float(carry.state.qc[0]) == 0  # frozen, excluded from moisture coupling
        return SimpleNamespace(state=carry.state.replace(theta=carry.state.theta+1, qv=x-1e-6))
    monkeypatch.setattr(driver, "_shifted_boundary", lambda *a, **kw: {})
    monkeypatch.setattr(om, "_initial_carry_for_run", lambda st, nl: SimpleNamespace(state=st))
    monkeypatch.setattr(om, "_commit_to_operational_device", lambda c: c)
    monkeypatch.setattr(om, "build_clock_base", lambda nl: None)
    monkeypatch.setattr(om, "_advance_chunk", advance)
    monkeypatch.setattr(dyn_carry_fp32, "real_carry", lambda c: c)
    import jax
    monkeypatch.setattr(jax, "block_until_ready", lambda c: c)
    pipeline = SimpleNamespace(_wrf_sound_steps=lambda *a: 4, _domain_int=lambda *a: 0)
    report = {}
    endpoint, _sp = _backward_endpoint(SimpleNamespace(state=state), nl, ctl, None, pipeline, report)
    assert calls == list(range(-60, 0))
    np.testing.assert_array_equal(endpoint["t_2"], [60.])
    np.testing.assert_array_equal(endpoint["qc"], np.asarray(state.qc))
    assert endpoint["qv"][0] < 0  # no dfi_array_reset/floor between DDFI stages
    assert report["bck_filtered"] is False and report["bck_nonfinite"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_filtered_gate_refuses_nonfinite_candidate_or_oracle(tmp_path, bad):
    import importlib.util
    from netCDF4 import Dataset
    spec = importlib.util.spec_from_file_location("dfi_filtered_gate", Path(__file__).parents[1] /
                                                 "proofs/s1_dfi12/compare_filtered.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    paths = [tmp_path / n for n in ("port", "oracle", "raw")]
    for path in paths:
        with Dataset(path, "w") as ds:
            ds.createDimension("Time", None)
            ds.createDimension("x", 1)
            ds.createVariable("U", "f4", ("Time", "x"))[:] = [[1.]]
    assert gate.compare(*paths)["primary_passed"]
    for bad_path in paths[:2]:
        with Dataset(bad_path, "r+") as ds:
            ds["U"][:] = bad
        assert not gate.compare(*paths)["primary_passed"]
        with Dataset(bad_path, "r+") as ds:
            ds["U"][:] = 1.


@pytest.mark.parametrize("opt,expected", [(1, .7), (2, .25), (3, .25)])
def test_launch_w_is_filtered_only_for_dfl(monkeypatch, opt, expected):
    from gpuwrf.dfi import driver, driver12
    monkeypatch.setattr(driver, "_prognostic_fields", lambda *a, **kw: {"W": np.array([.25], np.float32)})
    fields = driver12._launch_fields(None, dfi_control_from_namelist(nml(opt)),
                                    {"w": np.array([.7], np.float32)}, [], None, preserve_filtered_w=True)
    np.testing.assert_array_equal(fields["W"], np.array([expected], np.float32))


def test_dfl_end_date_only_launch_resets_w_like_wrf(monkeypatch):
    from gpuwrf.dfi import driver, driver12
    data = nml(1)
    assert driver12._dfl_preserves_w(data)
    data["time_control"]["run_hours"] = 0
    assert not driver12._dfl_preserves_w(data)
    monkeypatch.setattr(driver, "_prognostic_fields", lambda *a, **kw: {"W": np.array([.25], np.float32)})
    fields = driver12._launch_fields(None, dfi_control_from_namelist(data),
                                    {"w": np.array([.7], np.float32)}, [], None,
                                    preserve_filtered_w=driver12._dfl_preserves_w(data))
    np.testing.assert_array_equal(fields["W"], np.array([.25], np.float32))


def test_filtered_radar_slot_is_zero_and_nonzero_radar_is_refused(tmp_path):
    from netCDF4 import Dataset
    from gpuwrf.dfi.driver12 import _radar_field
    path = tmp_path / 'input'
    with Dataset(path, 'w') as ds:
        for name, size in [('Time',None), ('bottom_top',2), ('south_north',2), ('west_east',2)]:
            ds.createDimension(name,size)
        ds.createVariable('QVAPOR','f4',tuple(ds.dimensions))[:] = np.zeros((1,2,2,2),np.float32)
    _radar_field(path)
    with Dataset(path,'r+') as ds:
        assert not np.asarray(ds['RAD_TTEN_DFI'][:]).any()
        assert ds['RAD_TTEN_DFI'].getncattr('units') == 'K s-1'
        ds['RAD_TTEN_DFI'][0,0,0,0] = 1e-5
    with pytest.raises(ValueError,match='radar DFI'):
        _radar_field(path)


def test_native_forecast_admits_dfl_fractional_remaining_hour(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline as pipeline
    cfg = pipeline.NestedPipelineConfig(tmp_path/'in', tmp_path/'out', tmp_path/'proof', hours=.85, max_dom=1)
    def next_stage():
        raise RuntimeError('fractional forecast admitted')
    monkeypatch.setattr(pipeline, '_batch_ensemble_size_from_env', next_stage)
    with pytest.raises(RuntimeError,match='fractional forecast admitted'):
        pipeline._execute_nested_pipeline(cfg, _prefetch_handles=[])


@pytest.mark.parametrize('hours', [0, -.5, np.nan, np.inf])
def test_native_forecast_refuses_invalid_duration(tmp_path, hours):
    from gpuwrf.integration import nested_pipeline as pipeline
    cfg = pipeline.NestedPipelineConfig(tmp_path/'in',tmp_path/'out',tmp_path/'proof',hours=hours,max_dom=1)
    with pytest.raises(ValueError,match='hours must be positive'):
        pipeline._execute_nested_pipeline(cfg,_prefetch_handles=[])


@pytest.mark.parametrize('opt', [1, 2])
def test_prepass_dispatch_and_resume_use_the_filtered_clock(tmp_path, monkeypatch, opt):
    import json
    from dataclasses import dataclass, replace
    from gpuwrf.dfi import driver, driver12
    @dataclass
    class Config:
        input_dir: Path
        output_dir: Path
        proof_dir: Path
        scratch_dir: Path | None = None
        max_dom: int = 1
        resume_checkpoint: Path | None = None
        hours: float = 1
    src = tmp_path/'input'; src.mkdir()
    (src/'namelist.input').write_text(SWISS.read_text().replace('dfi_opt                             = 3',f'dfi_opt = {opt}'))
    (src/'wrfinput_d01').write_bytes(b'raw')
    cfg = Config(src,tmp_path/'out',tmp_path/'proof')
    calls=[]
    def run(config, *, work_dir, nml, fst_dir):
        ctl = dfi_control_from_namelist(nml)
        calls.append(ctl.dfi_opt)
        assert ctl.dfi_opt == opt
        fst_dir.mkdir(parents=True)
        (fst_dir/'wrfinput_d01').write_bytes(b'filtered')
        marker = dict(dfi_opt=opt,forecast_offset_s=int((ctl.forecast_start-ctl.start).total_seconds()))
        (fst_dir/driver.MARKER).write_text(json.dumps(marker))
        return fst_dir, marker
    monkeypatch.setattr(driver12,'run_dfi12',run)
    result, report = driver.apply_dfi_prepass(cfg)
    expected_hours = .925 if opt==1 else 1.
    assert result.hours == expected_hours and calls == [opt]
    assert (cfg.output_dir/'wrfinput_initialized_d01').read_bytes() == b'filtered'
    resumed, rep = driver.apply_dfi_prepass(replace(cfg,resume_checkpoint=tmp_path/'checkpoint'))
    assert rep is None and resumed.hours == expected_hours and resumed.input_dir == result.input_dir
    again, rep = driver.apply_dfi_prepass(replace(cfg,input_dir=result.input_dir))
    assert rep is None and again.hours == expected_hours and calls == [opt]


def test_dfl_unaligned_midpoint_runs_until_wrf_clock_passes_original_end(tmp_path):
    import json
    from dataclasses import dataclass
    from gpuwrf.dfi.driver import _dfl_forecast_config, MARKER
    @dataclass
    class Config:
        hours: float
    (tmp_path/MARKER).write_text(json.dumps(dict(dfi_opt=1,forecast_offset_s=271,dt_s=18)))
    result = _dfl_forecast_config(Config(1),tmp_path)
    assert result.hours == .925  # ceil((3600-271)/18)=185; clock ends 1 s past original stop.
    assert 271 + result.hours*3600 == 3601
