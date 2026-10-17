"""TDFI driver building blocks against WRF semantics (CPU, no model step).

* ``start_em_pressure`` vs the CPU-WRF oracle: WRF wrote ``wrfinput_initialized_d01`` after the
  forecast stage's ``start_domain``, so its P/AL/P_HYD are start_em's re-derivation from the
  filtered PH/MU/THM in the same file (fixture: two 6x6 column blocks, all levels).
* boundary re-anchoring reproduces ``bxs + dtbc*btxs`` for negative leads (backward stage) and
  the forward stage's continued ``dtbc``; ``constant_bc`` zeroes the tendency.
* ``dfi_array_reset`` (REAL division, max(0) for species) and the savehydmeteors=0 land rule.
* stage-start W (forward stage and forecast) = start_em's set_w_surface fill.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "o2_dfi_start_em_crop.npz"


def test_start_em_pressure_reproduces_wrf_filtered_file_bitwise():
    from gpuwrf.dfi.driver import start_em_pressure
    from gpuwrf.integration.d02_replay import _WRF_INIT_LIBM32

    z = np.load(FIXTURE)
    for b in range(z["PH"].shape[0]):
        base = {n: z[n][b] if z[n].ndim >= 2 else z[n] for n in
                ("PB", "PHB", "MUB", "C3H", "C4H", "C3F", "C4F", "P_TOP", "P00", "T00", "TLP", "TISO",
                 "TLP_STRAT", "P_STRAT")}
        got = start_em_pressure(ph=z["PH"][b], mu=z["MU"][b], t_2=z["THM"][b], base=base, libm=_WRF_INIT_LIBM32)
        for name in ("P", "AL", "P_HYD"):
            ref = z[name][b].astype(np.float32)
            ulp = np.abs(got[name].view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64))
            assert int(ulp.max()) == 0, (b, name, int(ulp.max()), float(np.abs(got[name] - ref).max()))


def test_start_em_pressure_is_sensitive_to_the_filtered_mass():
    from gpuwrf.dfi.driver import start_em_pressure
    from gpuwrf.integration.d02_replay import _WRF_INIT_LIBM32

    z = np.load(FIXTURE)
    base = {n: z[n][0] if z[n].ndim >= 2 else z[n] for n in
            ("PB", "PHB", "MUB", "C3H", "C4H", "C3F", "C4F", "P_TOP", "P00", "T00", "TLP", "TISO",
             "TLP_STRAT", "P_STRAT")}
    got = start_em_pressure(ph=z["PH"][0], mu=z["MU"][0] + np.float32(1.0), t_2=z["THM"][0], base=base,
                            libm=_WRF_INIT_LIBM32)
    assert np.abs(got["P"] - z["P"][0]).max() > 0.1


class _Leaf(np.ndarray):
    pass


class _FakeState:
    __slots__ = ("u_bdy", "p_bdy", "qv_bdy", "pb_bdy", "phb_bdy", "mub_bdy")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _wrf_line(r0, r1, cadence, lead):
    return r0 + lead * ((r1 - r0) / cadence)


@pytest.mark.parametrize("constant_bc", [False, True])
def test_boundary_re_anchoring_reproduces_wrf_dtbc(constant_bc):
    from gpuwrf.coupling.boundary_apply import interpolate_boundary_leaf
    from gpuwrf.dfi.driver import _shifted_boundary

    rng = np.random.default_rng(3)
    records = rng.normal(size=(3, 4, 2, 3, 5)) * 10.0 + 280.0
    pconst = np.broadcast_to(rng.normal(size=(1, 4, 2, 3, 5)), (3, 4, 2, 3, 5)).copy()
    state = _FakeState(u_bdy=records, p_bdy=pconst, qv_bdy=None, pb_bdy=None, phb_bdy=None, mub_bdy=None)
    cadence, dt, n_bck = 10800.0, 18.0, 60
    shifted_b = _shifted_boundary(state, shift_s=-(n_bck + 1) * dt, cadence_s=cadence, constant_bc=constant_bc)
    assert set(shifted_b) == {"u_bdy", "p_bdy"}  # None leaves skipped
    np.testing.assert_array_equal(np.asarray(shifted_b["p_bdy"]), pconst[:2])  # IC leaf: values kept, 2 records
    tend = 0.0 if constant_bc else 1.0
    for n in (1, 7, n_bck):
        # backward step n: lead x = step_index*dt_bck with step_index = n - n_bck - 1, true dtbc = -n*dt
        x = (n - n_bck - 1) * (-dt)
        got = np.asarray(interpolate_boundary_leaf(np.asarray(shifted_b["u_bdy"]), x, cadence))
        want = _wrf_line(records[0], records[0] + tend * (records[1] - records[0]), cadence, -n * dt)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
    shifted_f = _shifted_boundary(state, shift_s=-n_bck * dt, cadence_s=cadence, constant_bc=constant_bc)
    for n in (1, 30, 60):
        got = np.asarray(interpolate_boundary_leaf(np.asarray(shifted_f["u_bdy"]), n * dt, cadence))
        want = _wrf_line(records[0], records[0] + tend * (records[1] - records[0]), cadence, (n - n_bck) * dt)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def test_dfi_array_reset_is_real_division_with_species_floor():
    from gpuwrf.dfi.driver import _reset

    acc = {"u": np.array([1.0, -3.0], np.float32), "qc": np.array([-1e-9, 2e-6], np.float32)}
    tot = np.float32(0.99999994)
    out = _reset(acc, tot, [("qc", "QCLOUD", True)])
    assert out["u"].dtype == np.float32
    assert np.array_equal(out["u"], (acc["u"] / tot).astype(np.float32))
    assert out["u"][1] < 0  # dynamics are not floored
    assert out["qc"][0] == 0.0 and out["qc"][1] == np.float32(np.float32(2e-6) / tot)


def test_land_reset_follows_dfi_array_reset_savehydmeteors0():
    from gpuwrf.dfi.driver import _land_reset

    saved = {"SNOW": np.array([0.0, 5.0, 0.0]), "SNOWH": np.array([0.0, 0.05, 0.0]),
             "SNOWC": np.array([0.0, 1.0, 0.0]), "CANWAT": np.array([0.1, 0.0, 0.0]),
             "TSK": np.array([270.0, 260.0, 280.0]), "TSLB": np.full((4, 3), 275.0),
             "SMOIS": np.full((4, 3), 0.3)}
    end = {"SNOW": np.array([0.0, 4.0, 1.0]), "SNOWH": np.array([0.0, 0.04, 0.01]),
           "SNOWC": np.array([0.0, 0.9, 0.4]), "CANWAT": np.array([0.0, 0.2, 0.0]),
           "TSK": np.array([272.0, 258.0, 271.0]), "TSLB": np.full((4, 3), 276.0),
           "SMOIS": np.full((4, 3), 0.31)}
    saved = {k: v.astype(np.float32) for k, v in saved.items()}
    end = {k: v.astype(np.float32) for k, v in end.items()}
    out = _land_reset(end, saved)
    np.testing.assert_array_equal(out["SNOW"], [0.0, 5.0, 1.0])
    np.testing.assert_array_equal(out["SNOWC"], np.float32([0.0, 1.0, 0.4]))
    np.testing.assert_array_equal(out["CANWAT"], np.float32([0.1, 0.2, 0.0]))
    # SNOWC==0 only in column 0: TSK/TSLB/SMOIS restored there, stage-end values elsewhere
    np.testing.assert_array_equal(out["TSK"], [270.0, 258.0, 271.0])
    np.testing.assert_array_equal(out["TSLB"][:, 0], 275.0)
    np.testing.assert_array_equal(out["SMOIS"][:, 2], np.float32(0.31))


def test_stage_start_w_is_the_kinematic_fill_not_the_filtered_w(monkeypatch):
    """Forward-stage AND forecast start re-set W (start_of_simulation, use_input_w=.false.).

    [M] CPU-WRF history written at the TDFI forward start has W == W_sfc*znw**2 (3.4e-8 rel.);
    carrying the backward-filtered W into the forward stage gave PH 3.4 m2/s2 RMSE vs WRF on a
    2+2-step window.
    """
    from types import SimpleNamespace

    import gpuwrf.integration.d02_replay as d02
    from gpuwrf.dfi import driver

    shape = (3, 2, 2)
    filt = {"u": np.full((2, 2, 3), 4.0, np.float32), "v": np.full((2, 3, 2), -2.0, np.float32),
            "w": np.full(shape, 0.7, np.float32), "ph": np.zeros(shape, np.float32),
            "mu": np.zeros((2, 2), np.float32), "t_2": np.zeros((2, 2, 2), np.float32),
            "qv": np.full((2, 2, 2), 1e-3, np.float32)}
    bundle = SimpleNamespace(grid="grid", metrics="metrics")
    calls = []

    def fake_set_w_surface(w, u, v, *, grid, metrics):
        calls.append((grid, metrics))
        assert not np.any(np.asarray(w)), "set_w_surface must start from W=0 (fill), not the filtered W"
        assert u is filt["u"] and v is filt["v"]
        return np.full(shape, 0.25, np.float32), True, 0.0

    monkeypatch.setattr(d02, "_wrf_set_w_surface", fake_set_w_surface)
    monkeypatch.setattr(driver, "_start_em_pressure", lambda *a, **k: {})
    fields = driver._prognostic_fields(None, "d01", filt, [("qv", "QVAPOR", True)], bundle=bundle)
    assert calls == [("grid", "metrics")]
    np.testing.assert_array_equal(fields["W"], np.float32(0.25))
    assert fields["W"].dtype == np.float32


def test_stage_namelist_edits_and_stage_directory(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from gpuwrf.dfi.driver import MARKER, _write_stage_dir

    src = tmp_path / "src"
    src.mkdir()
    (src / "namelist.input").write_text("&time_control\n start_year = 2023, 2023,\n start_hour = 0,\n/\n"
                                        "&physics\n mp_physics = 8,\n/\n&dfi_control\n dfi_opt = 3,\n/\n")
    (src / "wrfbdy_d01").write_text("bdy")
    with netCDF4.Dataset(src / "wrfinput_d01", "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("x", 3)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.array(list("2023-01-15_00:00:00"), "S1")
        ds.createVariable("U", "f4", ("Time", "x"))[0] = np.zeros(3, np.float32)
        ds.setncattr("START_DATE", "2023-01-15_00:00:00")
    dst = _write_stage_dir(src, tmp_path / "stage", fields={"U": np.arange(3), "NOTINFILE": np.ones(2)},
                           when=datetime(2023, 1, 14, 23, 51), marker={"dfi_opt": 3},
                           namelist_edits={"start_year": 2022, "start_minute": 51, "sst_update": 0, "dfi_opt": 0})
    assert (dst / "wrfbdy_d01").is_symlink() and (dst / MARKER).exists()
    text = (dst / "namelist.input").read_text()
    assert "start_year = 2022," in text and "start_minute = 51," in text and "sst_update = 0," in text
    assert "dfi_opt = 0," in text and "start_hour = 0," in text
    with netCDF4.Dataset(dst / "wrfinput_d01") as ds:
        assert ds.getncattr("START_DATE") == "2023-01-14_23:51:00"
        assert b"".join(ds.variables["Times"][0].tolist()) == b"2023-01-14_23:51:00"
        np.testing.assert_array_equal(ds.variables["U"][0], [0.0, 1.0, 2.0])
    with netCDF4.Dataset(src / "wrfinput_d01") as ds:  # the source input is never modified
        np.testing.assert_array_equal(ds.variables["U"][0], 0.0)


def _swiss_input(tmp_path, text=None):
    src = tmp_path / "in"
    src.mkdir()
    namelist = Path(__file__).resolve().parent / "fixtures" / "o2_dfi_swiss_tdfi.namelist"
    (src / "namelist.input").write_text(text if text is not None else namelist.read_text())
    (src / "wrfinput_d01").write_bytes(b"raw")
    return src


def _config(src, tmp_path, **kw):
    from types import SimpleNamespace

    base = dict(input_dir=src, output_dir=tmp_path / "out", proof_dir=tmp_path / "proof", scratch_dir=tmp_path / "scr",
                max_dom=1, resume_checkpoint=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_run(monkeypatch):

    from gpuwrf.dfi import driver

    calls = []

    def run(config, *, work_dir, nml, fst_dir):
        calls.append((config, work_dir, fst_dir))
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        Path(fst_dir).mkdir(parents=True, exist_ok=True)
        (Path(fst_dir) / "wrfinput_d01").write_bytes(b"filtered")
        (Path(fst_dir) / driver.MARKER).write_text("{}")
        return Path(fst_dir), {"dfi_opt": 3}

    monkeypatch.setattr(driver, "run_tdfi", run)
    monkeypatch.setattr(driver, "dc_replace", lambda obj, **kw: type(obj)(**{**vars(obj), **kw}))
    return calls


def test_prepass_runs_persists_and_writes_the_wrf_named_file(tmp_path, fake_run):
    from gpuwrf.dfi.driver import apply_dfi_prepass

    src = _swiss_input(tmp_path)
    cfg, report = apply_dfi_prepass(_config(src, tmp_path))
    assert len(fake_run) == 1 and report == {"dfi_opt": 3, "prepass_wall_s": report["prepass_wall_s"],
                                              "wrfinput_initialized": str(tmp_path / "out/wrfinput_initialized_d01")}
    assert cfg.input_dir == tmp_path / "out" / "dfi_input"
    assert (tmp_path / "out/wrfinput_initialized_d01").read_bytes() == b"filtered"
    assert (tmp_path / "proof/dfi_report.json").exists()
    assert not (tmp_path / "scr/dfi_tdfi").exists()  # stage scratch removed
    # the filtered directory is never filtered twice
    again, rep2 = apply_dfi_prepass(_config(cfg.input_dir, tmp_path))
    assert rep2 is None and again.input_dir == cfg.input_dir and len(fake_run) == 1


def test_prepass_resume_reuses_the_persisted_input_without_filtering(tmp_path, fake_run):
    from gpuwrf.dfi.driver import apply_dfi_prepass

    src = _swiss_input(tmp_path)
    fresh = _config(src, tmp_path, resume_checkpoint=tmp_path / "ckpt")
    cfg, report = apply_dfi_prepass(fresh)
    assert report is None and cfg.input_dir == src and not fake_run  # nothing persisted yet
    apply_dfi_prepass(_config(src, tmp_path))
    cfg, report = apply_dfi_prepass(fresh)
    assert report is None and cfg.input_dir == tmp_path / "out" / "dfi_input" and len(fake_run) == 1


def test_prepass_inactive_and_nested_requests(tmp_path, fake_run):
    from gpuwrf.dfi.driver import apply_dfi_prepass

    text = (Path(__file__).resolve().parent / "fixtures" / "o2_dfi_swiss_tdfi.namelist").read_text()
    off = _swiss_input(tmp_path, text.replace(" dfi_opt                             = 3,", " dfi_opt = 0,"))
    cfg, report = apply_dfi_prepass(_config(off, tmp_path))
    assert report is None and cfg.input_dir == off
    (tmp_path / "in").rename(tmp_path / "in_off")
    on = _swiss_input(tmp_path)
    with pytest.raises(ValueError, match="nested"):
        apply_dfi_prepass(_config(on, tmp_path, max_dom=2))
    assert not fake_run
