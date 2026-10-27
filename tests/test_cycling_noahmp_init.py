"""WRF cycling=.true. Noah-MP lifecycle: a NON-restart NOAHMP_INIT, never a warm restore.

Expectations follow pristine phys/module_sf_noahmpdrv.F NOAHMP_INIT IF(.NOT.restart)
(:2040-2048 glacier snow, :2109-2139 canopy): tvxy=tgxy=tahxy=TSK (273.15 with snow),
CANWAT=0 -> canliqxy, canicexy=fwetxy=0, eahxy=2000, sneqvoxy=0, alboldxy=0.65.
"""
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset

REPO = Path(__file__).resolve().parents[1]
SWISS = REPO / "examples/switzerland_d01"
WARM = {"TV": 281.5, "TG": 280.25, "TAH": 282.75, "EAH": 1234.5, "CANLIQ": .21, "CANICE": .07,
        "FWET": .17, "XSAI": .4, "CANWAT": .28, "QSNOWXY": 1e-5}


def _case(root, *, cycling, warm=True):
    root.mkdir(parents=True)
    shutil.copyfile(SWISS / "wrfinput_d01", root / "wrfinput_d01")
    text = (SWISS / "namelist.input").read_text()
    if cycling:
        text = text.replace("&time_control", "&time_control\n cycling = .true.,", 1)
    (root / "namelist.input").write_text(text)
    if warm:
        with Dataset(root / "wrfinput_d01", "r+") as ds:
            land = np.asarray(ds["XLAND"][0]) < 1.5
            for name, value in WARM.items():
                ds[name][0] = np.where(land, np.float32(value), np.asarray(ds[name][0]))
    return root


def _wrf_noahmp_init(path):
    with Dataset(path) as ds:
        tsk, snow = (np.asarray(ds[n][0], np.float64) for n in ("TSK", "SNOW"))
        seaice = np.asarray(ds["SEAICE"][0]) if "SEAICE" in ds.variables else np.zeros_like(tsk)
        glacier = (np.asarray(ds["IVGTYP"][0]) == int(ds.ISICE)) & (seaice <= 0)
    snow = np.where(glacier, np.maximum(snow, 10.), snow)
    canopy = np.where((snow > 0) & (tsk > 273.15), 273.15, tsk)
    return {"tv": canopy, "tg": canopy, "tah": canopy, "eah": np.full_like(tsk, 2000.),
            "canliq": np.zeros_like(tsk), "canice": np.zeros_like(tsk), "fwet": np.zeros_like(tsk),
            "sneqvo": np.zeros_like(tsk), "albold": np.full_like(tsk, .65)}


def _land(root):
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    return build_noahmp_land_state(root, "d01")


def _assert_carry_equal(land, attr, expected, mask):
    # Exact in the carry's own precision: NOAHMP_INIT assigns 273.15 / 0.65 to REAL
    # variables, so under the REAL release carry (GPUWRF_FAST_DEFAULTS=1) the WRF
    # value is float32(273.15); the legacy f64 carry keeps the f64 literal (E147/E148).
    actual = np.asarray(getattr(land, attr))
    assert actual.dtype in (np.float32, np.float64), (attr, actual.dtype)
    np.testing.assert_array_equal(actual[mask], np.asarray(expected, np.float64).astype(actual.dtype)[mask],
                                  err_msg=attr)


def test_cycling_true_runs_wrf_non_restart_noahmp_init(tmp_path):
    root = _case(tmp_path / "cycled", cycling=True)
    land, _static, meta = _land(root)
    expected = _wrf_noahmp_init(root / "wrfinput_d01")
    with Dataset(root / "wrfinput_d01") as ds:
        is_land = np.asarray(ds["XLAND"][0]) < 1.5
    assert is_land.any() and (np.asarray(expected["tv"])[is_land] != WARM["TV"]).all()
    for attr, value in expected.items():
        _assert_carry_equal(land, attr, value, is_land)
    # The cold path never reads XSAI; history XSAI comes from LAI (land_history XSAI_INIT).
    assert not np.any(np.asarray(land.sai)[is_land] == np.float32(WARM["XSAI"]))
    assert "non-restart" in meta["land_cycle"]["policy"]
    assert {"TV", "TG", "TAH", "EAH", "CANLIQ", "CANWAT", "XSAI"} <= set(meta["land_cycle"]["not_read"])


def test_default_spun_up_input_path_is_unchanged(tmp_path):
    # cycling=.false.: the existing heuristic still accepts a spun-up input.
    land, _static, meta = _land(_case(tmp_path / "plain", cycling=False))
    with Dataset(tmp_path / "plain/wrfinput_d01") as ds:
        is_land = np.asarray(ds["XLAND"][0]) < 1.5
    assert np.all(np.asarray(land.tv)[is_land] == np.float32(WARM["TV"]))
    assert "land_cycle" not in meta


def test_cycling_true_does_not_restore_marked_land_cycle(tmp_path):
    """A GPUWRF_LAND_CYCLE marker (always written by native cycling) must not
    trigger the restart-branch analogue when WRF runs with cycling=.true."""
    from gpuwrf.cycling.core import cycle
    reference = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h/wrfout_d01_2023-01-15_01:00:00")
    if not reference.exists():
        pytest.skip("original CPU-WRF Swiss savepoint not mounted; self-contained cycling test above stays mandatory")
    stamp = "2023-01-15_01:00:00"
    template = tmp_path / "template"
    shutil.copyfile(SWISS / "wrfinput_d01", template)
    with Dataset(template, "r+") as ds:
        ds["Times"][0] = np.frombuffer(stamp.encode(), dtype="S1")
        ds.START_DATE = stamp
    out = tmp_path / "cycled" / "wrfinput_d01"
    cycle(reference, template, out)
    text = (SWISS / "namelist.input").read_text().replace("&time_control", "&time_control\n cycling = .true.,", 1)
    (out.parent / "namelist.input").write_text(text)
    land, _static, meta = _land(out.parent)
    expected = _wrf_noahmp_init(out)
    with Dataset(out) as ds, Dataset(reference) as src:
        assert "GPUWRF_LAND_CYCLE" in ds.ncattrs()
        is_land = np.asarray(ds["XLAND"][0]) < 1.5
        # The warm source state really differs from the NOAHMP_INIT values.
        assert np.any(np.asarray(src["TV"][0])[is_land] != np.asarray(expected["tv"], np.float32)[is_land])
    for attr in ("tv", "tg", "tah", "eah", "canliq", "fwet"):
        _assert_carry_equal(land, attr, expected[attr], is_land)
    assert meta["land_cycle"]["land_cycle_marker_present"] is True
    assert meta["source"] == "WRF cycling non-restart NOAHMP_INIT"
