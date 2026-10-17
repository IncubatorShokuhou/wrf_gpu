"""&dfi_control binding, WRF admission rules and TDFI stage windows (pure Python)."""

from datetime import datetime
from pathlib import Path

import pytest

from gpuwrf.dfi.control import dfi_control_from_namelist, dfi_refusals
from gpuwrf.io.namelist_check import _parse_wrf_namelist

SWISS = Path(__file__).resolve().parent / "fixtures" / "o2_dfi_swiss_tdfi.namelist"


def _nml(**edits):
    nml = _parse_wrf_namelist(SWISS.read_text())
    for dotted, value in edits.items():
        group, key = dotted.split("__")
        nml.setdefault(group, {})[key] = value
    return nml


def test_swiss_tdfi_windows_follow_set_timekeeping():
    ctl = dfi_control_from_namelist(_nml())
    assert ctl.active and ctl.dfi_opt == 3 and ctl.nfilter == 7 and ctl.cutoff_s == 1200
    assert ctl.start == datetime(2023, 1, 15) and ctl.bckstop == datetime(2023, 1, 14, 23, 42)
    assert ctl.fwd_start == datetime(2023, 1, 14, 23, 51)       # bckstop + (start-bckstop)/2
    assert (ctl.bck_run_s, ctl.n_bck, ctl.fwd_run_s, ctl.n_fwd) == (1080, 60, 1080, 60)
    assert ctl.dt_s == 18.0 and ctl.time_step_dfi == -1 and not ctl.constant_bc
    assert dfi_refusals(_nml()) == []


def test_non_divisible_window_runs_ceil_steps_and_integer_half():
    ctl = dfi_control_from_namelist(_nml(dfi_control__dfi_bckstop_minute=40, dfi_control__dfi_fwdstop_minute=10))
    assert ctl.bck_run_s == 1200 and ctl.n_bck == 67               # ceil(1200/18)
    assert ctl.fwd_start == datetime(2023, 1, 14, 23, 50)
    assert ctl.fwd_run_s == 1200 and ctl.n_fwd == 67


def test_restart_switches_dfi_off_like_check_a_mundo():
    ctl = dfi_control_from_namelist(_nml(time_control__restart=True))
    assert not ctl.active and dfi_refusals(_nml(time_control__restart=True)) == []


@pytest.mark.parametrize("edits, needle", [
    ({"dfi_control__dfi_nfilter": 8}, "RHOFIL"),
    ({"dfi_control__dfi_savehydmeteors": 1}, "dfi_savehydmeteors"),
    ({"dfi_control__dfi_write_dfi_history": True}, "dfi_write_dfi_history"),
    ({"domains__time_step_dfi": 12}, "time_step_dfi"),
    ({"dynamics__moist_adv_dfi_opt": 1}, "moist_adv_dfi_opt"),
    ({"bdy_control__multi_bdy_files": True}, "multi_bdy_files"),
    ({"fdda__grid_fdda": 1}, "nudging"),
    ({"fdda__obs_nudge_opt": [1]}, "nudging"),
    ({"physics__bl_pbl_physics": 10}, "bl_pbl_physics=10"),
    ({"physics__sst_update": 1}, "sst_update=1"),
    ({"physics__sf_surface_physics": 3}, "Noah-MP"),
    ({"physics__sf_surface_physics": [4, 2]}, "Noah-MP"),
    ({"physics__sf_urban_physics": 1}, "sf_urban_physics"),
    ({"dynamics__km_opt": 2}, "dfi_tke"),
    ({"dfi_control__dfi_bckstop_day": 15}, "dfi_bckstop"),
    ({"dfi_control__dfi_fwdstop_day": 14, "dfi_control__dfi_fwdstop_hour": 23}, "dfi_fwdstop"),
])
def test_unreproduced_dfi_configurations_are_refused(edits, needle):
    reasons = dfi_refusals(_nml(**edits))
    assert any(needle in r for r in reasons), reasons


def test_nested_runs_are_refused_and_resume_turns_dfi_off():
    assert any("nested" in r for r in dfi_refusals(_nml(), max_dom=2))
    assert dfi_refusals(_nml(dfi_control__dfi_nfilter=8), resume=True) == []


def test_time_step_dfi_equal_to_time_step_is_accepted():
    assert dfi_refusals(_nml(domains__time_step_dfi=18)) == []
