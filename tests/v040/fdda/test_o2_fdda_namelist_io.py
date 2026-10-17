"""&fdda binding/refusals and the wrffdda_d0N reader (lane o2-fdda)."""

from __future__ import annotations

from datetime import datetime, timedelta

import netCDF4
import numpy as np
import pytest

from gpuwrf.io import fdda_namelist as fn
from gpuwrf.io.namelist_binding import NamelistNotHonouredError, require_namelist_honoured
from gpuwrf.io.namelist_check import (UnsupportedSchemeError, validate_namelist,
                                      validate_operational_namelist)

RELEASE_PHYSICS = {"mp_physics": [8], "bl_pbl_physics": [5], "sf_sfclay_physics": [5], "ra_lw_physics": [4],
                   "ra_sw_physics": [4], "sf_surface_physics": [4], "cu_physics": [0]}
FDDA = {"grid_fdda": [1], "gfdda_interval_m": [60], "gfdda_end_h": [6], "guv": [3e-4], "gt": [3e-4], "gq": [3e-4]}


def nl(**fdda):
    return {"physics": dict(RELEASE_PHYSICS), "dynamics": {"diff_opt": [1], "km_opt": [4]},
            "fdda": {**FDDA, **fdda}}


def test_grid_fdda_1_and_2_are_accepted_obs_sfdda_refused():
    validate_operational_namelist(nl())
    warnings = require_namelist_honoured(nl(), ("d01",))
    assert any("FGDT/GUV/GT/GQ" in w for w in warnings)  # remaining global metadata gap is disclosed
    assert not any("HFX_FDDA" in w for w in warnings)  # now a held history diagnostic
    assert not any("HFX_FDDA" in w for w in require_namelist_honoured(nl(grid_fdda=[0]), ("d01",)))
    validate_operational_namelist(nl(grid_fdda=[2]))
    require_namelist_honoured(nl(grid_fdda=[2]), ("d01",))
    for key, value, word in (("obs_nudge_opt", [1], "bservation"),
                             ("grid_sfdda", [1], "urface")):
        with pytest.raises((UnsupportedSchemeError, NamelistNotHonouredError)) as exc:
            validate_namelist(nl(**{key: value}))
            require_namelist_honoured(nl(**{key: value}), ("d01",))
        assert word in str(exc.value), (key, str(exc.value)[:300])


@pytest.mark.parametrize("edit, word", [
    ({"gfdda_begin_h": [1]}, "gfdda_begin_h"),
    ({"gfdda_end_m": [30]}, "gfdda_end_m"),
    ({"gfdda_interval_h": [1]}, "gfdda_interval_h"),
    ({"gfdda_interval_m": [0]}, "must be > 0"),
    ({"guv": [-1e-4]}, "guv must be positive"),
    ({"io_form_gfdda": 11}, "io_form_gfdda"),
    ({"bogus_fdda_key": [1]}, "bogus_fdda_key"),
    ({"grid_fdda": [3]}, "does not exist"),
])
def test_wrf_fatal_and_unported_fdda_selections_are_refused(edit, word):
    with pytest.raises(NamelistNotHonouredError) as exc:
        require_namelist_honoured(nl(**edit), ("d01",))
    assert word in str(exc.value)


def test_pbl_weights_need_mynn_pblh():
    bad = nl(if_no_pbl_nudging_t=[1])
    bad["physics"]["bl_pbl_physics"] = [1]
    bad["physics"]["sf_sfclay_physics"] = [1]
    assert any("PBLH" in e for e in fn.fdda_namelist_errors(bad, ("d01",)))
    assert not fn.fdda_namelist_errors(nl(if_no_pbl_nudging_t=[1]), ("d01",))


def test_spectral_surface_obs_keys_are_ignored_like_wrf_psufddagd():
    ok = nl(gph=[3e-4], xwavenum=[3], if_zfac_ph=[1], obs_coef_wind=[6e-4], max_obs=[150000],
            sgfdda_end_h=[6], guv_sfc=[3e-4])
    assert fn.fdda_namelist_errors(ok, ("d01",)) == []


def test_bound_options_and_stepfg():
    opts = fn.grid_fdda_options(nl(fgdt=[3], fgdtzero=[1], if_ramping=1, dtramp_min=-60.0,
                                   if_zfac_q=[1], k_zfac_q=[12]), "d01", 18.0)
    assert opts["stepfg"] == 10 and opts["fgdtzero"] == 1 and opts["end_h_sfc"] == 0
    assert opts["if_ramping"] == 1 and opts["dtramp_min"] == -60.0 and opts["k_zfac_q"] == 12
    assert opts["interval_m"] == 60 and opts["end_h"] == 6
    assert fn.stepfg(0.0, 18.0) == 1 and fn.stepfg(1.0, 54.0) == 1 and fn.stepfg(2.0, 54.0) == 2
    # per-domain values: d02 off
    two = nl(grid_fdda=[1, 0])
    assert fn.grid_fdda_options(two, "d02", 6.0) is None
    # gfdda_interval_m=0 falls back to gfdda_interval (set_timekeeping)
    assert fn.grid_fdda_options(nl(gfdda_interval_m=[0], gfdda_interval=[180]), "d01", 18.0)["interval_m"] == 180


def _write_wrffdda(path, times, nz=3, ny=4, nx=5, rec=None):
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", nx), ("south_north", ny),
                           ("bottom_top", nz), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1)):
            ds.createDimension(name, size)
        tv = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for k, t in enumerate(times):
            tv[k] = np.frombuffer(f"{t:%Y-%m-%d_%H:%M:%S}".encode(), dtype="S1")
        dims = {"U": ("bottom_top", "south_north", "west_east_stag"),
                "V": ("bottom_top", "south_north_stag", "west_east"),
                "T": ("bottom_top", "south_north", "west_east"), "Q": ("bottom_top", "south_north", "west_east")}
        for key, dd in dims.items():
            for suffix in ("OLD", "NEW"):
                v = ds.createVariable(f"{key}_NDG_{suffix}", "f4", ("Time", *dd))
                for k in range(len(times)):
                    v[k] = np.full(v.shape[1:], 10 * k + (1 if suffix == "NEW" else 0), np.float32)


def test_wrffdda_reader_records_activation_and_refusals(tmp_path):
    pytest.importorskip("jax")
    from gpuwrf.io.wrffdda import load_grid_fdda

    start = datetime(2023, 1, 15)
    times = [start + timedelta(minutes=60 * k) for k in range(7)]
    _write_wrffdda(tmp_path / "wrffdda_d01", times)
    kw = dict(run_start=start, nz=3, ny=4, nx=5)
    inputs = load_grid_fdda(tmp_path, nl(), "d01", dt_s=54.0, run_seconds=6 * 3600.0, **kw)
    # records 0..6 h (end_h=6 inclusive), read at the first step at/after t: ceil(3600 k / 54)
    assert inputs.record_steps == tuple(int(np.ceil(3600 * k / 54.0)) for k in range(6))
    assert np.asarray(inputs.u_old).shape == (6, 3, 4, 6) and float(inputs.t_new[2, 0, 0, 0]) == 21.0
    # end_h bounds the stream: end_h=2 -> records 0,1,2 h only
    assert load_grid_fdda(tmp_path, nl(gfdda_end_h=[2]), "d01", dt_s=54.0, run_seconds=6 * 3600.0,
                          **kw).record_steps == (0, 67, 134)
    assert load_grid_fdda(tmp_path, nl(grid_fdda=[0]), "d01", dt_s=54.0, **kw) is None
    with pytest.raises(ValueError, match="has records up to"):
        load_grid_fdda(tmp_path, nl(gfdda_end_h=[12]), "d01", dt_s=54.0, run_seconds=12 * 3600.0, **kw)
    with pytest.raises(ValueError, match="gfdda_interval_m"):
        load_grid_fdda(tmp_path, nl(gfdda_interval_m=[30]), "d01", dt_s=54.0, run_seconds=3600.0, **kw)
    with pytest.raises(ValueError, match="needs the real.exe analysis file"):
        load_grid_fdda(tmp_path / "missing", nl(), "d01", dt_s=54.0, **kw)
    with pytest.raises(ValueError, match="must be finite"):
        load_grid_fdda(tmp_path, nl(), "d01", dt_s=54.0, run_seconds=3600.0, run_start=start, nz=3, ny=4, nx=6)


@pytest.mark.parametrize("form", ["list", "scalar"])
def test_omitted_domain_entries_take_wrf_registry_defaults(form):
    """F1 (rv-fdda): Fortran assigns only the written entries -> nested children are NOT nudged unless set."""

    one = (lambda v: [v]) if form == "list" else (lambda v: v)
    three = nl(grid_fdda=one(1), gfdda_interval_m=one(60), gfdda_end_h=one(6), guv=one(3e-4), gt=one(3e-4),
               gq=one(3e-4))
    domains = ("d01", "d02", "d03")
    assert fn.grid_fdda_options(three, "d01", 18.0)["guv"] == pytest.approx(3e-4)
    assert fn.grid_fdda_code(three, "d02") == 0 and fn.grid_fdda_code(three, "d03") == 0
    assert fn.grid_fdda_options(three, "d02", 6.0) is None and fn.grid_fdda_options(three, "d03", 2.0) is None
    assert fn.fdda_namelist_errors(three, domains) == []
    # d02 switched on but its coefficients/interval omitted -> WRF Registry defaults (0), not d01's values;
    # interval 0 is WRF-fatal for that domain.
    two_on = nl(grid_fdda=[1, 1], gfdda_interval_m=[60], gfdda_end_h=[6], guv=[3e-4], gt=[3e-4], gq=[3e-4])
    assert any("gfdda_interval_m=0 (d02)" in e for e in fn.fdda_namelist_errors(two_on, domains))
    two_ok = nl(grid_fdda=[1, 1], gfdda_interval_m=[60, 60], gfdda_end_h=[6], guv=[3e-4], gt=[3e-4], gq=[3e-4])
    d02 = fn.grid_fdda_options(two_ok, "d02", 6.0)
    assert d02["guv"] == 0.0 and d02["gt"] == 0.0 and d02["gq"] == 0.0 and d02["end_h"] == 0
    assert fn.grid_fdda_options(two_ok, "d03", 2.0) is None
    # single-entry keys (one value for every domain) still apply to all domains
    ramp = nl(grid_fdda=[1, 1], gfdda_interval_m=[60, 60], if_ramping=1, dtramp_min=60.0)
    assert fn.grid_fdda_options(ramp, "d02", 6.0)["if_ramping"] == 1
    assert fn.grid_fdda_options(ramp, "d02", 6.0)["dtramp_min"] == 60.0
