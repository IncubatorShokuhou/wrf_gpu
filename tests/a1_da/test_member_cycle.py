"""member_cycle.write_member_cycle (a1-io): per-member analysis wrfinputs + next-window cycle cases.

Real CPU-WRF WN3 0502 frame at 2026-05-03 12Z (read-only; skip if absent). Member B is a copy of that frame with
different passive/land state (TSLB, QKE) so 'passive fields from ITS OWN frame' is testable; E39 mutants included.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import state_io as sio
from gpuwrf.assimilation.letkf.member_cycle import ANALYSED, REBALANCED, write_member_cycle
from gpuwrf.io.netcdf_lock import Dataset

RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260502_18z_a1/run/run")
T_A = "2026-05-03_12:00:00"
FRAME = RUN / f"wrfout_d01_{T_A}"


def _alisios_reads():
    """ALISIOS files opened here: the 0502 frame at 05-03 12Z + the case files write_cycle_case hashes/slices."""
    if not FRAME.is_file():
        return {}
    from gpuwrf.assimilation.letkf.member_cycle import member_cycle_read_plan
    return member_cycle_read_plan([FRAME], RUN / "wrfinput_d01", RUN, T_A)[0]


from _a1io_guard import module_guard  # noqa: E402

_alisios_guard = module_guard(_alisios_reads, autouse=False)
NAMES = ANALYSED + REBALANCED


@pytest.fixture
def frames(tmp_path, monkeypatch, _alisios_guard):
    if not FRAME.is_file():
        pytest.skip("CPU-WRF 0502 run absent")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    b = tmp_path / "memberB" / FRAME.name
    b.parent.mkdir()
    shutil.copyfile(FRAME, b)
    with Dataset(str(b), "r+") as ds:
        ds.set_auto_mask(False)
        land = ds["LANDMASK"][0] == 1
        tslb = ds["TSLB"][0]
        tslb[:, land] += np.float32(0.25)
        ds["TSLB"][0] = tslb
        ds["QKE"][0] = ds["QKE"][0] * np.float32(1.01)
    return [FRAME, b]


def _stack(frames):
    vals = [sio.read_frame(f, T_A, NAMES) for f in frames]
    return {n: np.stack([v[n] for v in vals]) for n in NAMES}


def check_members(manifest_path, frames, analysis):
    """Each member: Times t_a, copied vars bitwise == ITS OWN frame, analysed == its slice, admitted; manifest is
    consumable like run_members.sh (members[m]['case'] for m != 'ctrl')."""
    man = json.loads(Path(manifest_path).read_text())
    labels = [k for k in man["members"] if k != "ctrl"]
    assert len(labels) == len(frames)
    for i, label in enumerate(labels):
        rec = man["members"][label]
        case = Path(rec["case"])
        assert (case / "namelist.input").is_file() and (case / "wrfbdy_d01").is_file()
        assert sio.sha256(case / "wrfinput_d01") == rec["sha256"] == rec["case_wrfinput_sha256"]
        assert all(v == "PASS" for v in rec["admission"].values()), rec["admission"]
        assert not any(rec["spec_zone_changed"].values())
        with Dataset(rec["wrfinput"]) as ds, Dataset(str(frames[i])) as fr:
            ds.set_auto_mask(False)
            fr.set_auto_mask(False)
            assert np.asarray(ds["Times"][0]).tobytes().decode() == T_A
            prov = json.loads(ds.getncattr(sio.PROVENANCE_ATTRIBUTE))
            skip = set(NAMES) | {"THM", "P_HYD"}
            for name in prov["copied_from_frame"]:
                if name not in skip:
                    assert np.array_equal(ds[name][0], fr[name][0]), (label, name)
            for name in NAMES:
                assert np.array_equal(ds[name][0], np.asarray(analysis[name][i]).astype(ds[name].dtype)), (label, name)
            assert "GPUWRF_LAND_CYCLE" in ds.ncattrs()
    return man


def test_zero_increment_members_equal_own_frames(frames, tmp_path):
    analysis = _stack(frames)
    path = write_member_cycle(frames, analysis, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "cycle",
                              member_ids=["m00", "m01"])
    man = check_members(path, frames, analysis)
    for i, label in enumerate(("m00", "m01")):  # zero increment: THM / P_HYD bitwise too
        assert all(v == 0 for v in man["members"][label]["n_changed"].values())
        with Dataset(man["members"][label]["wrfinput"]) as ds, Dataset(str(frames[i])) as fr:
            for name in ("THM", "P_HYD", "TSLB", "QKE"):
                assert np.array_equal(ds[name][0], fr[name][0]), (label, name)
    assert man["hours"] == 6 and man["t_a"] == T_A and man["n_members"] == 2


def test_frame_mixup_mutant_caught(frames, tmp_path):
    analysis = _stack([frames[0], frames[0]])
    path = write_member_cycle([frames[0], frames[0]], analysis, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "mix")
    with pytest.raises(AssertionError):  # member 1 must carry B's TSLB/QKE
        check_members(path, frames, analysis)


def test_interior_increment_and_spec_zone_guard(frames, tmp_path):
    geo = sio.read_geometry(frames[0], base_wrfinput=RUN / "wrfinput_d01")
    base = _stack(frames)
    analysis = {n: np.array(v) for n, v in base.items()}
    bump = sio.boundary_taper(70, 120, 5, 5) * np.exp(-(((np.arange(120) - 70) / 8.0) ** 2)[None, :]
                                                    - (((np.arange(70) - 35) / 8.0) ** 2)[:, None])
    for i, f in enumerate(frames):
        bg = sio.read_frame(f, T_A, NAMES + ("MU",))
        an = {**bg, "T": (bg["T"].astype(np.float64) + 0.4 * bump * (np.arange(44)[:, None, None] < 15)).astype(np.float32)}
        out = sio.rebalance_increment(bg, an, geo)
        analysis["T"][i], analysis["P"][i], analysis["PH"][i] = an["T"], out["P"], out["PH"]
    path = write_member_cycle(frames, analysis, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "inc")
    man = check_members(path, frames, analysis)
    rec = man["members"]["m00"]
    assert rec["n_changed"]["T"] > 1000 and rec["n_changed"]["PH"] > 1000 and rec["n_changed"]["U"] == 0
    assert rec["rebalance"]["residual_change_rel_max"] <= 1e-6
    bad = {n: np.array(v) for n, v in analysis.items()}
    bad["T"][1, :, 2, 60] += np.float32(0.5)  # inside the specified/relaxation zone
    with pytest.raises(ValueError, match="specified/relaxation zone"):
        write_member_cycle(frames, bad, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "bad")
    assert not (tmp_path / "bad" / "ENSEMBLE_MANIFEST.json").exists()


def test_refusals(frames, tmp_path):
    analysis = _stack(frames)
    with pytest.raises(ValueError, match="missing"):
        write_member_cycle(frames, {k: v for k, v in analysis.items() if k != "PH"}, T_A, RUN / "wrfinput_d01", RUN, 6,
                           tmp_path / "a")
    with pytest.raises(ValueError, match="leading axis"):
        write_member_cycle(frames[:1], analysis, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "b")
    with pytest.raises(ValueError, match="ctrl"):
        write_member_cycle(frames, analysis, T_A, RUN / "wrfinput_d01", RUN, 6, tmp_path / "c", member_ids=["ctrl", "x"])
    with pytest.raises(ValueError):  # frame not valid at t_a
        write_member_cycle(frames, analysis, "2026-05-03_15:00:00", RUN / "wrfinput_d01", RUN, 6, tmp_path / "d")
