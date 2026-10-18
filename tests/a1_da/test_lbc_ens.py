"""Perturbed-LBC ensemble prototype (lane w41-da-ens) on the shipped non-ALISIOS Swiss example case."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import lbc_ens

CASE = Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01"


def test_member_deviations_and_cutoff():
    t = ["2023-01-15_00:00:00", "2023-01-15_03:00:00", "2023-01-15_06:00:00"]
    ctrl = {s: {"T": np.zeros((2, 3, 4))} for s in t[:2]}
    mem = {s: {"T": np.full((2, 3, 4), k + 1.0)} for k, s in enumerate(t[:2])}
    out = lbc_ens.member_deviations(ctrl, mem, t, perturb_until=t[1])
    assert out[t[2]] == {} and np.all(out[t[1]]["T"] == 2.0)
    with pytest.raises(ValueError):
        lbc_ens.member_deviations(ctrl, mem, t)


def test_ic_consistency_ring():
    a = np.zeros((2, 14, 16))
    b = a.copy()
    b[:, 7, 8] = 3.0                      # interior differences do not matter
    assert lbc_ens.ic_consistency({"T": a}, {"T": b}, 5)["T"] == 0.0
    b[:, 0, 0] = 1.0
    assert lbc_ens.ic_consistency({"T": a}, {"T": b}, 5)["T"] == 1.0


@pytest.mark.skipif(not (CASE / "wrfbdy_d01").exists(), reason="Swiss example case not in this checkout")
def test_member_wrfbdys_decode_to_requested_deviation(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    times = lbc_ens.record_times(CASE / "wrfbdy_d01")
    assert times[0] == "2023-01-15_00:00:00" and len(times) >= 3
    with Dataset(str(CASE / "wrfinput_d01")) as ds:
        nz, ny, nx = ds["T"].shape[1:]
    rng = np.random.default_rng(0)
    pattern = rng.normal(size=(nz, ny, nx))
    records = {}
    for k, t in enumerate(times):
        records[t] = {"T": 0.3 * (k + 1) * pattern} if k < 3 else {}
    manifest = lbc_ens.write_member_wrfbdys(CASE / "wrfbdy_d01", CASE / "wrfinput_d01", {"m00": records}, tmp_path / "lbc")
    rec = manifest["members"]["m00"]
    assert not rec["zero_deviation"] and "T_BXS" in rec["changed_variables"]
    assert max(rec["physical_encoding_max_state_ulp"].values()) <= 1.0
    with Dataset(str(CASE / "wrfbdy_d01")) as a, Dataset(rec["output"]) as b:
        same = [np.array_equal(a[v][k], b[v][k]) for v in ("T_BXS", "T_BYE") for k in (4, 5)]
        assert all(same)                                  # records after the cutoff keep their bytes
        assert not np.array_equal(a["T_BXS"][0], b["T_BXS"][0])
