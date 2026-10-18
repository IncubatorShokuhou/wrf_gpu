"""cycle_case land='cold' (a1-io): run-start perturbed-IC member cases. Synthetic IO fixtures from
test_cycle_case (a1-case); deletion/mutation checks per E39."""
import json
import shutil

import numpy as np
import pytest

from gpuwrf.assimilation.letkf.cycle_case import write_cycle_case
from gpuwrf.cycling.core import MARKER
from gpuwrf.io.netcdf_lock import Dataset

from test_cycle_case import ANALYSIS, source  # noqa: F401  (fixture reuse)


def _cold_member(root, tmp_path, name="member"):
    member = tmp_path / name
    shutil.copyfile(root / "wrfinput_d01", member)  # atmosphere would be perturbed; land untouched
    return member


def test_cold_member_admitted(source, tmp_path):  # noqa: F811
    root, _ = source
    member = _cold_member(root, tmp_path)
    manifest = write_cycle_case(root, {"d01": member}, ANALYSIS, 24, tmp_path / "case", land="cold")
    assert manifest["land"] == "cold" and manifest["admission"]["land_cold"]["status"] == "PASS"
    assert manifest["admission"]["land_cold"]["fields"] >= 20 and "land_cycle_marker" not in manifest["admission"]
    assert str(root / "wrfinput_d01") in manifest["inputs"]
    assert json.loads((tmp_path / "case" / "cycle_case_manifest.json").read_text()) == manifest


def test_cold_refuses_changed_land(source, tmp_path):  # noqa: F811
    root, _ = source
    member = _cold_member(root, tmp_path)
    with Dataset(member, "r+") as ds:
        ds["TV"][0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="TV differs"):
        write_cycle_case(root, {"d01": member}, ANALYSIS, 24, tmp_path / "case", land="cold")
    assert not (tmp_path / "case").exists()


def test_cold_refuses_marker_and_default_still_requires_it(source, tmp_path):  # noqa: F811
    root, analysis = source
    with pytest.raises(ValueError, match="must not carry"):
        write_cycle_case(root, {"d01": analysis}, ANALYSIS, 24, tmp_path / "c1", land="cold")
    member = _cold_member(root, tmp_path)
    with pytest.raises(ValueError, match="provenance"):  # default land='cycled' unchanged
        write_cycle_case(root, {"d01": member}, ANALYSIS, 24, tmp_path / "c2")
    with pytest.raises(ValueError, match="land must be"):
        write_cycle_case(root, {"d01": member}, ANALYSIS, 24, tmp_path / "c3", land="warm")
    with Dataset(analysis) as ds:
        assert MARKER in ds.ncattrs()


def test_cold_refuses_source_not_at_t_a(source, tmp_path):  # noqa: F811
    root, _ = source
    member = _cold_member(root, tmp_path)
    with Dataset(root / "wrfinput_d01", "r+") as ds:
        ds["Times"][0] = np.frombuffer(b"2026-05-04_00:00:00", dtype="S1")
    with pytest.raises(ValueError, match="valid at t_a"):
        write_cycle_case(root, {"d01": member}, ANALYSIS, 24, tmp_path / "case", land="cold")
