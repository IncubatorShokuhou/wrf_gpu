"""Regression tests for verdict_v2 (cr-w43-teacher stage-2 E200 controls): malformed REAL receipts must not give TEACHER-OK.

Base = the real frozen-scorer receipts of wn3_20260614 (TEACHER-OK, no twin) and wn3_20260227 (own twin C_0227), copied to tmp.
Controls: ACCANHS d03 lead 17 NaN / +Inf / -Inf (as the frozen scorer records them: finite_gpu < n), deleted bounded T2
(GPU-only and both sides), lead drop, metadata change, finite keys absent (INCOMPLETE), invalid twin -> no C.
The scorer side is tested too: compare_wrfout_grid.StatsAccumulator counts NaN, +Inf, -Inf as non-finite.
"""
import copy, importlib.util, json, shutil, sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SC = Path("<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score")
_spec = importlib.util.spec_from_file_location("verdict_v2", HERE / "verdict_v2.py")
V2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V2)

needs = pytest.mark.skipif(not (SC / "wn3_20260614/x/d03.json").exists(), reason="lane receipts not present")


def stage(tmp_path, case):
    x = tmp_path / case / "x"
    x.mkdir(parents=True)
    for d in ("d01", "d02", "d03"):
        shutil.copy(SC / case / "x" / f"{d}.json", x / f"{d}.json")
    pre = json.loads((SC / case / "PRECONDITIONS.json").read_text())
    integ = json.loads((SC / case / "integ/INTEGRITY.json").read_text())
    return x, pre, integ


def edit(x, dom, fn):
    p = x / f"{dom}.json"
    d = json.loads(p.read_text()); fn(d); p.write_text(json.dumps(d))


def nonfinite(lead, kind):
    def fn(d):
        r = d["field_summaries"]["ACCANHS"]["by_lead"][lead]
        assert r["lead_h"] == lead
        r["finite_gpu"] = r["n"] - 1  # the frozen scorer's record of one NaN / +Inf / -Inf GPU element
        r["note"] = kind
    return fn


@needs
def test_healthy_real_receipts_unchanged(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["case_teacher"] == "TEACHER-OK"
    assert all(not d["coverage_v2"]["fail"] and not d["coverage_v2"]["incomplete"] for d in r["domains"].values())


@needs
@pytest.mark.parametrize("kind", ["nan", "+inf", "-inf"])
@pytest.mark.parametrize("side", ["finite_gpu", "finite_cpu"])
def test_accanhs_nonfinite_fails(tmp_path, kind, side):
    x, pre, integ = stage(tmp_path, "wn3_20260614")

    def fn(d):
        r = d["field_summaries"]["ACCANHS"]["by_lead"][17]
        r[side] = r["n"] - 1
    edit(x, "d03", fn)
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["case_teacher"] == "FAIL" and r["domains"]["d03"]["teacher"] == "FAIL"
    assert any("ACCANHS@17 non-finite" in s for s in r["domains"]["d03"]["coverage_v2"]["fail"])
    # pinned v1 is blind to it (the finding): keep that recorded
    assert V2.V1.verdict("wn3_20260614", x, None, integ)["case_teacher"] == "TEACHER-OK"


@needs
def test_deleted_bounded_t2_gpu_side_fails(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")

    def fn(d):
        del d["field_summaries"]["T2"]
        d["inventory"]["common"].remove("T2"); d["inventory"]["cpu_only"].append("T2")
    edit(x, "d01", fn)
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d01"]["teacher"] == "FAIL" and r["case_teacher"] == "FAIL"
    assert V2.V1.verdict("wn3_20260614", x, None, integ)["case_teacher"] == "TEACHER-OK"


@needs
def test_unbounded_field_missing_on_gpu_fails(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")

    def fn(d):
        del d["field_summaries"]["ACCANHS"]
        d["inventory"]["common"].remove("ACCANHS"); d["inventory"]["cpu_only"].append("ACCANHS")
    edit(x, "d03", fn)
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d03"]["teacher"] == "FAIL"
    assert any("cpu_only" in s for s in r["domains"]["d03"]["coverage_v2"]["fail"])


@needs
def test_unbounded_field_deleted_both_sides_fails(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")

    def fn(d):
        del d["field_summaries"]["ACCANHS"]
        d["inventory"]["common"].remove("ACCANHS"); d["inventory"]["cpu_variable_count"] -= 1
    edit(x, "d01", fn)
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d01"]["teacher"] == "FAIL"
    assert any("CPU variable count 374" in s for s in r["domains"]["d01"]["coverage_v2"]["fail"])


@needs
def test_common_field_without_summary_fails(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")
    edit(x, "d02", lambda d: d["field_summaries"].pop("ACCANHS"))
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d02"]["teacher"] == "FAIL"
    assert any("ACCANHS has no summary" in s for s in r["domains"]["d02"]["coverage_v2"]["fail"])


@needs
def test_deleted_t2_both_sides_fails(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")

    def fn(d):
        del d["field_summaries"]["T2"]
        d["inventory"]["common"].remove("T2"); d["inventory"]["cpu_variable_count"] -= 1
    edit(x, "d02", fn)
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    f = r["domains"]["d02"]["coverage_v2"]["fail"]
    assert r["domains"]["d02"]["teacher"] == "FAIL" and any("mandatory field T2" in s for s in f)


@needs
def test_lead_drop_and_metadata_fail(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")
    edit(x, "d01", lambda d: d["field_summaries"]["ACCANHS"]["by_lead"].pop(40))
    edit(x, "d02", lambda d: d["field_summaries"]["SMOIS"]["metadata"]["gpu"].update(dtype="float64"))
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d01"]["teacher"] == "FAIL" and r["domains"]["d02"]["teacher"] == "FAIL"


@needs
def test_absent_finite_counts_incomplete(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")
    edit(x, "d03", lambda d: d["field_summaries"]["QCLOUD"]["by_lead"][5].pop("finite_gpu"))
    r = V2.verdict("wn3_20260614", x, None, integ, pre)
    assert r["domains"]["d03"]["teacher"] == "INCOMPLETE" and r["case_teacher"] == "INCOMPLETE"


@needs
def test_preconditions_times_fail(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260614")
    pre = dict(pre, times_ok=False)
    assert V2.verdict("wn3_20260614", x, None, integ, pre)["case_teacher"] == "FAIL"


@needs
def test_invalid_twin_supplies_no_c(tmp_path):
    x, pre, integ = stage(tmp_path, "wn3_20260227")
    c = tmp_path / "C" / "x"; c.mkdir(parents=True)
    for d in ("d01", "d02", "d03"):
        shutil.copy(SC / "C_0227/x" / f"{d}.json", c / f"{d}.json")
    ok = V2.verdict("wn3_20260227", x, c, integ, pre)
    assert ok["case_teacher"] == "TEACHER-OK" and ok["twin_v2"]["valid"]
    edit(c, "d03", nonfinite(17, "nan"))
    bad = V2.verdict("wn3_20260227", x, c, integ, pre)
    assert bad["twin_v2"].get("invalid"), bad["twin_v2"]
    assert bad["domains"]["d03"]["teacher"] == "FAIL"  # d03 breaches lose their floor


@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_frozen_scorer_counts_nonfinite(val):
    sys.path.insert(0, "<USER_HOME>/src/wrf_gpu2/scripts")
    import compare_wrfout_grid as cg
    g = np.ones((3, 4)); c = np.ones((3, 4)); g[1, 2] = val
    acc = cg.StatsAccumulator(); acc.update(g, c)
    assert acc.n == 12 and acc.finite_cpu == 12 and acc.finite_gpu == 11
