"""alisios_guard read spy and DENY-day refusal (lane a1-da). Fake guard_call: no ALISIOS tool or data is touched."""
import types

import netCDF4
import numpy as np
import pytest

from gpuwrf.assimilation.letkf import alisios_guard as ag


class _Refused(PermissionError):
    pass


def _fake(verdicts):
    calls = {"admit": 0}

    def scan(days, mode=None, **kw):
        return {d: verdicts.get(d, "ALLOW") for d in days}

    def admit(issue_days, valid_days, mode=None, receipt_out=None, **kw):
        calls["admit"] += 1
        if receipt_out:
            open(receipt_out, "w").write("{}")
        return {"result": "ALLOW"}

    return types.SimpleNamespace(scan=scan, admit=admit, GuardRefused=_Refused), calls


@pytest.fixture
def alisios_tree(tmp_path, monkeypatch):
    root = tmp_path / "alisios_data"
    root.mkdir()
    monkeypatch.setattr(ag, "DATA_ROOTS", (root.resolve(),))
    for name in ("a.nc", "b.nc"):
        with netCDF4.Dataset(root / name, "w") as ds:
            ds.createDimension("x", 2)
            ds.createVariable("v", "f4", ("x",))[:] = np.arange(2)
    (root / "notes.txt").write_text("payload")
    other = tmp_path / "ours.nc"
    with netCDF4.Dataset(other, "w") as ds:
        ds.createDimension("x", 1)
    return root, other


def test_listed_reads_recorded_unlisted_refused(alisios_tree, tmp_path, monkeypatch):
    root, other = alisios_tree
    fake, calls = _fake({})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    with ag.guarded_session({root / "a.nc": "20260503"}, ["20260502"], tmp_path / "rc") as session:
        with netCDF4.Dataset(root / "a.nc") as ds:
            assert ds["v"][1] == 1
        with netCDF4.Dataset(other):            # not ALISIOS data: untouched
            pass
        with pytest.raises(_Refused):
            netCDF4.Dataset(root / "b.nc")      # unlisted netCDF payload
        with pytest.raises(_Refused):
            open(root / "notes.txt").read()     # unlisted python open
    assert calls["admit"] == 1
    assert list(session.receipt["spy"]["observed"]) == [str((root / "a.nc").resolve())]
    assert len(session.receipt["spy"]["refused_paths"]) == 2
    assert netCDF4.Dataset is not None and not getattr(netCDF4.Dataset, "__name__", "") == "checked_dataset"


def test_deny_day_refuses_before_anything_opens(alisios_tree, tmp_path, monkeypatch):
    root, _ = alisios_tree
    fake, calls = _fake({"20260427": "DENY"})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    with pytest.raises(_Refused):
        with ag.guarded_session({root / "a.nc": ["20260426", "20260427"]}, ["20260423"], tmp_path / "rc"):
            pytest.fail("session body must not run")
    assert calls["admit"] == 0


def test_drop_deny_days_filters_candidates(monkeypatch):
    fake, _ = _fake({"20260427": "DENY"})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    kept, verdicts = ag.drop_deny_days([("x", "20260426"), ("y", "20260427"), ("z", "20260501")])
    assert kept == ["x", "z"] and verdicts["20260427"] == "DENY"


def test_empty_plan_skips_guard_but_still_refuses_alisios_opens(alisios_tree, tmp_path, monkeypatch):
    root, other = alisios_tree
    fake, calls = _fake({})
    monkeypatch.setattr(ag, "_guard_call", lambda: fake)
    with ag.guarded_session({}, [], tmp_path / "rc") as session:
        with netCDF4.Dataset(other):
            pass
        with pytest.raises(_Refused):
            netCDF4.Dataset(root / "a.nc")
    assert calls["admit"] == 0 and session.receipt["guard_result"] == "NO_ALISIOS_READS_PLANNED"
