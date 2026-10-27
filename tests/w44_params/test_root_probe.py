"""Keep a multi-field difference from aborting the paid root diagnostic arm."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "proofs/w44_params/root_gpu.py"
SPEC = importlib.util.spec_from_file_location("params_root_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_reports_every_changed_field_after_first_difference():
    before = {"theta": np.array([300., 301.], np.float32),
              "qv": np.array([.01, .02], np.float32),
              "double": np.array([1.], np.float64)}
    after = {name: np.nextafter(value, np.full_like(value, np.inf))
             for name, value in before.items()}
    rows = PROBE.differences(before, after)
    assert [row["leaf"] for row in rows] == list(before)
    assert [row["changed"] for row in rows] == [2, 2, 1]
    assert all(row["max_ulp"] == 1 for row in rows)
    assert all(row["max_abs"] > 0 and row["max_rel_nonzero_reference"] > 0 for row in rows)
    assert PROBE.differences(before, before) == []


def test_signed_zero_is_reported_as_nonbyte_without_relative_division():
    row, = PROBE.differences({"x": np.array([-0.], np.float32)},
                             {"x": np.array([0.], np.float32)})
    assert row["changed"] == 0 and row["max_ulp"] == 1
    assert row["max_abs"] == row["max_rel_nonzero_reference"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_fields_cannot_be_classified_as_roundoff(bad):
    with pytest.raises(AssertionError, match="x"):
        PROBE.differences({"x": np.array([1.], np.float32)},
                          {"x": np.array([bad], np.float32)})


def test_memory_inventory_uses_own_fused_cache_not_a_shared_status_row():
    regular_call, fused_call = object(), object()
    _aot_calls = {('d01', 'regular_key'): regular_call, ('d01', 'miss'): None}

    def _advance_windowed():
        return _aot_calls

    def advance():
        return _advance_windowed

    aot_name = 'fused_d01'
    cached_calls = {'signature': (fused_call, 'this_runtime_key')}

    def _fused_call():
        return aot_name, cached_calls

    def program():
        return _fused_call

    cache = {'d01': program, 'd02': None}

    def lookup(name):
        return cache[name]

    result = PROBE.runtime_calls(SimpleNamespace(advance=advance, fused_cascade=lookup))
    assert result == {('d01', 'regular_key'): regular_call,
                      ('fused_d01', 'this_runtime_key'): fused_call}
