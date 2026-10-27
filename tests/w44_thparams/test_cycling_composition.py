"""Merge resolution guard: the Thompson family and main's cycling flag share one namelist pytree.

OperationalNamelist.tree_flatten appends ... thompson_parameters, cycling; tree_unflatten must pop them
last-in first-out. A swapped pop order drops the Thompson operands (or mis-assigns a child) only when
BOTH are present, which no family-only or cycling-only test exercises.
"""
from dataclasses import replace

import jax
import numpy as np
import pytest

from gpuwrf import runtime_parameters as registry
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_mode import OperationalNamelist


@pytest.fixture
def original(monkeypatch):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True, mp_physics=8)


def bind(namelist, values):
    return registry.bind_domain_parameters(namelist, {"d01": {"thompson_parameters": values}}, "d01")


def roundtrip(namelist):
    return jax.tree.unflatten(jax.tree.structure(namelist), jax.tree.leaves(namelist))


def test_family_and_cycling_round_trip_together(original):
    values = {"nt_c": 50e6, "tno": 10.0}
    both = replace(bind(original, values), cycling=True)
    aux = both.tree_flatten()[1]
    assert aux[-2:] == (("thompson_parameters", ("nt_c", "tno")), ("cycling", True))
    back = roundtrip(both)
    assert back.cycling is True
    assert sorted(back.thompson_parameters) == sorted(values)
    for name, value in values.items():
        assert np.asarray(back.thompson_parameters[name]) == np.float32(value)
    assert back.tree_flatten()[1] == aux
    assert registry.restart_parameter_digests(back) == registry.restart_parameter_digests(both)


def test_cycling_alone_and_family_alone_keep_their_tails(original):
    cycling = replace(original, cycling=True)
    assert cycling.tree_flatten()[1] == original.tree_flatten()[1] + (("cycling", True),)
    assert roundtrip(cycling).cycling is True and roundtrip(cycling).thompson_parameters is None
    family = bind(original, {"ato": .304})
    back = roundtrip(family)
    assert back.cycling is False and sorted(back.thompson_parameters) == ["ato"]
