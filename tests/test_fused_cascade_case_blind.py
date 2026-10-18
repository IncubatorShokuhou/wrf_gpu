"""The fused nested cascade is ONE program for every forecast case on one grid.

ALISIOS G2 (2026-10-18): the fused d02+d03 cascade closed over the parent/child
namelists, so their traced children (Noah-MP TMN/VEGFRA, metrics, tendencies, ...)
were baked into the fused HLO as constants. Every issue date compiled its own fused
program, and the closure-blind cheap key of v0.3.x (AOTKEY-v4) served one date's
executable to other dates. The namelists now ride as jit arguments, like the namelist
of ``_advance_chunk_fori`` (see tests/test_aot_cheap_key.py::test_date_axis_*).

CPU only: lowers the real fused cascade of a small nest for three issue dates whose
case leaves differ, and requires one cheap key AND one lowered HLO.
"""
from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.nesting.boundary_construction import build_nest_force_weights
from gpuwrf.runtime import domain_tree as dt
from gpuwrf.runtime.aot_key_probe import capture_call
from gpuwrf.runtime.operational_mode import OperationalNamelist, _initial_carry_for_run

# Issue dates of the G2 incident (0227 bundle priming case, 0702, 0707) and the case
# value their namelists carry in a traced child (stands in for TMN/VEGFRA).
CASES = (("2026-02-28_00:00:00", 0.25), ("2026-07-03_00:00:00", 0.5), ("2026-07-08_00:00:00", 0.75))
PARENT_DT, CHILD_DT, RATIO = 9.0, 3.0, 3


def _namelist(grid, shapes, *, date, case_value, dt_s):
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tendencies = Tendencies(**{
        k: jnp.full(shapes[sk.get(k, k)], case_value, dtype=DEFAULT_DTYPES.dtype_for(k))
        for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
    })
    return OperationalNamelist(grid=grid, tendencies=tendencies, metrics=grid.metrics, dt_s=dt_s,
                               acoustic_substeps=6, time_utc=date)


def _fused_record(date, case_value):
    """Cheap key + lowered HLO digest of the fused d02(+d03) cascade for one case."""
    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    state = State(**{f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f)) for f, s in shapes.items()})
    parent = _namelist(grid, shapes, date=date, case_value=case_value, dt_s=PARENT_DT)
    child = _namelist(grid, shapes, date=date, case_value=2.0 * case_value, dt_s=CHILD_DT)
    weights = build_nest_force_weights(parent_grid_ratio=RATIO, i_parent_start=3, j_parent_start=3,
                                       parent_grid=grid, child_grid=grid)
    program = dt._build_fused_cascade_program(
        parent_name="d02", parent_namelist=parent, parent_cadence=4, child_names=("d03",),
        child_namelists=(child,), child_weights=(weights,), child_bdy_widths=(int(state.u_bdy.shape[2]),),
        child_ratios=(RATIO,), child_cadences=(4 * RATIO,))
    carries = (_initial_carry_for_run(state, parent), _initial_carry_for_run(state, child))
    record = capture_call(program, carries[0], (carries[1],), 1, (1,), lower=True)
    assert record is not None and record["program"] == "fused", "the fused call computed no cheap key"
    return record["cheap_key"], hashlib.sha256(record["lowered"].as_text().encode()).hexdigest()


@pytest.fixture(autouse=True)
def _nested_aot(monkeypatch):
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.delenv("GPUWRF_AOT_VERIFY", raising=False)


def test_fused_cascade_is_one_program_for_three_issue_dates():
    records = [_fused_record(date, value) for date, value in CASES]
    keys, hlos = {r[0] for r in records}, {r[1] for r in records}
    assert None not in keys
    assert len(hlos) == 1, (
        f"the fused HLO depends on the case: {[h[:12] for h in hlos]} -- a case leaf is baked as a constant")
    assert len(keys) == 1, f"the fused cheap key depends on the case: {[k[:12] for k in keys]}"


def test_detector_catches_a_closed_over_child_namelist(monkeypatch):
    """Mutant: the child advance re-reads its namelist from a closure (the v0.4.0 program).

    The key stays case-blind (the closure is not the fused function's), so only the
    lowered-HLO comparison of the test above can see the baked case leaves.
    """
    original = dt._advance_chunk
    closed = {}

    def closed_over_child(carry, namelist, *args, **kwargs):
        concrete = closed.get(float(namelist.dt_s))
        return original(carry, concrete if concrete is not None else namelist, *args, **kwargs)

    real_build = dt._build_fused_cascade_program

    def build(**kwargs):
        closed.clear()
        closed[CHILD_DT] = kwargs["child_namelists"][0]
        return real_build(**kwargs)

    monkeypatch.setattr(dt, "_advance_chunk", closed_over_child)
    monkeypatch.setattr(dt, "_build_fused_cascade_program", build)
    records = [_fused_record(date, value) for date, value in CASES]
    assert len({r[0] for r in records}) == 1, "precondition: the mutant must not change the cheap key"
    assert len({r[1] for r in records}) == len(CASES), "the closed-over child namelist must bake its case leaves"


def test_fused_key_tracks_bound_tables_like_the_advance_key(monkeypatch):
    """Bound host tables stay keyed after the namelists became arguments (cr-w41-bundlekey P1).

    The fused key carrier must still let aot_baked_tables select the namelists' physics
    tables: a changed Thompson snow_sa entry changes the HLO, so it must change the key.
    """
    from dataclasses import dataclass, replace

    import jax

    from gpuwrf.physics import thompson_column as tc
    from gpuwrf.runtime import aot_baked_tables as bt
    from gpuwrf.runtime import aot_cheap_key as ck

    @jax.tree_util.register_pytree_node_class
    @dataclass(frozen=True)
    class Carry:
        state: object

        def replace(self, **kw):
            return replace(self, **kw)

        def tree_flatten(self):
            return (self.state,), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

    monkeypatch.setattr(dt, "build_child_boundary_package", lambda child, *a, **k: child)
    monkeypatch.setattr(dt, "force_child_carry_o3rad", lambda child, *a, **k: child)

    def advance(carry, namelist, *args, **kwargs):
        return carry.replace(state=carry.state + tc.THOMPSON_TABLES.snow_sa[0])

    monkeypatch.setattr(dt, "_advance_chunk", advance)
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.0, mp_physics=8, ra_sw_physics=0,
                             ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0)
    carry = Carry(jnp.ones((1,), jnp.float32))
    records = []
    for changed in (False, True):
        if changed:
            table = tc.THOMPSON_TABLES
            monkeypatch.setattr(tc, "THOMPSON_TABLES", table._replace(snow_sa=table.snow_sa.at[0].add(1.0)))
        jax.clear_caches()
        discovered = []
        product_key = dt._fused_cascade_cheap_key

        def spy(fn, aux, *args, **kwargs):
            discovered.append(len(bt._namelists(fn, aux)))
            return product_key(fn, aux, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(dt, "_fused_cascade_cheap_key", spy)
            program = dt._build_fused_cascade_program(
                parent_name="d02", parent_namelist=nl, parent_cadence=7, child_names=("d03",),
                child_namelists=(nl,), child_weights=(None,), child_bdy_widths=(1,), child_ratios=(3,),
                child_cadences=(7,))
            record = capture_call(program, carry, (carry,), 1, (1,), components=True, lower=True)
        assert record is not None and record["cheap_key"] is not None
        assert discovered == [2], "the fused key must see the parent and child namelists"
        assert record["components"]["baked_tables"] != ck.canonical_digest([])
        records.append((record["cheap_key"], hashlib.sha256(record["lowered"].as_text().encode()).hexdigest()))
    assert records[0][1] != records[1][1], "precondition: the changed table changes the HLO"
    assert records[0][0] != records[1][0], "a changed bound table must change the fused cheap key"
