"""Diagnostic evidence regressions only; no WRF fidelity/ON trajectory claim."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import census as cs


def _row(census, phase="pre_boundary_guard", field="qv", shape=(2, 4, 5)):
    host = jax.device_get(census)
    return next(r for r in cs.f2_negative_summary(host, shape)["records"]
                if r["phase"] == phase and r["field"] == field)


def _observe(phase="pre_boundary_guard", field="qv"):
    return jax.jit(lambda c, x, step: cs.observe_f2_state(
        c, phase, SimpleNamespace(**{field: x}), step))


@pytest.mark.parametrize("phase", cs.F2_PHASES)
def test_later_larger_negative_accumulates_minimum_and_all_locations(phase):
    run = _observe(phase)
    early = jnp.ones((2, 4, 5), jnp.float32).at[0, 0, 2].set(-1e-12)
    later = jnp.ones_like(early).at[1, 2, 3].set(-0.25).at[0, 1, 1].set(-0.125)
    first = run(cs.initial_census(), early, 5)
    second = run(first, later, 9)
    # Equal minima do not overwrite the first attaining step/index.
    tied = jnp.ones_like(early).at[0, 1, 1].set(-0.25)
    final = run(second, tied, 12)
    row = _row(final, phase)
    assert row["negative_count"] == 4
    assert (row["ring0_count"], row["interior_count"]) == (1, 3)
    assert row["minimum_negative"] == {
        "value": -0.25, "step": 9, "flat_index": 33, "index_kji": [1, 2, 3]}
    assert row["coverage"] == "OBSERVED" and row["exact_real32_extremum"]
    assert row["present_observations"] == 3
    p, f = cs.F2_PHASES.index(phase), cs.F2_FIELDS.index("qv")
    assert int(final.f2_first_step[p, f, 0]) == 5
    assert int(final.f2_first_index[p, f, 0]) == 2
    assert float(final.f2_first_value[p, f, 0]) == np.float32(-1e-12)
    np.testing.assert_array_equal(final.guards, first.guards)
    np.testing.assert_array_equal(final.work, first.work)
    # Observer accepts no repair/state return; all operands survive untouched.
    assert float(later[1, 2, 3]) == -0.25


def test_argmin_ties_choose_lowest_flat_index_and_top_bottom_are_interior():
    x = jnp.ones((2, 4, 5), jnp.float32)
    x = x.at[1, 2, 3].set(-2).at[0, 1, 2].set(-2)
    result = _observe()(cs.initial_census(), x, 7)
    row = _row(result)
    assert row["minimum_negative"]["index_kji"] == [0, 1, 2]
    assert row["ring0_count"] == 0 and row["interior_count"] == 2


def test_nonfinite_never_disappears_into_a_finite_negative_bound():
    x = jnp.ones((2, 4, 5), jnp.float32)
    x = x.at[0, 0, 0].set(jnp.nan).at[0, 1, 2].set(jnp.inf).at[1, 2, 3].set(-jnp.inf)
    result = _observe()(cs.initial_census(), x, 3)
    row = _row(result)
    assert row["nonfinite_count"] == 3
    assert row["negative_count"] == row["ring0_count"] == row["interior_count"] == 0
    assert row["minimum_negative"] is None
    p, f = cs.F2_PHASES.index("pre_boundary_guard"), cs.F2_FIELDS.index("qv")
    assert int(result.f2_first_step[p, f, 1]) == 3
    assert np.isnan(result.f2_first_value[p, f, 1])
    json.dumps(cs.f2_negative_summary(jax.device_get(result), x.shape), allow_nan=False)


def test_optional_none_is_na_unvisited_is_unknown_and_numbers_have_no_cap():
    result = _observe(field="Nr")(cs.initial_census(), jnp.zeros((2, 4, 5), jnp.float32), 1)
    absent = _row(result, field="Nh")
    assert absent["coverage"] == "N_A" and absent["absent_observations"] == 1
    assert not absent["exact_real32_extremum"] and absent["minimum_negative"] is None
    assert _row(result, phase="step_entry", field="Nr")["coverage"] == "UNOBSERVED"
    assert _row(result, field="Nr")["moisture_cap_applicability"] == "N_A"
    assert _row(result, field="Nr")["coverage"] == "OBSERVED"
    # A field absent at one observation and present at another is not fully covered.
    result = _observe(field="Nh")(result, jnp.zeros((2, 4, 5), jnp.float32), 2)
    assert _row(result, field="Nh")["coverage"] == "PARTIAL"


def test_non_real32_extremum_is_explicitly_unsupported():
    x = jnp.ones((2, 4, 5), jnp.float64).at[0, 1, 2].set(-1e-100)
    assert x.dtype == jnp.float64
    row = _row(_observe()(cs.initial_census(), x, 1))
    assert row["negative_count"] == 1 and row["non_real32_observations"] == 1
    assert not row["exact_real32_extremum"]


def test_location_uint64_counts_do_not_wrap_and_legacy_carry_is_unavailable():
    c = cs.initial_census()
    p, f = cs.F2_PHASES.index("pre_boundary_guard"), cs.F2_FIELDS.index("qv")
    c = c._replace(f2_negative_location=c.f2_negative_location.at[p, f].set(2**32),
                   f2_counts=c.f2_counts.at[p, f, 0].set(2**33))
    x = jnp.ones((2, 4, 5), jnp.float32).at[0, 0, 0].set(-1).at[1, 1, 1].set(-2)
    row = _row(_observe()(c, x, 1))
    assert row["ring0_count"] == row["interior_count"] == 2**32 + 1
    assert row["negative_count"] == row["ring0_count"] + row["interior_count"]
    legacy = cs.DeviceCensus(*c[:8])
    assert cs.f2_negative_summary(jax.device_get(legacy), x.shape) == {"status": "UNAVAILABLE"}
    result = _observe()(legacy, x, 2)
    assert result.f2_min_negative is None
    assert int(result.f2_counts[p, f, 0]) == 2**33 + 2


@pytest.mark.parametrize("flag", ["0", "false", "off"])
def test_off_has_no_leaves_operations_state_access_or_io(monkeypatch, tmp_path, flag):
    monkeypatch.setenv("GPUWRF_CENSUS", flag)
    assert cs.enabled() is False
    # Invalid phase/state/step prove the early return precedes all new work.
    assert cs.observe_f2_state(None, "invalid", object(), object()) is None
    graph = jax.make_jaxpr(lambda x: cs.observe_f2_state(None, "invalid", x, 1))(jnp.ones(3))
    assert graph.jaxpr.eqns == [] and jax.tree.leaves(None) == []
    cs.write_segment(tmp_path / "disabled", {"d01": SimpleNamespace(census=None)}, {}, {}, None, {})
    assert not (tmp_path / "disabled").exists()


def test_summary_sidecar_preserves_census_schema_and_reads_at_segment_only(tmp_path):
    import jsonschema
    from pathlib import Path

    x = jnp.ones((2, 4, 5), jnp.float32).at[0, 1, 2].set(-0.25)
    c = _observe()(cs.initial_census(), x, 4)
    carry = SimpleNamespace(census=c, state=SimpleNamespace(qv=x))
    nl = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3,
                         radiation_cadence_steps=33, cumulus_cadence_steps=6)
    cs.write_segment(tmp_path, {"d01": carry}, {"d01": SimpleNamespace(namelist=nl)},
                     {"d01": 1}, SimpleNamespace(written={"d01": ["file"]}), {"d01": 67})
    payload = json.loads((tmp_path / "census.json").read_text())
    jsonschema.validate(payload, json.loads(Path(cs.__file__).with_name("census_schema.json").read_text()))
    summary = json.loads((tmp_path / "f2_negative_summary.json").read_text())
    assert summary["domains"]["d01"] == cs.f2_negative_summary(jax.device_get(c), x.shape)
    assert payload["domains"]["d01"]["f2"] == cs.f2_record(jax.device_get(c), x.shape)
    assert not list(tmp_path.glob("*.tmp"))
