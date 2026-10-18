"""Record windows (w41-bdyrec): step programs independent of the input record counts.

Windowed consumers must read exactly the records a full stack reads (bitwise,
same jit pipeline), poison a bracket outside the window instead of reading
another record, and the host owner must keep every needed record resident.
"""
import dataclasses
import math
from datetime import datetime
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import boundary_apply as ba
from gpuwrf.coupling.boundary_records import lateral_window_scope
from gpuwrf.integration.record_windows import (
    WINDOW_RECORDS,
    RecordWindows,
    host_cast_to_state,
    prepare_record_windows,
)
from gpuwrf.io.lower_boundary import LowerBoundary, lower_boundary_fields, lower_boundary_history
from gpuwrf.kernels import dyn_boundary_fp32 as kb

R, I = 9, 21600.0
_LEAF = (np.random.default_rng(7).standard_normal((R, 4, 5, 3, 7)) * 10 + 280).astype(np.float32)
_LEADS = (0.0, 54.0, 18.0, I, I + 18.0, 2 * I - 18.0, 3.5 * I, 7 * I, 8 * I, 8 * I + 18.0, 8 * I + 54.0)


def _window(base):
    rows = np.minimum(np.arange(base, base + WINDOW_RECORDS), R - 1)
    return jnp.asarray(_LEAF[rows]), jnp.asarray(np.array([base, R - 2], np.int32))


def _brackets(lead):
    """Every record a consumer may read at ``lead`` (native ceil-1, legacy floor, +1)."""
    g = lead / I
    low = {min(max(math.ceil(g) - 1, 0), R - 2), min(max(math.floor(g), 0), R - 1)}
    return {r for lo in low for r in (lo, min(lo + 1, R - 1))}


CONSUMERS = {
    "tendency": lambda leaf, lead: ba.boundary_tendency_leaf(leaf, lead, I),
    "root_rate": lambda leaf, lead: jnp.stack(ba._root_record_value_rate_real4(leaf, lead, I)),
    "scalar_rate": lambda leaf, lead: jnp.stack(ba._scalar_record_value_rate_real4(leaf, lead, I)),
    "interp": lambda leaf, lead: ba.interpolate_boundary_leaf(leaf, lead, I),
}


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("name", sorted(CONSUMERS))
def test_windowed_consumer_reads_the_full_stack_records(monkeypatch, name, native):
    monkeypatch.setattr(ba, "_NATIVE_BOUNDARY_FP32", native)
    jax.clear_caches()  # traces are cached per function object (E140/E161): no cross-mode reuse
    consumer = CONSUMERS[name]

    def full(leaf, lead):
        return consumer(leaf, lead)

    def windowed(leaf, lead, window):
        with lateral_window_scope(window):
            return consumer(leaf, lead)

    full_call, window_call = jax.jit(full), jax.jit(windowed)
    checked = poisoned = 0
    for lead in _LEADS:
        full = np.asarray(full_call(jnp.asarray(_LEAF), jnp.asarray(lead)))
        for base in range(R - 1):
            leaf, window = _window(base)
            out = np.asarray(window_call(leaf, jnp.asarray(lead), window))
            resident = _brackets(lead) <= set(range(base, base + WINDOW_RECORDS))
            if np.isnan(out).all():
                poisoned += 1
                assert not resident, (lead, base)
                continue
            checked += 1
            assert out.tobytes() == full.tobytes(), (name, lead, base)
    assert checked >= len(_LEADS) and poisoned


def test_native_interpolation_kernel_window_is_bitwise_and_poisons():
    full_call = jax.jit(lambda leaf, lead: kb.interpolate_leaf(leaf, lead, I))
    window_call = jax.jit(lambda leaf, lead, window: kb.interpolate_leaf(leaf, lead, I, window))
    for lead in _LEADS:
        full = np.asarray(full_call(jnp.asarray(_LEAF), jnp.asarray(lead)))
        for base in range(R - 1):
            leaf, window = _window(base)
            out = np.asarray(window_call(leaf, jnp.asarray(lead), window))
            lower = min(max(math.ceil(lead / I) - 1, 0), R - 2)
            if np.isnan(out).all():
                assert not {lower, lower + 1} <= set(range(base, base + WINDOW_RECORDS)), (lead, base)
            else:
                assert out.tobytes() == full.tobytes(), (lead, base)


def _owner(dt=54.0, records=R, interval=I, aux=None):
    leaf = np.zeros((records, 4, 5, 2, 3), np.float32)
    return RecordWindows(dt_s=dt, lateral={"u_bdy": leaf}, interval_s=interval, lower_boundary=aux)


@pytest.mark.parametrize("dt", [54.0, 18.0, 7.5, 300.0])
def test_lateral_range_covers_every_bracket_of_the_call(dt):
    owner = _owner(dt)
    total = int(R - 1) * int(I / dt)
    rng = np.random.default_rng(3)
    starts = sorted({1, 2, *rng.integers(1, total, 60).tolist(), total - 1, int(I / dt), int(I / dt) + 1})
    for start in starts:
        for n in (1, 2, 7, int(3600 / dt), int(I / dt)):
            if start + n - 1 > total:
                continue
            lo, hi = owner.lateral_range(start, n)
            for step in range(start, start + n):
                for lead in (step * dt, step * dt + dt / 3, step * dt + dt / 2, step * dt + dt):
                    assert _brackets(min(lead, (R - 1) * I + dt)) <= set(range(lo, hi + 1)), (start, n, lead)
            for chunk_start, size in owner.split(start, n):
                lo, hi = owner.lateral_range(chunk_start, size)
                assert hi - lo + 1 <= WINDOW_RECORDS


class _State(SimpleNamespace):
    def replace(self, _cast=True, **kw):
        return _State(**{**vars(self), **kw})


class _Carry(SimpleNamespace):
    def replace(self, **kw):
        return _Carry(**{**vars(self), **kw})


class _Clock(SimpleNamespace):
    def _replace(self, **kw):
        return _Clock(**{**vars(self), **kw})


def test_prepare_refreshes_only_when_the_window_moves_and_matches_records():
    records = np.arange(R, dtype=np.float32)[:, None, None, None, None] * np.ones((1, 4, 5, 2, 3), np.float32)
    owner = RecordWindows(dt_s=54.0, lateral={"u_bdy": records}, interval_s=I)
    carry = _Carry(state=_State(u_bdy=jnp.asarray(records)))
    clock = _Clock(bdy_window=None, lower_boundary=None)
    per_interval = int(I / 54.0)
    installs = 0
    for step in range(1, (R - 1) * per_interval + 1, 97):
        before = carry.state.u_bdy
        carry, prepared = owner.prepare(carry, clock, step, 1)
        installs += carry.state.u_bdy is not before
        base, last = np.asarray(prepared.bdy_window).tolist()
        assert last == R - 2
        lo, hi = owner.lateral_range(step, 1)
        assert base <= lo and hi <= base + WINDOW_RECORDS - 1
        # Slot s holds global record base+s (clamped past the stack end).
        np.testing.assert_array_equal(np.asarray(carry.state.u_bdy)[:, 0, 0, 0, 0],
                                      np.minimum(np.arange(base, base + WINDOW_RECORDS), R - 1))
        owner.note(carry)
    assert installs <= R  # a few refreshes per run, not one per step


def _aux(records=13, interval_steps=400):
    act = (np.arange(records) * interval_steps).astype(np.int32)
    sst = (280 + np.arange(records, dtype=np.float32))[:, None, None] * np.ones((1, 2, 3), np.float32)
    return LowerBoundary(jnp.asarray(act), jnp.asarray(sst), jnp.zeros_like(sst), jnp.asarray(sst / 10),
                         jnp.full_like(sst, 0.08))


def test_aux_window_selects_the_full_record_and_sentinel_is_loud():
    full = _aux()
    owner = RecordWindows(dt_s=54.0, lower_boundary=full)
    clock = _Clock(bdy_window=None, lower_boundary=None)
    select = jax.jit(lower_boundary_fields)
    # Initial history (own step 0) before any call, then after every call:
    # the writer runs on the window of the call that completed own_step.
    for own in [0] + list(range(1, 12 * 400 + 1, 37)):
        if own:
            _carry, prepared = owner.prepare(_Carry(), clock, own, 1)
            windowed = select(prepared.lower_boundary, jnp.asarray(own - 1))
            expected = select(full, jnp.asarray(own - 1))
            for key in expected:
                assert np.asarray(windowed[key]).tobytes() == np.asarray(expected[key]).tobytes(), (own, key)
        history = owner.lower_boundary_history(own)
        reference = lower_boundary_history(full, own)
        assert history.keys() == reference.keys()
        for key in reference:
            assert np.asarray(history[key]).tobytes() == np.asarray(reference[key]).tobytes(), (own, key)
    # History of a step outside the current window is refused (stale owner).
    with pytest.raises(ValueError, match="not in the window"):
        owner.lower_boundary_history(1)
    # A call that outruns its window (no refresh) selects the NaN sentinel.
    _carry, prepared = owner.prepare(_Carry(), clock, 1, 1)
    outrun = select(prepared.lower_boundary, jnp.asarray(3 * 400))
    assert np.isnan(np.asarray(outrun["SST"])).all()


def test_record_count_does_not_change_window_shapes():
    shapes = []
    for records in (5, 9, 13, 25):
        owner = RecordWindows(dt_s=54.0, lateral={"u_bdy": np.zeros((records, 4, 5, 2, 3), np.float32)},
                              interval_s=I, lower_boundary=_aux(records + 1))
        carry, prepared = owner.prepare(_Carry(state=_State(u_bdy=None)), _Clock(bdy_window=None, lower_boundary=None),
                                        1, 1)
        shapes.append((carry.state.u_bdy.shape, prepared.bdy_window.shape,
                       tuple(leaf.shape for leaf in prepared.lower_boundary)))
    assert len(set(shapes)) == 1


def test_memory_plan_digest_ignores_real_exe_only_keys(tmp_path):
    from netCDF4 import Dataset

    from gpuwrf.runtime import gpu_allocator as ga

    with Dataset(tmp_path / "wrfinput_d01", "w") as ds:
        ds.createDimension("west_east", 3)
        ds.DX, ds.DY = 3000.0, 3000.0
    base = "&domains\n time_step = 54,\n max_dom = 1,\n num_metgrid_levels = {lev},\n num_metgrid_soil_levels = {soil},\n/\n"

    def digest(text):
        (tmp_path / "namelist.input").write_text(text)
        return ga._case_description(SimpleNamespace(input_dir=tmp_path, namelist=None, max_dom=None))[0]

    era5 = digest(base.format(lev=38, soil=4))
    assert era5 == digest(base.format(lev=34, soil=2))
    assert era5 != digest(base.format(lev=38, soil=4).replace("time_step = 54", "time_step = 60"))


def test_memory_plan_digest_keeps_runtime_keys_sharing_a_line_with_dropped_keys(tmp_path):
    # A whole-line filter lost the runtime assignment after a dropped key on the same line
    # (time_step 54 and 60 collided): the digest filters parsed key=value pairs.
    from netCDF4 import Dataset

    from gpuwrf.runtime import gpu_allocator as ga

    with Dataset(tmp_path / "wrfinput_d01", "w") as ds:
        ds.createDimension("west_east", 3)
        ds.DX, ds.DY = 3000.0, 3000.0

    def digest(line):
        (tmp_path / "namelist.input").write_text(f"&domains\n max_dom = 1,\n{line}\n/\n")
        return ga._case_description(SimpleNamespace(input_dir=tmp_path, namelist=None, max_dom=None))[0]

    for dropped in ("p_top_requested = 5000", "num_metgrid_levels = 38", "run_hours = 24", "start_year = 2026"):
        assert digest(f" {dropped}, time_step = 54,") != digest(f" {dropped}, time_step = 60,"), dropped
        assert digest(f" time_step = 54, {dropped},") != digest(f" time_step = 60, {dropped},"), dropped
    # the dropped key itself still does not split the digest, on a shared line too
    assert digest(" p_top_requested = 5000, time_step = 54,") == digest(" p_top_requested = 1000, time_step = 54,")
    assert digest(" run_hours = 24, time_step = 54,") == digest(" run_hours = 72, time_step = 54,")


@dataclasses.dataclass(frozen=True)
class _Namelist:
    boundary_config: object
    lower_boundary: object = None
    record_windows: object = None


_META = {"schema": "wrfbdy-decoupled-leaf-v1", "times": R}


def _raw(seed=3):
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal((R, 4, 5, 2, 3)) * 10 + 280 for name in ("u_bdy", "w_bdy")}


def test_windows_are_chosen_before_the_carry_and_adopt_the_carry_dtype():
    raw = _raw()
    state = _State(**{name: jax.device_put(jnp.asarray(value)) for name, value in raw.items()})
    namelist = _Namelist(boundary_config=SimpleNamespace(update_cadence_s=I), lower_boundary=_aux())
    state, namelist, pending, report = prepare_record_windows(state=state, namelist=namelist, dt_s=54.0, root=True,
                                                              boundary_meta=_META)
    assert report["lateral"] == {"records": R, "leaves": 2} and report["aux4"] == {"records": 13}
    # The carry initialisation only ever sees window shapes, in the raw dtype.
    rows = np.arange(WINDOW_RECORDS)
    for name, value in raw.items():
        leaf = getattr(state, name)
        assert leaf.shape[0] == WINDOW_RECORDS and np.asarray(leaf).tobytes() == value[rows].tobytes()
    assert namelist.lower_boundary.activation_steps.shape == (WINDOW_RECORDS,)
    # The carry casts u to REAL on the device (the release precision chain), w stays f64.
    carry = _Carry(state=_State(u_bdy=state.u_bdy.astype(jnp.float32), w_bdy=state.w_bdy))
    namelist, _ = pending.finish(carry, namelist)
    owner = namelist.record_windows
    assert owner.records == R and owner._lateral_leaf is carry.state.u_bdy and owner._lateral_base == 0
    clock = _Clock(bdy_window=None, lower_boundary=None)
    full = {name: jax.device_put(jnp.asarray(value)) for name, value in raw.items()}
    full["u_bdy"] = full["u_bdy"].astype(jnp.float32)
    for step in (1, 2 * int(I / 54.0) + 5, 6 * int(I / 54.0)):
        carry, prepared = owner.prepare(carry, clock, step, 1)
        base = int(np.asarray(prepared.bdy_window)[0])
        rows = np.minimum(np.arange(base, base + WINDOW_RECORDS), R - 1)
        for name in raw:  # host-cast refresh == device cast of the full stack, bitwise
            assert np.asarray(getattr(carry.state, name)).tobytes() == np.asarray(full[name])[rows].tobytes()
        owner.note(carry)


def test_finish_refuses_a_carry_that_is_not_a_cast_of_the_records():
    raw = _raw()
    state = _State(**{name: jax.device_put(jnp.asarray(value)) for name, value in raw.items()})
    namelist = _Namelist(boundary_config=SimpleNamespace(update_cadence_s=I))
    state, namelist, pending, _ = prepare_record_windows(state=state, namelist=namelist, dt_s=54.0, root=True,
                                                         boundary_meta=_META)
    carry = _Carry(state=_State(u_bdy=state.u_bdy + 1.0, w_bdy=state.w_bdy))
    with pytest.raises(RuntimeError, match="not a cast"):
        pending.finish(carry, namelist)


@pytest.mark.parametrize("bad", [1.0e-40, np.nan])
def test_cast_sensitive_records_keep_the_full_stacks(bad):
    raw = _raw()
    raw["w_bdy"][R - 1, 0, 0, 0, 0] = bad
    state = _State(**{name: jax.device_put(jnp.asarray(value)) for name, value in raw.items()})
    namelist = _Namelist(boundary_config=SimpleNamespace(update_cadence_s=I))
    windowed, same, pending, report = prepare_record_windows(state=state, namelist=namelist, dt_s=54.0, root=True,
                                                             boundary_meta=_META)
    assert windowed is state and same is namelist and pending is None
    assert report["lateral"].startswith("full stacks: w_bdy")


def test_host_cast_equals_the_device_cast_and_keeps_hazards_on_device():
    raw = _raw()
    template = _State(u_bdy=jnp.zeros((2, 4, 5, 2, 3), jnp.float32), w_bdy=jnp.zeros((2, 4, 5, 2, 3), jnp.float64))
    leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in raw.items()}
    cast = host_cast_to_state(template, leaves)
    assert cast["w_bdy"] is leaves["w_bdy"]  # already the State dtype
    assert cast["u_bdy"].dtype == jnp.float32
    assert np.asarray(cast["u_bdy"]).tobytes() == np.asarray(leaves["u_bdy"].astype(jnp.float32)).tobytes()
    hazard = dict(leaves, u_bdy=leaves["u_bdy"].at[0, 0, 0, 0, 0].set(1.0e-40))
    assert host_cast_to_state(template, hazard)["u_bdy"] is hazard["u_bdy"]


def test_namelist_treedef_pickles_without_the_owner():
    """AOT executable metas pickle the call treedef; the owner (host records + device sharding) stays out."""
    import copy
    import pickle

    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import Tendencies, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _RecordWindowsAux

    owner = RecordWindows(dt_s=54.0, lateral={"u_bdy": _LEAF}, interval_s=I,
                          sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    with pytest.raises(TypeError):  # the live owner itself cannot be pickled
        pickle.dumps(owner)
    def dumps(obj):
        try:
            return pickle.dumps(obj)
        except TypeError as exc:
            raise AssertionError(f"not picklable: {exc}") from exc

    holder = _RecordWindowsAux(owner)
    clone = pickle.loads(dumps(holder))
    assert clone == holder and hash(clone) == hash(holder) and clone.value is None
    assert copy.copy(holder).value is owner and copy.deepcopy(holder).value is owner

    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    alias = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tendencies = Tendencies(**{k: jnp.zeros(shapes[alias.get(k, k)], DEFAULT_DTYPES.dtype_for(k))
                               for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")})
    namelist = OperationalNamelist(grid=grid, tendencies=tendencies, metrics=grid.metrics, dt_s=54.0,
                                   acoustic_substeps=6, time_utc="2024-09-01_00:00:00")
    namelist = dataclasses.replace(namelist, record_windows=owner)
    leaves, tree = jax.tree_util.tree_flatten(namelist)
    assert jax.tree_util.tree_unflatten(tree, leaves).record_windows is owner
    restored = jax.tree_util.tree_unflatten(pickle.loads(dumps(tree)), leaves)
    assert restored.record_windows is None and len(jax.tree_util.tree_leaves(restored)) == len(leaves)
