"""Composed warmstart/parameter regression; run against the composed source tree.

Real day/night MYNN profiles are packed into a small State. This is a forwarding
and lifecycle regression, not an independent WRF oracle or forecast comparison.
The production lifecycle, column conversion, tiling and initializer all execute.
"""
import inspect
import json
from pathlib import Path

import gpuwrf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.validation.tier2 import make_ideal_grid


SOURCE = Path(gpuwrf.__file__).resolve().parents[2]
PACKET = SOURCE / "tests/v025/b_phys/fixtures/mynn_predict_saw_wrf.json"
FIELDS = ("qke", "el_pbl", "qsq", "qc_bl", "qi_bl", "cldfra_bl")


def real_state(row):
    """Use all twelve real profiles; alternate cold and retained-QKE columns."""
    profiles, surface = row["state"], row["flux"]
    ncol, nz = np.asarray(profiles["theta"]).shape
    grid = make_ideal_grid(nx=1, ny=ncol, nz=nz, dx_m=row["dx"], dy_m=row["dx"])
    state = State.zeros(grid)

    def volume(name):
        return np.asarray(profiles[name], np.float32).T[:, :, None]

    # Reconstruct faces from the recorded mass winds. No MYNN math is replaced.
    u = np.repeat(volume("u"), 2, axis=2)
    v_mass = volume("v")
    v = np.empty((nz, ncol + 1, 1), np.float32)
    v[:, 0] = v_mass[:, 0]
    for j in range(ncol):
        v[:, j + 1] = np.float32(2) * v_mass[:, j] - v[:, j]
    w = np.concatenate((volume("w"), volume("w")[-1:]), axis=0)
    z = np.concatenate((np.zeros((1, ncol, 1), np.float32),
                        np.cumsum(volume("dz"), axis=0)), axis=0)
    qke = np.float32(2) * volume("tke")
    qke[:, ::2] = 0  # WRF's cold-column branch within cycling.
    # Guarantee the other columns select the retained/background branch.
    assert np.all(qke[:, 1::2].max(axis=0) >= np.float32(.0002))
    updates = {name: jnp.asarray(volume(name)) for name in ("theta", "qv", "qc", "qi", "qs", "qsq")}
    updates.update(u=u, v=v, w=w, p=volume("p"), ph=z * np.float32(C.GRAVITY_M_S2), qke=qke,
                   ustar=np.asarray(surface["ustar"], np.float32)[:, None],
                   xland=np.asarray(surface["xland"], np.float32)[:, None])
    return state.replace(_cast=False, **{k: jnp.asarray(v, jnp.float32) for k, v in updates.items()})


def assert_bytes(left, right):
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def exercise(row, tiled, monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    monkeypatch.setattr(C, "_MYNN_COLUMN_TILING", tiled)
    monkeypatch.setattr(C, "_MYNN_COLUMN_TILE_COLS", 4)
    traces = []

    def initialize(state, first, alp1):
        traces.append(1)
        initialized = C._mynn_state_with_first_call_qke(
            state, None, first, cycling=True,
            **({} if alp1 is None else {"alp1": alp1}))
        return tuple(getattr(initialized, name) for name in FIELDS)

    with jax.enable_x64(False):
        state = real_state(row)
        call = jax.jit(initialize)
        first = jnp.asarray(True)
        static = call(state, first, None)
        default = call(state, first, jnp.float32(.23))
        changed = call(state, first, jnp.float32(.253))
        assert_bytes(static, default)
        assert all(np.isfinite(np.asarray(v)).all() for v in changed)
        assert np.any(np.asarray(changed[0])[:, ::2] != np.asarray(default[0])[:, ::2]), (
            "ALP1 did not reach cycling QKE initialization")
        # The WRF background-QKE clear and non-first/restart lifecycle stay intact.
        np.testing.assert_array_equal(np.asarray(changed[0])[:, 1::2], 0)
        later = call(state, jnp.asarray(False), jnp.float32(.253))
        assert_bytes(later, tuple(getattr(state, name) for name in FIELDS))
        assert len(traces) == 2, "ALP1 value or first-step flag caused a new trace"
        assert C._mynn_state_with_first_call_qke(
            state, None, True, restart=True, cycling=True, alp1=jnp.float32(.253)) is state
        call.clear_cache()


@pytest.mark.parametrize("row", json.loads(PACKET.read_text())["rows"], ids=lambda row: row["label"])
@pytest.mark.parametrize("tiled", [False, True], ids=["direct", "tiled"])
def test_cycling_qke_uses_traced_alp1(row, tiled, monkeypatch):
    exercise(row, tiled, monkeypatch)


@pytest.mark.parametrize("tiled", [False, True], ids=["direct", "tiled"])
def test_default_hardcode_in_cycling_branch_is_detected(tiled, monkeypatch):
    # Mutate only the NEW cycling callsite, leaving the cold-start forwarding and
    # all physics intact. The same positive test must fail at its activity gate.
    original = C._mynn_state_with_first_call_qke
    text = inspect.getsource(original)
    anchor = 'initialize_qke=initialize, return_moments=True, **({} if alp1 is None else {"alp1": alp1})'
    assert text.count(anchor) == 1, "cycling ALP1 mutation site changed"
    mutated = text.replace(anchor, 'initialize_qke=initialize, return_moments=True, alp1=jnp.float32(.23)')
    namespace = dict(original.__globals__)
    exec(compile(mutated, "<cycling-alp1-default-mutant>", "exec"), namespace)
    monkeypatch.setattr(C, original.__name__, namespace[original.__name__])
    row = json.loads(PACKET.read_text())["rows"][0]
    with pytest.raises(AssertionError, match="ALP1 did not reach cycling QKE initialization"):
        exercise(row, tiled, monkeypatch)
