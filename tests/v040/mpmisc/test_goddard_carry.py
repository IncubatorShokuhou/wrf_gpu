"""Original Registry mp7 cumulative/held carry and memory contract."""
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import GODDARD4ICE_CARRY_LEAVES
from gpuwrf.contracts.state import State
from gpuwrf.io.restart_schema import validate_held_state
from test_ferrier_carry import grid


def test_goddard_only_registered_carries_allocated():
    state = State.zeros(grid(), mp_physics=7)
    assert state.qh.shape == state.theta.shape
    assert state.phys_tot.shape == state.theta.shape and state.phys_tot.dtype == jnp.float32
    assert all(getattr(state, n) is None for n in ("Nh", "qvolg", "qvolh", "nwfa", "nifa"))
    total = 0
    for name in GODDARD4ICE_CARRY_LEAVES:
        value = np.asarray(getattr(state, name))
        assert value.dtype == np.float32
        assert value.shape == ((2, 2) if name == "goddard_sr" else (8, 2, 2))
        np.testing.assert_array_equal(value, 0)
        total += value.nbytes
    assert total == 12 * 8 * 2 * 2 * 4 + 2 * 2 * 4
    default = State.zeros(grid())
    assert all(getattr(default, name) is None for name in GODDARD4ICE_CARRY_LEAVES)


def test_goddard_missing_restart_refused():
    required = ("qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES)
    fields = {name: np.zeros((2, 2) if name in ("goddard_sr", "hail_acc") else (8, 2, 2), np.float32)
              for name in ("qv", *required)}
    validate_held_state(fields, {"mp_physics": 7, "use_mp_re": 0})
    for name in required:
        with pytest.raises(ValueError, match="missing Goddard4ice.*E78"):
            validate_held_state({k: v for k, v in fields.items() if k != name}, {"mp_physics": 7, "use_mp_re": 0})


@pytest.mark.parametrize("name", ["qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES])
@pytest.mark.parametrize("damage", ["shape", "dtype"])
def test_goddard_restart_malformed_registered_field_refused(name, damage):
    required = ("qv", "qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES)
    fields = {n: np.zeros((2, 2) if n in ("goddard_sr", "hail_acc") else (8, 2, 2), np.float32)
              for n in required}
    fields[name] = fields[name][..., :1] if damage == "shape" else fields[name].astype(np.int32)
    with pytest.raises(ValueError, match=f"{name} shape/dtype differs.*E78"):
        validate_held_state(fields, {"mp_physics": 7, "use_mp_re": 0})
