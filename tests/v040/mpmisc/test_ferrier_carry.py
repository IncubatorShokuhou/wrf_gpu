"""WRF Registry mp5/mp15 carry allocation and missing-restart refusal (E78)."""
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.precision import FERRIER_CARRY_LEAVES, STATE_FIELD_ORDER
from gpuwrf.contracts.state import State
from gpuwrf.io.restart_schema import validate_held_state


def grid():
    eta = jnp.linspace(1, 0, 9)
    return GridSpec(Projection("lambert", 28.3, -16.4, 3000., 3000., 2, 2),
                    TerrainProvenance("ferrier-structure-test", "ferrier", (2, 2), "m", "native", 0., True),
                    VerticalCoord("hybrid_eta", 8, 5000., eta), BCMetadata("ideal", (), 1, "linear", True),
                    eta, jnp.zeros((2, 2)))


@pytest.mark.parametrize("option,required", [(8, ()), (5, ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr")),
                                           (15, ("qrimef", "ferrier_sr"))])
def test_registry_carry_and_memory(option, required):
    state = State.zeros(grid(), mp_physics=option)
    # Historical precision-registry order differs from State slot order.
    # Both keep their own prefixes and append the same new REAL leaves.
    assert set(State.__slots__) == set(STATE_FIELD_ORDER)
    assert tuple(n for n in State.__slots__ if n in FERRIER_CARRY_LEAVES) == FERRIER_CARRY_LEAVES
    assert tuple(n for n in STATE_FIELD_ORDER if n in FERRIER_CARRY_LEAVES) == FERRIER_CARRY_LEAVES
    assert tuple(n for n in FERRIER_CARRY_LEAVES if getattr(state, n) is not None) == required
    for name in required:
        value = np.asarray(getattr(state, name))
        assert value.dtype == np.float32 and value.shape == ((2, 2) if name == "ferrier_sr" else (8, 2, 2))
        np.testing.assert_array_equal(value, 1 if name == "f_rimef_phy" else 0)
    assert sum(getattr(state, n).nbytes for n in required) == sum(
        (2 * 2 if n == "ferrier_sr" else 8 * 2 * 2) * 4 for n in required)


@pytest.mark.parametrize("option,required", [(5, ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr")),
                                           (15, ("qrimef", "ferrier_sr"))])
def test_missing_restart_fails_closed(option, required):
    fields = {name: np.zeros((2, 2) if name == "ferrier_sr" else (8, 2, 2), np.float32)
              for name in ("qv", *required)}
    validate_held_state(fields, {"mp_physics": option})
    for name in required:
        with pytest.raises(ValueError, match="missing Ferrier carry.*E78"):
            validate_held_state({k: v for k, v in fields.items() if k != name}, {"mp_physics": option})


def test_force64_preserves_wrf_real_carry():
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision
    state = State.zeros(grid(), mp_physics=5)
    result = _enforce_operational_precision(state, force_fp64=True)
    for name in ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy"):
        assert getattr(result, name).dtype == jnp.float32


@pytest.mark.parametrize("option,required", [(5, ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr")),
                                           (15, ("qrimef", "ferrier_sr"))])
def test_ferrier_restart_shape_and_storage_refused(option, required):
    fields = {name: np.zeros((2, 2) if name == "ferrier_sr" else (8, 2, 2), np.float32)
              for name in ("qv", *required)}
    for name in required:
        for bad in (fields[name][..., :1], fields[name].astype(np.int32), fields[name].astype(np.float64)):
            with pytest.raises(ValueError, match=f"{name} shape/dtype differs.*E78"):
                validate_held_state({**fields, name: bad}, {"mp_physics": option})
