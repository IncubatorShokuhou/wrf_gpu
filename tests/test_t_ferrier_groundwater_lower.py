"""Regression for the actual c62 TypedNdArray lowering failure in Noah-MP."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("option", [1, 2])
def test_groundwater_native_cuda_lower_with_static_timean(monkeypatch, option):
    from gpuwrf.backends import pallas
    from gpuwrf.physics.noahmp import groundwater_options as gw
    from gpuwrf.physics.noahmp.land_columns import layered_column_call

    monkeypatch.setattr(pallas, "cuda_native", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_INTERPRET", "0")
    surface = jax.ShapeDtypeStruct((1, 33), np.float32)
    profile = jax.ShapeDtypeStruct((4, 1, 33), np.float32)
    soil_parameter = jax.ShapeDtypeStruct((4, 1, 1), np.float32)
    if option == 1:
        args = (gw.GroundwaterState(surface, surface, surface), profile, profile,
                soil_parameter, profile, surface, soil_parameter, soil_parameter,
                soil_parameter, soil_parameter, surface)
    else:
        args = (surface, surface, profile, profile, profile, profile, soil_parameter,
                soil_parameter, *(soil_parameter,) * 6, surface, surface)

    def column(*values):
        # CLI options bake TIMEAN from MPTABLE into the program. Passing it as
        # a dynamic test operand misses the installed Triton literal-neg bug.
        if option == 1:
            return gw.groundwater_step(*values[:-1], jnp.asarray(6., jnp.float32), values[-1])
        return gw._equilibrium_soilwater(*values, jnp.asarray(6., jnp.float32))

    def call(*values):
        return layered_column_call(column, values, grid_shape=(1, 33), name="static_timean_groundwater")

    traced = jax.jit(call).trace(*args)
    ir = traced.lower(lowering_platforms=("cuda",)).as_text()
    assert "__gpu$xla.gpu.triton" in ir
    assert "static_timean_groundwater" in ir
