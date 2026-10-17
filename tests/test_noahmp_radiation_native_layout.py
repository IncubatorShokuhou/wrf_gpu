"""Adversarial native-layout regression controls, not pristine-WRF fidelity."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels.phys_noahmp_radiation import radiation_native
from gpuwrf.physics.noahmp.energy_radiation import (
    TwoStreamParams,
    _radiation_twostream_impl,
)


def _inputs():
    # Unequal dimensions and distinct point/band values expose axis confusion.
    y, x = np.indices((2, 3))
    plane = lambda value: jnp.asarray(value + np.zeros((2, 3)), jnp.float32)
    bands = lambda visible, infrared: jnp.stack((plane(visible), plane(infrared)))
    land = SimpleNamespace(
        tg=plane(269.0), tv=plane(271.0), fwet=plane(0.4),
        smois=jnp.stack((plane(0.15 + 0.02 * x + 0.01 * y), plane(0.3))),
        sneqv=plane(25.0), sneqvo=plane(24.0), snowh=plane(0.12),
        tauss=plane(0.3), albold=plane(0.7),
    )
    forcing = SimpleNamespace(
        cosz=plane(0.3 + 0.03 * x + 0.07 * y),
        soldn=plane(200.0 + 10.0 * x + 20.0 * y), prcpsnow=plane(0.00001),
    )
    phen = SimpleNamespace(elai=plane(1.5), esai=plane(0.2), fveg=plane(0.7))
    params = TwoStreamParams(
        rhol=bands(0.1, 0.45), rhos=bands(0.15, 0.35),
        taul=bands(0.05, 0.3), taus=bands(0.03, 0.2), xl=plane(0.1),
        albsat=bands(0.15, 0.25), albdry=bands(0.2, 0.35),
        omegas=bands(0.8, 0.4),
    )
    return land, forcing, None, phen, params


@pytest.mark.parametrize("case", [
    "band_compact", "band_full", "band_prefix_only",
    "x_broadcast", "y_broadcast", "soil_x_broadcast",
])
def test_prefix_and_spatial_broadcast_on_rectangle(case, monkeypatch):
    assert jax.default_backend() == "cpu"
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    land, forcing, static, phen, params = _inputs()
    if case == "band_compact":
        params = params._replace(omegas=jnp.asarray([[0.8], [0.4]], jnp.float32))
    elif case == "band_full":
        y, x = jnp.indices((2, 3))
        params = params._replace(omegas=jnp.stack((0.8 - 0.02 * x, 0.4 + 0.03 * y)))
    elif case == "band_prefix_only":
        params = params._replace(omegas=jnp.asarray([0.8, 0.4], jnp.float32))
    elif case == "x_broadcast":
        forcing.soldn = jnp.asarray([10.0, 20.0, 30.0], jnp.float32)
    elif case == "y_broadcast":
        # A y vector needs its original singleton x axis under NumPy/JAX
        # broadcasting. A bare rank-one array always belongs to trailing x.
        forcing.soldn = jnp.asarray([40.0, 80.0], jnp.float32)[:, None]
    else:
        land.smois = jnp.asarray([[0.1, 0.2, 0.3], [0.3, 0.3, 0.3]], jnp.float32)
    args = (land, forcing, static, phen, params, jnp.float32(1800.0))
    actual = radiation_native(*args, interpret=True)
    reference = _radiation_twostream_impl(*args)
    actual_fields = {**actual[0]._asdict(), **actual[1]}
    reference_fields = {**reference[0]._asdict(), **reference[1]}
    assert actual_fields.keys() == reference_fields.keys()
    assert len(actual_fields) == 14
    for name, value in actual_fields.items():
        value = np.asarray(value)
        assert value.shape == (2, 3), name
        assert value.dtype == np.float32, name
        assert np.isfinite(value).all(), name
        # REAL leaf regression allowance; this is not a new WRF science gate
        # or a claim of compiled bit identity.
        np.testing.assert_allclose(
            value, np.asarray(reference_fields[name]),
            rtol=8 * np.finfo(np.float32).eps, atol=0.00001,
            err_msg=f"{case}: {name}",
        )
