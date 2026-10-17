"""Original Fortran routine outputs, not comparisons against a Python port."""
from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.stochastic import update_coefficients, spectral_field, skebs_tendencies, sppt_tendencies
from gpuwrf.physics.stochastic_setup import initialize_spectrum
from gpuwrf.physics.stochastic_rng import seed_array, gaussian_innovations

FIXTURE = Path(__file__).with_name("fixtures") / "u2_stoch_real4.npz"


@pytest.fixture(scope="module")
def oracle():
    manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(FIXTURE) as data:
        return {name: jnp.asarray(data[name]) for name in data.files}


def close(got, reference, *, fft=False):
    got, reference = np.asarray(got), np.asarray(reference)
    assert np.isfinite(reference).all() and np.isfinite(got).all()
    atol = 64 * np.finfo("float32").eps * np.max(np.abs(reference)) if fft else 1e-6
    np.testing.assert_allclose(got, reference, atol=atol, rtol=4e-6 if fft else 3e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_four_original_rng_innovation_ar_steps(oracle, compiled):
    fn = jax.jit(update_coefficients) if compiled else update_coefficients
    sine, cosine = oracle["sine_initial"], oracle["cosine_initial"]
    assert np.count_nonzero(oracle["amplitude"]) > 0
    for step in range(4):
        sine, cosine = fn(sine, cosine, oracle["amplitude"], oracle["alpha"],
                          oracle[f"sine_noise_{step}"], oracle[f"cosine_noise_{step}"])
        close(sine, oracle[f"sine_{step}"])
        close(cosine, oracle[f"cosine_{step}"])


@pytest.mark.parametrize("component", list("TUV"))
@pytest.mark.parametrize("vertical", [0, 1])
def test_original_fftpack_spectral_synthesis(oracle, component, vertical):
    kwargs = {"component": component, "dx": 9000., "dy": 9000.}
    if vertical:
        kwargs.update({name: oracle[name] for name in ("vertical_cosine", "vertical_sine", "vertical_amplitude")})
    fn = jax.jit(partial(spectral_field, **kwargs))
    for step in range(4):
        result = fn(oracle[f"sine_{step}"], oracle[f"cosine_{step}"])
        reference = oracle[f"field_{step}_{vertical}_{component}"]
        if not vertical:
            result = jnp.broadcast_to(result, reference.shape)
        close(result, reference, fft=True)


def test_original_mass_coupled_skebs_and_sppt(oracle):
    result = jax.jit(skebs_tendencies)(*[oracle[name] for name in
        ("u", "v", "theta", "u_pattern", "v_pattern", "theta_pattern", "mu", "mub", "c1h", "c2h")])
    for value, name in zip(result, ("u", "v", "theta")):
        close(value, oracle[f"skebs_{name}"])
    perturbed = jax.jit(partial(sppt_tendencies, stddev=0.5, cutoff=2.))(
        *result, oracle["qv"], oracle["sppt_pattern"])
    for value, name in zip(perturbed, ("u", "v", "theta", "qv", "pattern")):
        close(value, oracle[f"sppt_{name}"])
    # Original routine genuinely exercises both tails and preserves top slots.
    assert np.min(oracle["sppt_pattern"]) < -1 and np.max(oracle["sppt_pattern"]) > 1
    np.testing.assert_array_equal(perturbed[-1][-1], oracle["sppt_pattern"][-1])
    np.testing.assert_array_equal(result[0][-1], oracle["u"][-1])
    np.testing.assert_array_equal(result[0][:, -1], oracle["u"][:, -1])
    np.testing.assert_array_equal(result[1][:, :, -1], oracle["v"][:, :, -1])


def test_primitives_have_no_float64_arithmetic(oracle):
    program = jax.make_jaxpr(update_coefficients)(*[oracle[name] for name in
        ("sine_initial", "cosine_initial", "amplitude", "alpha", "sine_noise_0", "cosine_noise_0")])
    assert "f64[" not in str(program)
    program = jax.make_jaxpr(partial(spectral_field, component="U", dx=9000., dy=9000.,
        vertical_cosine=oracle["vertical_cosine"], vertical_sine=oracle["vertical_sine"],
        vertical_amplitude=oracle["vertical_amplitude"]))(
        oracle["sine_0"], oracle["cosine_0"])
    assert "f64[" not in str(program) and "c128[" not in str(program)


def test_operand_contracts_refuse_incompatible_memory_views(oracle):
    with pytest.raises(ValueError, match="float32"):
        update_coefficients(oracle["sine_initial"].astype(jnp.float64), *[oracle[name] for name in
            ("cosine_initial", "amplitude", "alpha", "sine_noise_0", "cosine_noise_0")])
    with pytest.raises(ValueError, match="two stochastic"):
        sppt_tendencies(*[oracle[name] for name in ("u", "v", "theta", "qv")],
                       oracle["sppt_pattern"][:1], stddev=.5, cutoff=2.)


@pytest.mark.parametrize("variable", list("WTPRSQO"))
def test_original_spectral_initialization_all_seven_variable_codes(oracle, variable):
    result = initialize_spectrum(18, 16, 5, variable=variable, dx=9000., dy=9000., dt=54,
                                 lengthscale=15000., vertical=1)
    for name, value in result.items():
        reference = np.asarray(oracle[f"setup_{variable}_{name}"])
        assert np.asarray(value).dtype == np.float32
        assert np.isfinite(value).all() and np.isfinite(reference).all()
        floor = 16*np.finfo("float32").eps*np.max(np.abs(reference))
        np.testing.assert_allclose(value, reference, rtol=4e-6, atol=floor)


def test_original_compiler_rng_and_resumed_seed_after_every_step(oracle):
    from datetime import datetime
    seed = seed_array(811, 7, datetime(2026, 10, 13, 18))
    np.testing.assert_array_equal(seed, oracle["seed_initial"])
    fn = jax.jit(partial(gaussian_innovations, shape=(16, 18)))
    for step in range(4):
        sine, cosine, seed = fn(jnp.asarray(seed))
        close(sine, oracle[f"sine_noise_{step}"])
        close(cosine, oracle[f"cosine_noise_{step}"])
        np.testing.assert_array_equal(seed, oracle[f"seed_{step}"])
