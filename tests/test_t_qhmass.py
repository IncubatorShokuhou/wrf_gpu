"""All-moist WRF loading at history, physics and acoustic caller seams."""
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.moisture import active_moist_arrays, total_moist_mass
from gpuwrf.contracts.physics_registry import moist_species_for_mp
from gpuwrf.coupling.physics_couplers import _wrf_hydrostatic_pressure_profiles_from_state
from gpuwrf.dynamics.core.advance_w import moist_cqw_calc_face
from gpuwrf.kernels.phys_hydrostatic_real import hydrostatic_profiles
from gpuwrf.runtime.operational_mode import _psfc_from_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs/t_qhmass"))
from wrf_moist_oracle import build_oracle


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    return build_oracle(ROOT, tmp_path_factory.mktemp("wrf_moist"))


def column(optional=(), dtype=jnp.float32):
    rng = np.random.default_rng(20261015)
    shape = (7, 3, 5)  # Masked native tile tail, nonuniform REAL operands.
    state = SimpleNamespace(**{name: jnp.asarray(rng.uniform(0, .001, shape), dtype)
                              for name in ("qv", "qc", "qr", "qi", "qs", "qg")},
                            qh=None, qi2=None, qi3=None, theta=jnp.ones(shape, dtype),
                            mu_total=jnp.asarray(rng.uniform(70000, 100000, shape[1:]), dtype),
                            mu_perturbation=jnp.zeros(shape[1:], dtype))
    for index, name in enumerate(optional):
        setattr(state, name, jnp.full(shape, (index+1)/1024, dtype))
    metrics = SimpleNamespace(c1h=jnp.linspace(.1, 1, shape[0], dtype=dtype),
                              c2h=jnp.linspace(50, 100, shape[0], dtype=dtype),
                              dnw=jnp.full(shape[0], -1/shape[0], dtype), p_top=jnp.asarray(5000, dtype))
    return state, metrics


def test_original_90_pa_counterexample():
    with jax.enable_x64():
        zero = jnp.zeros((3, 2, 2), jnp.float64)
        state = SimpleNamespace(**{n: zero for n in ("qv", "qc", "qr", "qi", "qs", "qg")},
                                qi2=None, qi3=None, qh=jnp.full_like(zero, .001),
                                mu_total=jnp.full((2, 2), 90000., jnp.float64),
                                mu_perturbation=jnp.zeros((2, 2), jnp.float64))
        metrics = SimpleNamespace(c1h=jnp.ones(3, jnp.float64), c2h=jnp.zeros(3, jnp.float64),
                                  dnw=jnp.full(3, -1/3, jnp.float64), p_top=jnp.float64(5000))
        np.testing.assert_allclose(_psfc_from_state(state, metrics), 95090., rtol=0, atol=1e-9)


@pytest.mark.parametrize("optional", [(), ("qh",), ("qi2",), ("qi2", "qi3"), ("qh", "qi2", "qi3")])
def test_pressure_and_cqw_against_pristine(original, optional, monkeypatch):
    state, metrics = column(optional)
    # The oracle's species set is independent of the selector under test.
    moist = tuple(getattr(state, name) for name in ("qv", "qc", "qr", "qi", "qs", "qg") + optional)
    expected, faces, cq = original(moist, state.mu_total, metrics.c1h, metrics.c2h, metrics.dnw, metrics.p_top)
    actual, actual_faces, psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=jnp.float32)
    for a, b in ((actual, expected), (actual_faces, faces), (psfc, faces[0])):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_array_equal(a, b)
    # CUDA's actual sweep uses explicit add/mul/sub.rn.f32. Its CPU Pallas
    # interpreter uses ordinary operators, which XLA may contract. Emulate
    # the explicit instruction rounding for this literal REAL4 byte gate.
    from gpuwrf.kernels import phys_hydrostatic_real as native
    def rounded(op, a, b, *, interpret):
        del interpret
        a, b = a.astype(jnp.float64), b.astype(jnp.float64)
        exact = {"add": lambda: a+b, "sub": lambda: a-b, "mul": lambda: a*b}[op]()
        return jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23).astype(jnp.float32)
    monkeypatch.setattr(native, "_rn", rounded)
    p_native, f_native = hydrostatic_profiles(state.mu_total, active_moist_arrays(state), metrics.c1h, metrics.c2h,
                                              metrics.dnw, metrics.p_top, interpret=True)
    np.testing.assert_array_equal(p_native, expected)
    np.testing.assert_array_equal(f_native, faces)
    # Existing M9 uses a column reduction rather than the downward recurrence.
    # At most nz+2 REAL additions (gamma_n) can distinguish the two sums.
    nz = state.qv.shape[0]
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(_psfc_from_state(state, metrics), faces[0], rtol=(nz+2)*eps, atol=0)
    # Existing calc-face combines species first. WRF interleaves the two levels;
    # bound both orders by the standard gamma_(2*n_species+1) REAL sum bound.
    got_cq = np.asarray(moist_cqw_calc_face(total_moist_mass(state)))
    steps = 2*len(moist)+1
    np.testing.assert_allclose(got_cq[1:nz], cq[1:nz], rtol=2*steps*eps/(1-steps*eps), atol=0)


def test_species_presence_excludes_scalars_and_preserves_no_hail_bytes():
    with jax.enable_x64():
        state, _ = column(("qi2", "qi3"), jnp.float64)
        for name in ("Ni", "Nr", "Nh", "qvolg", "qvolh", "qir", "qib", "nwfa", "nifa"):
            setattr(state, name, jnp.full_like(state.qv, 1.e9))
        core = state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
        for dtype in (jnp.float32, jnp.float64):
            old = core.astype(dtype) + state.qi2.astype(dtype) + state.qi3.astype(dtype)
            new = total_moist_mass(state, core_dtype=dtype)
            assert np.asarray(old).tobytes() == np.asarray(new).tobytes()
        assert len(active_moist_arrays(state)) == 8
        assert "qh" in moist_species_for_mp(17) and "qh" in moist_species_for_mp(18)
        assert "qh" not in moist_species_for_mp(22) and "qh" not in moist_species_for_mp(38)


def test_actual_acoustic_caller_passes_hail_loading(original, monkeypatch):
    """Observe the real glue at the pg_buoy_w call, not just its sum helper."""
    from gpuwrf.runtime import operational_mode as op
    state, metrics = column(("qh",))
    nz, ny, nx = state.qv.shape
    zero = jnp.zeros_like(state.qv)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        setattr(state, name, zero)
    state.theta = zero + 300
    state.p_perturbation = zero
    moist = tuple(getattr(state, n) for n in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"))
    _, _, cq = original(moist, state.mu_total, metrics.c1h, metrics.c2h, metrics.dnw, metrics.p_top)
    prep = SimpleNamespace(entry_state=state, theta_offset=300., theta_work=zero,
                           phb=jnp.zeros((nz+1, ny, nx), jnp.float32),
                           mu_save=jnp.zeros((ny, nx), jnp.float32), mub=state.mu_total)
    vector = lambda n: jnp.ones(n, jnp.float32)
    namelist = SimpleNamespace(boundary_config=SimpleNamespace(nested_frozen_wrf_boundary_bundle=False),
                               metrics=SimpleNamespace(c1f=vector(nz+1), c2f=vector(nz+1),
                                                       rdn=vector(nz), rdnw=vector(nz),
                                                       msfty=jnp.ones((ny, nx), jnp.float32)))
    class Reached(Exception):
        pass
    seen = []
    def at_pg_buoy(p, mu, mub, cqw, **kw):
        actual = np.asarray(cqw)
        np.testing.assert_array_equal(actual[1:nz], cq[1:nz])
        assert np.all(actual[1:nz] > 0)
        seen.append(True)
        raise Reached
    monkeypatch.setattr(op, "_moist_cqw_enabled", lambda: True)
    monkeypatch.setattr(op, "pg_buoy_w_moist", at_pg_buoy)
    with pytest.raises(Reached):
        op._acoustic_core_state_from_prep(None, prep, None, namelist, None)
    assert seen == [True]
