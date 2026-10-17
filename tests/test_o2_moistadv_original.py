"""o2-moistadv: moist_adv_opt=0 (WRF ORIGINAL) transports moisture with the plain WRF stencil.

<= v0.3.4 ``_rk_scan_step`` gated moisture transport on ``moist_adv_opt != 0`` (option 0 = NO dycore
transport) and both CLI loaders bound an OMITTED moist/scalar_adv_opt as 0.  Pristine WRF
(solve_em.F:2282-2420, module_em.F rk_scalar_tend :1265-1331) runs rk_scalar_tend + rk_update_scalar for
every option; ORIGINAL differs from positive-definite only at rk_step == rk_order (plain advect_scalar
instead of advect_scalar_pd, and no rk_update_scalar_pd pre-add).

(1) oracle: the final-stage moisture tendency for option 0 equals the UNMODIFIED pristine advect_scalar
    (tests/v025/b_core/scalar_oracle.py, BC30 tier-P bound) on real Swiss CPU-WRF h06 operands
    (tests/fixtures/o2_moistadv_swiss_h06.npz, proofs/o2_moistadv/build_fixture.py); options 1/2 do not
    (mutant: dispatching ORIGINAL to the PD/mono stencil fails).
(2) operational dispatch: with flux advection on, option 0 moves every moisture species in the real
    operational step (deletion-sensitive: the <= v0.3.4 gate leaves them untouched), conserves coupled
    moisture mass on the periodic warm bubble, and differs from option 1 (no limiter / no PD pre-add).
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.flux_advection import CoupledVelocities, advect_moisture_scalars  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/o2_moistadv_swiss_h06.npz"
GFORTRAN = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    if not (GFORTRAN.exists() and PRISTINE.exists()):
        pytest.skip("pristine WRF source / wrfbuild gfortran not available")
    sys.path.insert(0, str(HERE / "v025/b_core"))
    try:
        from bench_flux import error
        from scalar_oracle import build, evaluate
    finally:
        sys.path.remove(str(HERE / "v025/b_core"))
    root = tmp_path_factory.mktemp("advect_scalar")
    libs = {mode: build(root / mode, mode) for mode in ("off", "fast")}
    yield libs, evaluate, error
    shutil.rmtree(root, ignore_errors=True)


def _port(d, name, opt, final, dt=18.0):
    f = lambda k: jnp.asarray(d[k], jnp.float32)  # noqa: E731
    vel = CoupledVelocities(ru=f("ru"), rv=f("rv"), rom=f("rom"), msftx=f("map"), specified=True)
    rdx = float(d["rdx"])
    return np.asarray(advect_moisture_scalars(
        (f(name),), (f(name),), vel, moist_adv_opt=opt, is_final_rk_stage=final, mut=f("mu"),
        mu_old=f("mu"), c1=f("c1"), c2=f("c2"), rdx=rdx, rdy=rdx, rdzw=f("rdzw"), fzm=f("fzm"),
        fzp=f("fzp"), dt=dt, msfty=f("map"))[0])


@pytest.mark.parametrize("name", ["qv", "qc"])
def test_original_final_stage_matches_pristine_advect_scalar(oracle, name):
    libs, evaluate, error = oracle
    d = dict(np.load(FIXTURE))
    arrays = {k: d[k] for k in ("ru", "rv", "rom", "mu", "map", "fzm", "fzp", "rdzw", "c1", "c2")}
    arrays["q"] = d[name]
    ref, fast = evaluate(libs["off"], arrays, float(d["rdx"])), evaluate(libs["fast"], arrays, float(d["rdx"]))
    assert np.count_nonzero(ref) > 0
    got = _port(d, name, opt=0, final=True)
    result = error(ref, got, fast, ref)
    assert result["passed"], result
    # ORIGINAL is stage-independent: the final stage is the same plain stencil as RK1/RK2.
    np.testing.assert_array_equal(got, _port(d, name, opt=0, final=False))
    if name == "qc":  # sparse condensate: the PD/mono limiters are active -> must NOT match ORIGINAL
        for opt in (1, 2):
            assert not error(ref, _port(d, name, opt=opt, final=True), fast, ref)["passed"], opt


# --------------------------------------------------------------------------- #
# Operational dispatch on the real _physics_boundary_step (warm bubble, CPU). #
# --------------------------------------------------------------------------- #
def _setup_and_blob():
    from gpuwrf.ic_generators.idealized import build_warm_bubble_setup

    setup = build_warm_bubble_setup(require_gpu=False)
    s = setup.state.replace(u=jnp.full_like(setup.state.u, 30.0))  # uniform +x wind (cf. downstream test)
    nz, ny, nx = s.qv.shape
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing="ij")
    blob = jnp.exp(-(((xx - nx / 2) / (nx / 6.0)) ** 2 + ((yy - ny / 2) / max(ny / 6.0, 1.0)) ** 2
                     + ((zz - nz / 2) / (nz / 4.0)) ** 2)).astype(s.qv.dtype)
    state = s.replace(**{n: getattr(s, n) + a * 5.0e-3 * blob
                         for n, a in zip(om._MOISTURE_SPECIES, (1.0, 0.5, 0.2, 0.1, 0.05, 0.02), strict=True)})
    return setup, state


def _run(state, namelist, steps=3):
    fn = jax.jit(lambda c, s: om._physics_boundary_step(c, namelist, s, run_radiation=False))
    carry = om.initial_operational_carry(state)
    for k in range(steps):
        carry = fn(carry, jnp.asarray(k, dtype=jnp.int32))
    return carry.state


def test_original_option_transports_moisture_in_the_operational_step():
    setup, state = _setup_and_blob()
    nml = lambda opt: dataclasses.replace(setup.namelist, moist_adv_opt=opt, use_flux_advection=True,  # noqa: E731
                                          run_physics=False, run_boundary=False)
    orig, pd = _run(state, nml(0)), _run(state, nml(1))
    mass0 = np.asarray(state.mu_total)
    for name in om._MOISTURE_SPECIES:
        before, after = np.asarray(getattr(state, name)), np.asarray(getattr(orig, name))
        # <= v0.3.4: option 0 left every species byte-identical (no dycore transport).
        assert np.max(np.abs(after - before)) > 1e-3 * np.max(np.abs(before)), name
        assert np.all(np.isfinite(after)), name
        # moisture mass is conserved by the periodic flux form to the same level as option 1
        # (measured with mu_total, not the hybrid c1*mu+c2 weight: ~3e-8 relative drift).
        m_before = float(np.sum(before * mass0[None]))
        drift = abs(float(np.sum(after * np.asarray(orig.mu_total)[None])) - m_before)
        drift_pd = abs(float(np.sum(np.asarray(getattr(pd, name)) * np.asarray(pd.mu_total)[None])) - m_before)
        assert drift <= 1e-6 * max(abs(m_before), 1.0), (name, m_before, drift)
        assert drift <= 10.0 * drift_pd + 1e-9 * abs(m_before), (name, drift, drift_pd)
    # option 1 differs only through the final-stage limiter / PD pre-add: same transport scale.
    qv0, qv1 = np.asarray(orig.qv), np.asarray(pd.qv)
    assert np.max(np.abs(qv0 - qv1)) < 0.5 * np.max(np.abs(qv0 - np.asarray(state.qv)))
