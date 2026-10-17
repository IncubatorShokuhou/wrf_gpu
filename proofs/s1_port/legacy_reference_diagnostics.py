"""Retained failing legacy-reference diagnostics, NOT portability acceptance.

Manager decision 2026-10-13: old acoustic/PD paths are not WRF-faithful.
Explicit invocation only: pytest proofs/s1_port/legacy_reference_diagnostics.py
Raw failed receipts PORT12--17 are retained; no legacy model equations changed.
"""
from pathlib import Path
import sys
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tests"))
from test_portability_kernels import equal, real_dycore, transport

@pytest.mark.parametrize("specified", [False, True])
def test_acoustic_real_reference(real_dycore, monkeypatch, specified):
    from dataclasses import replace
    from types import SimpleNamespace
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
    from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core
    from gpuwrf.dynamics.core import acoustic as acoustic_reference
    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
    from gpuwrf.kernels import dyn_acoustic_fp32 as fast

    # Rebuild a real work state through the production caller mapping. Historical
    # b-core acoustic NPZs were cleaned; no hidden lane artifact dependency.
    s, base, metrics, ns = real_dycore
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "1")
    monkeypatch.setenv("GPUWRF_ACOUSTIC_W2PASS", "3")
    monkeypatch.setenv("GPUWRF_ACOUSTIC_MASS_BLOCK", "1")
    monkeypatch.setenv("GPUWRF_ACOUSTIC_UV_MASKED", "1")
    monkeypatch.setenv("GPUWRF_ACOUSTIC_W_RECUR_SL", "1")
    ns = replace(ns, metrics=metrics, force_fp64=False)
    carry = op._initial_carry_for_run(s, ns).replace(base_state=base)
    reference = s.replace(_cast=False, theta=s.theta + jnp.float32(.001),
                          u=s.u + jnp.float32(.0001), v=s.v - jnp.float32(.0001))
    prep = small_step_prep_wrf(s, 2, ns.dt_s, metrics=metrics,
                              reference_state=reference, base_state=base)
    pressure = calc_p_rho_wrf(prep, step=0)
    work = op._acoustic_core_state_from_prep(carry, prep, pressure, ns, ns.tendencies)
    work = jax.tree.map(lambda x: x.astype(jnp.float32)
                        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, work)
    work = work.replace(mu_work=prep.mu_work.astype(jnp.float32))
    cfg = AcousticCoreConfig(dt=2., dx=ns.grid.projection.dx_m, dy=ns.grid.projection.dy_m,
        top_lid=True, periodic_x=not specified, specified=specified, damp_opt=3, dampcoef=.2)
    options = {"xla_disable_hlo_passes": "algsimp"}
    coef = jax.jit(lambda w: fast.calc_coef_fp32(w, cfg), compiler_options=options)(work)
    m = SimpleNamespace(c1h=work.c1h, c2h=work.c2h, c1f=work.c1f,
                        c2f=work.c2f, rdn=work.rdn, rdnw=work.rdnw)
    ref_coef = jax.jit(lambda w: calc_coef_w_wrf_coefficients(w.mut, m,
        dt=cfg.dt, epssm=cfg.epssm, top_lid=True, cqw=w.cqw, c2a=w.c2a),
        compiler_options=options)(work)
    for key, value in zip(("a", "alpha", "gamma"), ref_coef):
        assert np.isfinite(value).all(), key
        assert coef[key].dtype == jnp.float32, key
        equal(coef[key], np.asarray(value, np.float32))
    actual = jax.jit(lambda w: fast.acoustic_substep_fp32(w, coefficients=coef, cfg=cfg),
                     compiler_options=options)(work)
    # The retained advance_uv helper has no specified/nested argument and
    # updates the whole domain (known B13). Supply the original WRF caller's
    # update rectangle, module_small_step_em.F:735-751, BEFORE downstream use.
    # This defines the reference operator's bounds; no model mask is changed.
    import types
    def owned_uv(w, *args, **kwargs):
        r = acoustic_reference.advance_uv_wrf(w, *args, **kwargs)
        if cfg.specified or cfg.nested:
            ny, nx = w.theta.shape[-2:]
            y, x = jnp.arange(ny)[:, None], jnp.arange(nx + 1)[None, :]
            own_u = (y >= cfg.spec_zone) & (y < ny - cfg.spec_zone)
            if not cfg.periodic_x:
                own_u = own_u & (x >= cfg.spec_zone) & (x <= nx - cfg.spec_zone)
            y, x = jnp.arange(ny + 1)[:, None], jnp.arange(nx)[None, :]
            own_v = (y >= cfg.spec_zone) & (y <= ny - cfg.spec_zone)
            if not cfg.periodic_x:
                own_v = own_v & (x >= cfg.spec_zone) & (x < nx - cfg.spec_zone)
            r = r.replace(u=jnp.where(own_u[None], r.u, w.u),
                          v=jnp.where(own_v[None], r.v, w.v))
        return r
    reference_step = types.FunctionType(acoustic_substep_core.__code__,
        {**acoustic_substep_core.__globals__, "advance_uv_wrf": owned_uv},
        acoustic_substep_core.__name__, acoustic_substep_core.__defaults__)
    reference_step.__kwdefaults__ = acoustic_substep_core.__kwdefaults__
    expected = jax.jit(lambda w: reference_step(w, a=coef["a"],
        alpha=coef["alpha"], gamma=coef["gamma"], cfg=cfg, cqw=w.cqw),
        compiler_options=options)(work)
    assert actual.to_dict().keys() == expected.to_dict().keys()
    failures = []
    for key, value in actual.to_dict().items():
        if value is not None:
            reference_value = np.asarray(expected.to_dict()[key])
            assert np.isfinite(reference_value).all(), key
            # The retained acoustic reference explicitly widens several work
            # leaves. Check native REAL storage and compare at REAL precision.
            if jnp.issubdtype(value.dtype, jnp.floating):
                assert value.dtype == jnp.float32, key
            a, b = np.asarray(value), reference_value.astype(value.dtype)
            assert np.isfinite(a).all() and np.isfinite(b).all(), key
            if np.issubdtype(a.dtype, np.floating):
                delta = np.abs(a - b)
                bound = 64 * np.finfo(a.dtype).eps * max(np.max(np.abs(b)), 1e-30)
                if np.max(delta) > bound:
                    index = np.unravel_index(np.argmax(delta), delta.shape)
                    failures.append(dict(field=key, max_error=float(np.max(delta)), bound=float(bound),
                        index=[int(i) for i in index], native=float(a[index]), reference=float(b[index]),
                        input=float(np.asarray(work.to_dict()[key])[index])))
            else:
                assert np.array_equal(a, b), key
    import json
    import os
    from pathlib import Path
    if os.environ.get("PORT_ACOUSTIC_DIAG"):
        Path(os.environ["PORT_ACOUSTIC_DIAG"]).write_text(json.dumps(failures, indent=2) + "\n")
    assert not failures, failures


@pytest.mark.parametrize("species_loop", ["0", "1"])
def test_positive_definite_flux_reference(transport, monkeypatch, species_loop):
    from dataclasses import replace
    from gpuwrf.dynamics import flux_advection as ref
    from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32

    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "0")
    monkeypatch.setenv("GPUWRF_DYN_PD_FP32", "0")
    monkeypatch.setenv("GPUWRF_DYN_PD_SPECIES_LOOP", species_loop)
    rand, vel, vectors = transport
    vel = replace(vel, ru=vel.ru * jnp.float32(1e6), rv=vel.rv * jnp.float32(1e6),
                  rom=vel.rom * jnp.float32(100), msftx=jnp.ones_like(vel.msftx))
    # The retained limiter's documented low-order reference scope is unit map.
    # Non-unit-map native qualification belongs to the pristine PD oracle.
    q = jnp.abs(jnp.stack([rand(vel.ru.shape) for _ in range(3)])) * jnp.float32(1e-4)
    q = q.at[:, :, ::3, ::3].set(0)  # Wet/dry transitions exercise PD flux scaling.
    mu = jnp.full(q.shape[-2:], 90000, jnp.float32)
    c1, c2 = jnp.ones(q.shape[1], jnp.float32), jnp.zeros(q.shape[1], jnp.float32)
    options = {"xla_disable_hlo_passes": "algsimp"}
    actual = jax.jit(lambda x: advect_scalar_pd_fp32(x, x, vel.ru, vel.rv, vel.rom,
        mu, mu, c1, c2, vel.msftx, vel.msftx, vectors["rdzw"], vectors["fzm"], vectors["fzp"],
        rdx=.001, rdy=.002, dt=18), compiler_options=options)(q)
    for field, value in zip(q, actual):
        expected = jax.jit(lambda x: ref.advect_scalar_flux_limited(x, x, vel,
            scalar_adv_opt=1, mut=mu, mu_old=mu, c1=c1, c2=c2, rdx=.001, rdy=.002,
            dt=18, **vectors), compiler_options=options)(field)
        import json
        import os
        from pathlib import Path
        if os.environ.get("PORT_PD_DIAG"):
            a, b = np.asarray(value), np.asarray(expected)
            delta = np.abs(a - b)
            k, y, x = np.unravel_index(np.argmax(delta), delta.shape)
            rows = dict(max_error=float(delta.max()), index=[int(k), int(y), int(x)],
                native=float(a[k,y,x]), reference=float(b[k,y,x]), input=float(field[k,y,x]),
                native_norm=float(np.max(np.abs(a))), reference_norm=float(np.max(np.abs(b))),
                interior_max=float(delta[:,1:-1,1:-1].max()),
                by_level=[float(v) for v in np.max(delta, axis=(1,2))])
            Path(os.environ["PORT_PD_DIAG"]).write_text(json.dumps(rows, indent=2) + "\n")
        equal(value, expected)
