"""Native facade/direct-interpreter parity plus supplemental valid stencil checks.

These tests are NOT CPU-WRF fidelity or GPU performance evidence. Scale-aware
REAL tolerances allow CPU contraction/summation order; all values must be finite.
Existing pristine-oracle tests cover physics/radiation alongside this file.
Old full acoustic/PD comparisons are retained as diagnostics under proofs/s1_port;
they are not portability acceptance references (manager decision 2026-10-13).
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def equal(actual, reference, eps=64):
    a, b = np.asarray(actual), np.asarray(reference)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.isfinite(a).all() and np.isfinite(b).all()
    bound = eps * np.finfo(a.dtype).eps * max(np.max(np.abs(b)), 1e-30)
    assert np.max(np.abs(a - b)) <= bound, (np.max(np.abs(a - b)), bound)


@pytest.fixture
def transport():
    from gpuwrf.dynamics.flux_advection import CoupledVelocities

    rng = np.random.default_rng(8153)
    nz, ny, nx = 8, 9, 11
    rand = lambda shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
    u, v = rand((nz, ny, nx + 1)), rand((nz, ny + 1, nx))
    w = rand((nz + 1, ny, nx)).at[0].set(0).at[-1].set(0)
    mx = jnp.asarray(rng.uniform(.8, 1.2, (ny, nx)), jnp.float32)
    vel = CoupledVelocities(ru=u[..., :-1], rv=v[:, :-1], rom=w,
        msftx=mx,
        msfux=jnp.ones((ny, nx + 1), jnp.float32),
        msfvx=jnp.ones((ny + 1, nx), jnp.float32),
        msfvy=jnp.ones((ny + 1, nx), jnp.float32),
        specified=True, ru_full=u, rv_full=v)
    vectors = dict(rdzw=jnp.linspace(1, 2, nz, dtype=jnp.float32),
                   fzm=jnp.full(nz + 1, .45, jnp.float32),
                   fzp=jnp.full(nz + 1, .55, jnp.float32))
    return rand, vel, vectors


@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("upstream", [False, True])
def test_specified_flux_faces(transport, axis, upstream):
    from gpuwrf.dynamics.flux_advection import specified_flux_faces
    from gpuwrf.kernels.dyn_flux_fp32 import specified_flux_faces_fp32

    rand, vel, _ = transport
    q = rand(vel.ru.shape)
    speed = vel.ru if axis == 2 else vel.rv
    equal(jax.jit(partial(specified_flux_faces_fp32, axis=axis, upstream=upstream))(q, speed),
          jax.jit(partial(specified_flux_faces, axis=axis, upstream=upstream))(q, speed))


def test_scalar_flux(transport, monkeypatch):
    from gpuwrf.dynamics import flux_advection as ref
    from gpuwrf.kernels.dyn_flux_fp32 import advect_scalar_flux_fp32

    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "0")
    rand, vel, vectors = transport
    q = rand(vel.ru.shape)
    args = dict(mut=jnp.full(q.shape[1:], 90000, jnp.float32),
                c1=jnp.ones(q.shape[0], jnp.float32), rdx=.001, rdy=.002, **vectors)
    expected = jax.jit(lambda q: ref.advect_scalar_flux(q, vel, **args))(q)
    actual = jax.jit(lambda q: advect_scalar_flux_fp32(q, vel.ru, vel.rv, vel.rom,
        vel.msftx, vectors["rdzw"], vectors["fzm"], vectors["fzp"], .001, .002))(q)
    equal(actual, expected)


@pytest.mark.parametrize("component", ["u", "v", "w"])
def test_momentum_flux(transport, monkeypatch, component):
    from gpuwrf.dynamics import flux_advection as ref
    from gpuwrf.kernels import dyn_momflux_fp32 as fast

    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "1")
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "0")
    rand, vel, vectors = transport
    shape = {"u": vel.ru_full.shape, "v": vel.rv_full.shape, "w": vel.rom.shape}[component]
    q = rand(shape)
    if component == "u":
        fn = lambda q: fast.advect_u_fp32(q, vel.ru_full, vel.rv_full, vel.rom,
             vel.msfux, vectors["rdzw"], vectors["fzm"], vectors["fzp"], .001, .002)
    elif component == "v":
        fn = lambda q: fast.advect_v_fp32(q, vel.ru_full, vel.rv_full, vel.rom,
             vel.msfvy, vel.msfvx, vectors["rdzw"], vectors["fzm"], vectors["fzp"], .001, .002)
    else:
        rdzw = vectors["rdzw"]
        fn = lambda q: fast.advect_w_fp32(q, vel.ru, vel.rv, vel.rom,
             vel.msftx, rdzw, vectors["fzm"], vectors["fzp"], .001, .002)
        vectors = {**vectors, "rdn": vectors["rdzw"]}
        vectors.pop("rdzw")
    expected = jax.jit(lambda q: getattr(ref, "advect_" + component + "_flux")(
        q, vel, rdx=.001, rdy=.002, **vectors))(q)
    equal(jax.jit(fn)(q), expected)


def test_stage_omega(transport, monkeypatch):
    from gpuwrf.dynamics.flux_advection import stage_omega_specified
    from gpuwrf.kernels.dyn_omega_fp32 import stage_omega_fp32

    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "0")
    _, vel, _ = transport
    nz, ny, nx = vel.ru.shape
    mu = jnp.full((ny, nx), 90000, jnp.float32)
    c1, c2 = jnp.linspace(.1, 1, nz, dtype=jnp.float32), jnp.zeros(nz, jnp.float32)
    dnw = jnp.full(nz, -1 / nz, jnp.float32)
    expected = jax.jit(lambda u, v: stage_omega_specified(u, v, mu, c1h=c1,
        c2h=c2, dnw=dnw, msfuy=vel.msfux, msfvx=vel.msfvx, msftx=vel.msftx,
        rdx=.001, rdy=.002))(vel.ru_full, vel.rv_full)
    actual = jax.jit(lambda u, v: stage_omega_fp32(u, v, mu, c1, c2, dnw,
        vel.msfux, vel.msfvx, vel.msftx, .001, .002))(vel.ru_full, vel.rv_full)
    equal(actual, expected)


@pytest.mark.parametrize("monotonic", [False, True])
def test_sixth_order_diffusion(transport, monkeypatch, monotonic):
    from gpuwrf.dynamics import explicit_diffusion as ref
    from gpuwrf.kernels.dyn_diff_fp32 import sixth_order_fp32

    monkeypatch.setattr(ref, "_NATIVE_DIFFUSION_FP32", False)
    rand, vel, _ = transport
    q = rand(vel.ru.shape)
    mu = jnp.full(q.shape[1:], 90000, jnp.float32)
    c1, c2 = jnp.ones(q.shape[0], jnp.float32), jnp.zeros(q.shape[0], jnp.float32)
    expected = jax.jit(lambda q: ref.wrf_sixth_order_scalar_tendf(q, mu,
        c1=c1, c2=c2, msftx=vel.msftx, msfty=vel.msftx, dt=18,
        diff_6th_factor=.12, monotonic=monotonic, specified_or_nested=True))(q)
    actual = jax.jit(lambda q: sixth_order_fp32(q, mu, c1, c2,
        vel.msftx, vel.msftx, name="m", dt=18, factor=.12, monotonic=monotonic))(q)
    equal(actual, expected)


@pytest.mark.parametrize("name", ["theta", "u", "v", "w"])
def test_coupled_work_numpy_reference(transport, name):
    from gpuwrf.kernels.dyn_rk_fp32 import coupled_work_fp32

    rand, vel, _ = transport
    cur, ref = rand(vel.ru.shape), rand(vel.ru.shape)
    mass = np.full(cur.shape[1:], 90000, np.float32)
    c1, c2 = np.ones(cur.shape[0], np.float32), np.zeros(cur.shape[0], np.float32)
    scale = np.asarray(vel.msftx)
    expected = mass * np.asarray(ref) - mass * np.asarray(cur)
    if name == "v":
        expected = expected * (np.float32(1) / scale)
    elif name != "theta":
        expected = expected / scale
    actual = jax.jit(lambda a, b: coupled_work_fp32(a, b, jnp.asarray(mass),
        jnp.asarray(mass), jnp.asarray(c1), jnp.asarray(c2), vel.msftx, name=name))(ref, cur)
    equal(actual, expected, eps=8)


def test_mcica_lane_tail_matches_jump_stream():
    from gpuwrf.kernels import rad_mcica as m

    p = jnp.asarray([[99000.13, 70000.37, 30000.21, 9000.43],
                     [101000.17, 71000.23, 31000.45, 10000.33]], jnp.float32)
    seeds = m._pressure_seeds(p)
    first, ng, nz = 151, 140, p.shape[-1]
    words = m.kiss_jump_stream(seeds, first=first, count=ng * nz)
    # Both conversions compiled: eager mul then add and compiled FMA differ by
    # one REAL ulp even for identical integer words.
    expected = jax.jit(lambda w: jax.lax.bitcast_convert_type(w, jnp.int32).astype(jnp.float32)
                       * jnp.float32(2.328306e-10) + jnp.float32(.5))(words)
    expected = jnp.moveaxis(expected.reshape((ng, nz) + p.shape[:-1]), (0, 1), (-1, -2))
    actual = jax.jit(lambda *s: m.kiss_random_lanes(s, first=first, ng=ng, nlay=nz))(*seeds)
    equal(actual, expected, eps=0)


@pytest.fixture
def real_dycore(monkeypatch):
    """Fresh real PROD crop; this is operand provenance, not WRF output truth."""
    import importlib.util
    from pathlib import Path
    from gpuwrf.contracts.state import BaseState
    from gpuwrf.kernels import dyn_rk_fp32 as native_rk
    from gpuwrf.kernels.dyn_rk_fp32 import real_state, real_metrics
    from gpuwrf.dynamics.core import rk_addtend_dry as rk

    for key in ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_DYN_CARRY_FP32", "GPUWRF_DYN_REAL_ALL"):
        monkeypatch.setenv(key, "1")
    # Full-suite collection already imported rk in legacy mode. Select its
    # intended native import-time branch locally without reloading classes or
    # leaking flags/helpers to the next test.
    monkeypatch.setattr(rk, "_NATIVE_RK_FP32", True)
    monkeypatch.setattr(rk, "_RK_WORK_DTYPE", jnp.float32)
    for name in ("real_state", "real_metrics", "real_base", "inverse_density_fp32", "diagnose_pressure_fp32"):
        monkeypatch.setattr(rk, name, getattr(native_rk, name), raising=False)
    monkeypatch.setattr(rk, "_inverse_density_from_theta_pressure", native_rk.inverse_density_fp32)
    jax.clear_caches()
    path = Path(__file__).parent / "v025/b_core/test_root_scalar_stacking.py"
    spec = importlib.util.spec_from_file_location("port_real_operands", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns, carry = module._fixture()
    original = carry.state
    state = real_state(original)
    base = BaseState(pb=(original.p_total - original.p_perturbation).astype(jnp.float32),
        phb=(original.ph_total - original.ph_perturbation).astype(jnp.float32),
        mub=(original.mu_total - original.mu_perturbation).astype(jnp.float32),
        t0=jnp.full_like(state.theta, 300), theta_base=jnp.full_like(state.theta, 300))
    try:
        yield state, base, real_metrics(ns.metrics), ns
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("option", [1, 2])
def test_eos_real_reference(real_dycore, option):
    from gpuwrf.kernels.dyn_rk_fp32 import diagnose_pressure_fp32, _diagnose_pressure_reference

    state, base, metrics, _ = real_dycore
    # E130: CPU algsimp reassociates divisions across cancellation differently
    # in the two compilation contexts. This is a parity-probe option only.
    options = {"xla_disable_hlo_passes": "algsimp"}
    actual = jax.jit(lambda s: diagnose_pressure_fp32(s, base, metrics, hypsometric_opt=option),
                     compiler_options=options)(state)
    reference = jax.jit(lambda s: _diagnose_pressure_reference(s, base, metrics, hypsometric_opt=option),
                        compiler_options=options)(state)
    for index, (a, b) in enumerate(zip(actual, reference)):
        # Perturbation pressure involves subtracting the large base field.
        equal(a + base.pb if index == 0 else a,
              b + base.pb if index == 0 else b)


@pytest.mark.parametrize("top_lid", [False, True])
def test_large_step_uv_real_reference(real_dycore, monkeypatch, top_lid):
    from gpuwrf.dynamics.core import rk_addtend_dry as ref

    # Release defaults carry GLUE_FUSED part "pin" (row-major with_layout_constraint on the fused operands,
    # layout only). XLA:CPU erases that LayoutConstraint custom call in algsimp, which this parity probe must
    # disable (E130) -> "no registered implementation ... LayoutConstraint for Host". Product programs never
    # disable algsimp; drop only "pin" here, every other release glue part stays as resolved (w44-redfix B).
    parts = [p for p in os.environ.get("GPUWRF_DYN_GLUE_FUSED", "0").split("_") if p != "pin"]
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "_".join(parts) or "0")
    state, base, m, ns = real_dycore
    zero_u, zero_v = jnp.zeros_like(state.u), jnp.zeros_like(state.v)
    args = dict(dx_m=float(ns.grid.projection.dx_m), dy_m=float(ns.grid.projection.dy_m), top_lid=top_lid,
                hypsometric_opt=2, base_state=base)
    options = {"xla_disable_hlo_passes": "algsimp"}  # E130, as in EOS above.
    actual = jax.jit(lambda s: ref.large_step_uv_fused(s, m, zero_u, zero_v, **args),
                     compiler_options=options)(state)

    def reference(s):
        pu, pv = ref.large_step_horizontal_pgf(s, m, **args)
        cu, cv = ref.large_step_coriolis(s, m, specified=True)
        ku, kv = ref.large_step_horizontal_curvature(s, m,
            dx_m=float(ns.grid.projection.dx_m), dy_m=float(ns.grid.projection.dy_m), specified=True)
        return (zero_u + pu) + cu + ku, (zero_v + pv) + cv + kv

    for a, b in zip(actual, jax.jit(reference, compiler_options=options)(state)):
        equal(a, b)


@pytest.mark.parametrize("top_lid", [False, True])
def test_rhs_ph_real_reference(real_dycore, monkeypatch, top_lid):
    from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf
    from gpuwrf.kernels.dyn_rhsph_fp32 import rhs_ph_fp32
    from gpuwrf.dynamics.core.small_step_prep import _u_face_average_2d, _v_face_average_2d

    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "0")
    s, b, m, ns = real_dycore
    rng = np.random.default_rng(191)
    ww = jnp.asarray(rng.normal(size=s.w.shape), jnp.float32)
    muu, muv = _u_face_average_2d(s.mu_total), _v_face_average_2d(s.mu_total)
    rdx, rdy = 1 / float(ns.grid.projection.dx_m), 1 / float(ns.grid.projection.dy_m)
    args = dict(u=s.u, v=s.v, ww=ww, ph=s.ph_perturbation, phb=b.phb,
        w=s.w, mut=s.mu_total, muu=muu, muv=muv, c1f=m.c1f, c2f=m.c2f,
        fnm=m.fnm, fnp=m.fnp, rdnw=m.rdnw, msfty=m.msfty,
        msfux=m.msfux, msfvy=m.msfvy, rdx=rdx, rdy=rdy, top_lid=top_lid,
        cfn=1.5, cfn1=-.5)
    actual = jax.jit(lambda: rhs_ph_fp32(**args, g=9.81))()
    expected = jax.jit(lambda: rhs_ph_wrf(**args, gravity=9.81,
        advective_order=5, specified=True))()
    equal(actual, expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_parity_gate_rejects_nonfinite(bad):
    with pytest.raises(AssertionError):
        equal(np.array([bad], np.float32), np.ones(1, np.float32))


@pytest.mark.parametrize("unstable", [False, True])
def test_boulac_matches_jax_recurrence(unstable, monkeypatch):
    from gpuwrf.kernels.phys_mynn_boulac import boulac_length_native
    from gpuwrf.physics.mynn_pbl import _boulac_length_onz
    from gpuwrf.physics import mynn_constants

    # Match the native kernel's WRF REAL PARAMETER, rather than the legacy
    # double-rounded GTR used by the default CPU test environment.
    monkeypatch.setattr(mynn_constants, "_MYNN_REAL_CONSTANTS", True)

    n, nz = 3, 9
    rng = np.random.default_rng(987)
    dz = jnp.asarray(rng.uniform(20, 80, (n, nz)), jnp.float32)
    zw = jnp.cumsum(dz, axis=-1) - dz
    theta = jnp.asarray(290 + np.arange(nz)[None] * (-.1 if unstable else .3)
                        + rng.uniform(-.1, .1, (n, nz)), jnp.float32)
    qtke = jnp.asarray(rng.uniform(.01, 5, (n, nz)), jnp.float32)
    opts = {"xla_disable_hlo_passes": "algsimp"}
    actual = jax.jit(boulac_length_native, compiler_options=opts)(zw, dz, qtke, theta)
    expected = jax.jit(_boulac_length_onz, compiler_options=opts)(zw, dz, qtke, theta)
    for a, b in zip(actual, expected):
        equal(a, b)


@pytest.mark.parametrize("column", [0, 3, 7])
def test_noahmp_radiation_matches_separate_jax(column, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from gpuwrf.kernels.phys_noahmp_radiation import radiation_native
    from gpuwrf.physics.noahmp.energy_radiation import _radiation_twostream_impl
    from gpuwrf.physics.noahmp.precision import real_tree

    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    root = Path(__file__).resolve().parents[1]
    path = root / "proofs/noahmp/energy_savepoint_gate.py"
    spec = importlib.util.spec_from_file_location("port_noah_operands", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    col = json.loads((path.parent / "savepoints_energy.json").read_text())["columns"][column]
    _, params = gate.build_params(col["vegtyp"], col["isltyp"])
    args = real_tree((gate.build_state(col), gate.build_forcing(col),
                     gate.build_static(col), gate.build_phen(col), params))
    actual = radiation_native(*args, gate.DT)
    expected = _radiation_twostream_impl(*args, gate.DT)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        equal(a, b)




@pytest.mark.parametrize("lead", [0., 18., 54., 90.])
def test_boundary_clock_literal_numpy(lead):
    from gpuwrf.kernels.dyn_boundary_fp32 import interpolate_leaf

    rng = np.random.default_rng(114)
    leaf = rng.uniform(-2, 2, (3, 4, 3, 7, 13)).astype(np.float32)
    cadence = np.float32(54)
    fraction = np.float32(lead) / cadence
    lower = int(np.clip(np.ceil(fraction) - 1, 0, len(leaf) - 2))
    alpha = np.clip(fraction - np.float32(lower), np.float32(0), np.float32(1))
    expected = leaf[lower] + (alpha * cadence) * ((leaf[lower + 1] - leaf[lower]) / cadence)
    actual = jax.jit(lambda x: interpolate_leaf(x, np.float32(lead), cadence),
                     compiler_options={"xla_disable_hlo_passes": "algsimp"})(leaf)
    equal(actual, expected, eps=8)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_boundary_work_literal_numpy(axis):
    from gpuwrf.kernels.dyn_boundary_fp32 import coupled_work_target

    rng = np.random.default_rng(119)
    nz, ny, nx, zone = 3, 7, 9, 2
    leaf = rng.uniform(-5, 5, (4, zone, nz, max(ny, nx))).astype(np.float32)
    save = rng.uniform(-2, 2, (nz, ny, nx)).astype(np.float32)
    mass = rng.uniform(70000, 100000, save.shape).astype(np.float32)
    scale = rng.uniform(.8, 1.2, (ny, nx)).astype(np.float32)
    expected = np.zeros_like(save)
    for j in range(ny):
        for i in range(nx):
            distance = min(i, nx - 1 - i) if axis == "x" else min(j, ny - 1 - j)
            if distance < zone:
                side = (0 if i < nx // 2 else 1) if axis == "x" else (2 if j < ny // 2 else 3)
                tangent = j if axis == "x" else i
                expected[:, j, i] = leaf[side, distance, :, tangent] - (
                    save[:, j, i] * mass[:, j, i]) / scale[j, i]
    actual = jax.jit(lambda a, b, c, d: coupled_work_target(a, b, c, d, axis=axis, zone=zone),
                     compiler_options={"xla_disable_hlo_passes": "algsimp"})(leaf, save, mass, scale)
    equal(actual, expected, eps=8)


def test_horizontal_diffusion_jax_reference(transport, monkeypatch):
    from gpuwrf.dynamics import explicit_diffusion as ref
    from gpuwrf.kernels.dyn_diff_fp32 import horizontal_diffusion_fp32

    monkeypatch.setattr(ref, "_NATIVE_DIFFUSION_FP32", False)
    rand, vel, _ = transport
    nz, ny, nx = vel.ru.shape
    fields = (rand(vel.ru_full.shape), rand(vel.rv_full.shape), rand(vel.rom.shape))
    kh = jnp.abs(rand(vel.ru.shape)) + jnp.float32(1)
    mu = jnp.full((ny, nx), 90000, jnp.float32)
    ch, zh = jnp.ones(nz, jnp.float32), jnp.zeros(nz, jnp.float32)
    cf, zf = jnp.ones(nz + 1, jnp.float32), jnp.zeros(nz + 1, jnp.float32)
    maps = (vel.msftx, vel.msftx * jnp.float32(1.03), vel.msfux,
            vel.msfux * jnp.float32(.99), vel.msfvx, vel.msfvy)
    tx, ty, ux, uy, vx, vy = maps
    options = {"xla_disable_hlo_passes": "algsimp"}
    expected = jax.jit(lambda u, v, w: ref.wrf_nested_horizontal_diffusion_momentum_tendency(
        u, v, w, kh, mu, c1h=ch, c2h=zh, c1f=cf, c2f=zf, msftx=tx, msfty=ty,
        msfux=ux, msfuy=uy, msfvx=vx, msfvy=vy, dx_m=9000, dy_m=3000),
        compiler_options=options)(*fields)
    for name, field, reference in zip(("u", "v", "w"), fields, expected):
        mass = jnp.broadcast_to(mu, (field.shape[0], ny, nx))
        actual = jax.jit(lambda q: horizontal_diffusion_fp32(q, kh, mass, *maps,
            name=name, dx=9000, dy=3000), compiler_options=options)(field)
        equal(actual, reference)
    scalar = rand(vel.ru.shape)
    mass = jnp.broadcast_to(mu, scalar.shape)
    actual = jax.jit(lambda q: horizontal_diffusion_fp32(q, kh, mass, *maps,
        name="m", dx=9000, dy=3000), compiler_options=options)(scalar)
    expected = jax.jit(lambda q: ref.horizontal_diffusion_coord_scalar_tendency(q, kh, mass,
        dx_m=9000, dy_m=3000, msftx=tx, msfty=ty, msfux=ux, msfuy=uy,
        msfvx=vx, msfvy=vy, nonperiodic_owned=True), compiler_options=options)(scalar)
    equal(actual, expected)


def test_species_stacked_flux_reference(transport, monkeypatch):
    from gpuwrf.dynamics import flux_advection as ref
    from gpuwrf.kernels.dyn_flux_fp32 import advect_scalar_flux_fp32_stacked

    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "0")
    rand, vel, vectors = transport
    fields = jnp.stack([rand(vel.ru.shape) for _ in range(3)])
    mass = jnp.full(fields.shape[-2:], 90000, jnp.float32)
    actual = jax.jit(lambda q: advect_scalar_flux_fp32_stacked(q, vel.ru, vel.rv, vel.rom,
        vel.msftx, vectors["rdzw"], vectors["fzm"], vectors["fzp"], .001, .002))(fields)
    for field, value in zip(fields, actual):
        expected = jax.jit(lambda q: ref.advect_scalar_flux(q, vel, mut=mass,
            c1=jnp.ones(q.shape[0], jnp.float32), rdx=.001, rdy=.002, **vectors))(field)
        equal(value, expected)




def test_acoustic_facade_matches_native_interpreter(real_dycore, monkeypatch):
    specified = True
    monkeypatch.setenv("GPUWRF_ACOUSTIC_ALIAS", "1")
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
    from jax.experimental import pallas as original_pallas
    from gpuwrf.backends import pallas as facade
    coef = jax.jit(lambda w: fast.calc_coef_fp32(w, cfg))(work)
    actual = jax.jit(lambda w: fast.acoustic_substep_fp32(w, coefficients=coef, cfg=cfg))(work)
    original = original_pallas.pallas_call
    calls = []
    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("name"))
        kwargs["interpret"] = True
        return original(*args, **kwargs)
    with monkeypatch.context() as control:
        control.setattr(facade, "pallas_call", interpreted, raising=False)
        ref_coef = jax.jit(lambda w: fast.calc_coef_fp32(w, cfg, interpret=True))(work)
        expected = jax.jit(lambda w: fast.acoustic_substep_fp32(w, coefficients=ref_coef,
            cfg=cfg, interpret=True))(work)
    assert {"b_core_coef_fp32", "b_core_uv_fp32", "b_core_mass_fp32", "b_core_pressure_fp32"} <= set(calls)
    assert any(str(name).startswith("b_core_w_") for name in calls)
    for name in coef:
        equal(coef[name], ref_coef[name])
    for name, value in actual.to_dict().items():
        if value is not None:
            equal(value, expected.to_dict()[name])


@pytest.mark.parametrize("species_loop", ["0", "1"])
def test_pd_facade_matches_native_interpreter(transport, monkeypatch, species_loop):
    from jax.experimental import pallas as original_pallas
    from gpuwrf.backends import pallas as facade
    from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32

    monkeypatch.setenv("GPUWRF_DYN_PD_SPECIES_LOOP", species_loop)
    rand, vel, vectors = transport
    q = jnp.abs(jnp.stack([rand(vel.ru.shape) for _ in range(3)])) * jnp.float32(1e-4)
    q = q.at[:, :, ::3, ::3].set(0)
    mu = jnp.full(q.shape[-2:], 90000, jnp.float32)
    c1, c2 = jnp.ones(q.shape[1], jnp.float32), jnp.zeros(q.shape[1], jnp.float32)
    def run(x, interpret):
        return advect_scalar_pd_fp32(x, x, vel.ru_full * jnp.float32(1e6),
            vel.rv_full * jnp.float32(1e6), vel.rom * jnp.float32(100),
            mu, mu, c1, c2, vel.msftx, vel.msftx * jnp.float32(1.03),
            vectors["rdzw"], vectors["fzm"], vectors["fzp"], rdx=.001, rdy=.002,
            dt=18, interpret=interpret)
    actual = jax.jit(lambda x: run(x, False))(q)
    original = original_pallas.pallas_call
    calls = []
    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("name"))
        kwargs["interpret"] = True
        return original(*args, **kwargs)
    with monkeypatch.context() as control:
        control.setattr(facade, "pallas_call", interpreted, raising=False)
        expected = jax.jit(lambda x: run(x, True))(q)
    assert len(calls) == 2
    equal(actual, expected)


def test_thompson_prep_facade_matches_native_interpreter(monkeypatch):
    import importlib.util
    from pathlib import Path
    from jax.experimental import pallas as original_pallas
    from gpuwrf.backends import pallas as facade
    from gpuwrf.kernels.phys_thompson_prep import sediment_and_finish

    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_SED_FP32", "1")
    root = Path(__file__).parent / "v025/b_thompson"
    spec = importlib.util.spec_from_file_location("port_thompson_operands", root / "test_mixed_phase_wrf_oracle.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    with np.load(root / "fixtures/mixed_phase_columns.npz") as data:
        state = fixtures._column_state(data, np.array([0, 1]))
    state = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), state)
    actual = jax.jit(lambda s: sediment_and_finish(s, 18.))(state)
    original = original_pallas.pallas_call
    calls = []
    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("name"))
        kwargs["interpret"] = True
        return original(*args, **kwargs)
    with monkeypatch.context() as control:
        control.setattr(facade, "pallas_call", interpreted, raising=False)
        expected = jax.jit(lambda s: sediment_and_finish(s, 18., interpret=True))(state)
    assert calls == ["thompson_sediment_prep_finish"]
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        equal(a, b)


@pytest.mark.parametrize("mode", ["dry", "ring", "ring_total"])
def test_boundary_stencils_facade_matches_native_interpreter(monkeypatch, mode):
    from jax.experimental import pallas as original_pallas
    from gpuwrf.backends import pallas as facade
    from gpuwrf.kernels import dyn_boundary_fp32 as boundary

    rng = np.random.default_rng(483)
    nz, ny, nx, width = 3, 12, 14, 5
    field = jnp.asarray(rng.normal(300, 1, (nz, ny, nx)), jnp.float32)
    leaf = jnp.asarray(rng.normal(301, 1, (2, 4, width, nz, nx)), jnp.float32)
    mu = jnp.full((ny, nx), 90000, jnp.float32)
    c1, c2 = jnp.ones(nz, jnp.float32), jnp.zeros(nz, jnp.float32)
    weights = jnp.linspace(.1, .01, width, dtype=jnp.float32)
    def run(x):
        if mode == "dry":
            return boundary.dry_relax(x, mu, c1, c2, jnp.ones_like(mu),
                leaf * jnp.float32(90000), jnp.float32(18), 54, weights, weights / 10,
                kind="theta", coupled=True, spec=1, relax=4)
        return boundary.ring_update(x, leaf, jnp.float32(18), 54, weights, weights / 10,
            spec=1, relax=4, do_relax=True, base=leaf * .01 if mode == "ring_total" else None)
    actual = jax.jit(lambda x: run(x))(field)
    original = original_pallas.pallas_call
    calls = []
    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("name"))
        kwargs["interpret"] = True
        kwargs.pop("compiler_params", None)
        return original(*args, **kwargs)
    with monkeypatch.context() as control:
        control.setattr(facade, "pallas_call", interpreted, raising=False)
        expected = jax.jit(lambda x: run(x))(field)
    assert calls == ["boundary_dry_relax_real" if mode == "dry" else "boundary_ring_update_real"]
    equal(actual, expected, eps=8)
    assert np.max(np.abs(np.asarray(actual))) > 0


@pytest.mark.parametrize("backend", ["cpu", "rocm", "tpu", "metal"])
def test_boundary_scalar_couple_numpy_and_non_cuda_rounding(monkeypatch, backend):
    from types import SimpleNamespace
    from gpuwrf.backends import pallas as facade, platform
    from gpuwrf.coupling import boundary_apply as boundary

    monkeypatch.setattr(platform, "runtime_backend", lambda: backend)
    rng = np.random.default_rng(742)
    shape = (7, 9, 11)
    field = rng.uniform(1e-5, 1e-3, shape).astype(np.float32)
    mu = rng.uniform(85000, 95000, shape[1:]).astype(np.float32)
    c1 = np.linspace(.1, 1, shape[0], dtype=np.float32)
    c2 = np.linspace(100, 5000, shape[0], dtype=np.float32)
    expected = field * (c1[:, None, None] * mu[None] + c2[:, None, None])
    arrays = tuple(jnp.asarray(x) for x in (field, mu, c1, c2))
    metrics = SimpleNamespace(c1h=arrays[2], c2h=arrays[3])
    boundary._couple_scalar_real4_arrays.clear_cache()
    actual = boundary.couple_scalar_real4(arrays[0], arrays[1], metrics)
    equal(actual, expected, eps=8)
    # Exercise the closure used by non-CPU accelerators: interpret=False must
    # still use portable RN arithmetic when CUDA is absent.
    fn = facade.pallas_call(partial(boundary._scalar_couple_kernel,
        nz=shape[0], ny=shape[1], nx=shape[2], interpret=False),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32), grid=((field.size + 255) // 256,),
        interpret=False)
    equal(jax.jit(fn)(*arrays), expected, eps=8)
    hlo = jax.jit(fn).lower(*arrays).as_text()
    assert "inline_asm" not in hlo and "triton_kernel_call" not in hlo
