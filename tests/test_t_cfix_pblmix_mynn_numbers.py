"""t-cfix-pblmix F8: MYNN mixes and applies QNC, QNWFA, QNIFA (WRF 4.7.1).

Oracle: WRF DMP_mf + mynn_tendencies from the pristine build objects, called with
the mp=28 caller flags on 16 REAL CPU-WRF c19 columns (QNCLOUD/QNWFA/QNIFA/QKE as
written by WRF); internal post-solve arrays from a private WRITE-only copy whose
MF/TEN output is byte-identical to the pristine objects
(proofs/t_cfix_pblmix/build_mynn_number_oracle.sh).  Gates mirror the QNI
precedent (tests/v025/b_phys/test_mynn_qni.py): scalar solve 8 ULP REAL32,
passive plume flux 5 % of the column maximum.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P, mynn_edmf as E
from gpuwrf.kernels import phys_mynn_plume as L
from gpuwrf.kernels.phys_mynn_numbers import AEROSOL_BOUNDS, mix_passive_numbers
from gpuwrf.kernels.phys_mynn_ni import mix_ice_number

ROOT = Path(__file__).resolve().parents[1]
FX = json.loads((ROOT / "proofs/t_cfix_pblmix/mynn_number_fixture.json").read_text())
C, S, W = FX["inputs"]["columns"], FX["inputs"]["scalars"], FX["wrf"]
DT = float(S["delt"][0])
NAMES = {"Nc": ("qnc", "qnc2", "s_awqnc"), "nwfa": ("qnwfa", "qnwfa2", "s_awqnwfa"),
         "nifa": ("qnifa", "qnifa2", "s_awqnifa")}


@pytest.fixture(autouse=True)
def clear_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def _a(src, name, dtype=np.float32):
    return np.asarray(src[name], dtype)


def ulp_gate(candidate, reference, label):
    assert np.isfinite(candidate).all() and np.isfinite(reference).all(), f"{label} nonfinite"
    exponent = (np.asarray(reference, np.float32).view(np.uint32) >> 23) & 255
    spacing = np.ldexp(np.ones(reference.shape), np.maximum(exponent.astype(np.int32), 1) - 127 - 23)
    bound = np.maximum(8 * spacing, np.finfo(np.float32).tiny)
    bad = np.abs(candidate.astype(float) - reference.astype(float)) > bound
    assert not bad.any(), f"{label}: {int(bad.sum())} cells > 8 ULP (max {np.abs(candidate - reference).max():.3e})"


def _state(dtype=jnp.float32):
    z = jnp.zeros_like(jnp.asarray(_a(C, "qv"), dtype))
    col = lambda n: jnp.asarray(_a(C, n), dtype)  # noqa: E731
    return P.MynnPBLColumnState(col("u"), col("v"), col("w"), col("th"), col("qv"), 0.5 * col("qke"), col("p"),
                                col("rho"), col("dz"), z, z, z, qc=col("qc"), qi=col("qi"))


def test_oracle_census_is_active_on_real_mp28_columns():
    assert len(FX["inputs"]["j"]) >= 16 and np.asarray(C["qnc"]).max() > 1e7 and np.asarray(C["qnwfa"]).max() > 1e8
    for key in ("Dqnc", "Dqnwfa", "Dqnifa"):
        assert np.count_nonzero(W[key]) >= 100, key
    for key in ("s_awqnc", "s_awqnwfa", "s_awqnifa"):
        assert np.count_nonzero(W[key]) >= 8, key                      # real DMP_mf witness (3 columns)
    sqc, sqc2 = _a(W, "sqc"), _a(W, "sqc2")
    assert np.any((sqc > 1e-12) & (_a(C, "qnc") < 100)) or np.any(sqc2 > 1e-12)   # nc_min floor exercised


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["native", "retained"])
def test_number_solves_match_pristine_mynn_tendencies(monkeypatch, dtype):
    monkeypatch.setenv("GPUWRF_MYNN_TRIDIAG_REAL", "1")
    st = _state(dtype)
    t = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    passive = {n: t(_a(C, q)) for n, (q, _, _) in NAMES.items()}
    mf = {"s_aw": t(_a(W, "s_aw"))}
    mf.update({"s_awq" + n: t(_a(W, flux)) for n, (_, _, flux) in NAMES.items()})
    with jax.enable_x64(dtype == jnp.float64):
        out = jax.jit(lambda: mix_passive_numbers(passive, t(_a(W, "sqc")), t(_a(W, "sqc2")), st,
                                                  {"dfh": t(_a(W, "dfh"))}, DT, mf))()
        ni2 = jax.jit(lambda: mix_ice_number(t(_a(C, "qni")), t(_a(W, "sqi")), t(_a(W, "sqi2")), st,
                                             {"dfh": t(_a(W, "dfh"))}, DT,
                                             {"s_aw": mf["s_aw"], "s_awqni": t(_a(W, "s_awqni"))}))()
    ulp_gate(np.asarray(ni2, np.float32), _a(W, "qni2"), "QNI harness control")
    for name, (q, ref, _) in NAMES.items():
        got, want = np.asarray(out[name], np.float32), _a(W, ref)
        if name in AEROSOL_BOUNDS:
            # Cells WRF clamped to x_bound*exp(-zw/ht): exp amplifies the argument's
            # rounding by |zw/ht| (XLA may form zw*(1/ht), E130) -> input-ULP envelope
            # (|zw/ht| + 4) * 2 eps relative; every unclamped cell keeps the 8 ULP gate.
            low, high, ht = AEROSOL_BOUNDS[name]
            zw = np.concatenate([np.zeros_like(_a(C, "dz")[:, :1]),
                                 np.cumsum(_a(C, "dz")[:, :-1], axis=-1, dtype=np.float32)], -1)
            decay = np.exp(-zw / np.float32(ht))
            clamped = (np.isclose(want, low * decay, rtol=1e-5) | np.isclose(want, high * decay, rtol=1e-5))
            env = (zw / ht + 4.0) * 2 * np.finfo(np.float32).eps * np.abs(want)
            assert np.all(np.abs(got - want)[clamped] <= env[clamped]), name
            assert clamped.sum() >= 10, name                       # bounds exercised
            got, want = np.where(clamped, want, got), want
        ulp_gate(got, want, name)
        # WRF rate RQ*BLTEN = (q2 - q)/delt (mynn_tendencies :4881-4889, :4982-4998)
        rate = (np.asarray(out[name], np.float64) - _a(C, q, np.float64)) / DT
        wrf_rate = _a(W, {"Nc": "Dqnc", "nwfa": "Dqnwfa", "nifa": "Dqnifa"}[name], np.float64)
        # WRF forms the rate in REAL from REAL q2 and q: quantized by ULP(q2)/delt
        # (+ REAL32 minimum-normal floor: XLA:CPU flushes subnormal Nc, E117).
        big = np.maximum(np.abs(_a(W, ref)), np.abs(_a(C, q)))
        quantum = np.maximum(8 * np.spacing(big).astype(np.float64), np.finfo(np.float32).tiny) / DT
        assert np.all(np.abs(rate - wrf_rate) <= quantum + 1e-4 * np.abs(wrf_rate)), name


def test_unmixed_numbers_miss_pristine():
    # Deletion discrimination: the pre-fix port left QNC/QNWFA/QNIFA untouched.
    for name, (q, ref, _) in NAMES.items():
        assert np.abs(_a(C, q) - _a(W, ref)).max() > 1e-3 * np.abs(_a(W, ref)).max(), name


def _mf_args(dtype):
    f = np.float32
    qv, qc, qi = _a(C, "qv"), _a(C, "qc"), _a(C, "qi")
    sqv, sqc, sqi = qv / (f(1) + qv), qc / (f(1) + qv), qi / (f(1) + qv)
    th, exner = _a(C, "th"), _a(C, "exner")
    xlvcp = f(2.5e6) / f(7.0 * 287.0 / 2.0)
    thl = th - xlvcp / exner * sqc
    thv = th * (f(1) + f(461.6 / 287.0 - 1.0) * sqv)
    dz = _a(C, "dz")
    zw = np.concatenate([np.zeros_like(dz[:, :1]), np.cumsum(dz, axis=-1, dtype=np.float32)], axis=-1)
    t = lambda x: jnp.asarray(x, dtype)  # noqa: E731
    args = [t(x) for x in (sqv + sqc + sqi, sqv, sqc, _a(C, "u"), _a(C, "v"), _a(C, "w"), th, thl, thv,
                           _a(C, "tk"), _a(C, "qke"), _a(C, "p"), exner, _a(C, "rho"), dz, zw)]
    sc = {k: t(_a(S, k)) for k in ("ust", "flt", "fltv", "flq", "flqv", "pblh", "xland")}
    kw = dict(sc, ts=t(_a(S, "th_sfc")), dx=float(S["dx"][0]), dt=DT, psig_shcu=t(np.ones(len(S["dx"]))),
              cloud_base=t(np.full(len(S["dx"]), 9000.0)), qni=t(_a(C, "qni")),
              passive={n: t(_a(C, q)) for n, (q, _, _) in NAMES.items()})
    return args, kw


@pytest.mark.parametrize("entry", ["native", "retained"])
def test_passive_plume_fluxes_match_pristine_dmp_mf(monkeypatch, entry):
    # native = release dispatch (fused fp32 plume with in-kernel NUP sums); retained = XLA reference.
    native = entry == "native"
    for flag, value in (("GPUWRF_MYNN_PLUME_SUMS", "1"), ("GPUWRF_EDMF_FUSED_PLUME", "1" if native else "0"),
                        ("GPUWRF_MYNN_FP32_PLUME", "1" if native else "0")):
        monkeypatch.setenv(flag, value)
    dtype = jnp.float32 if native else jnp.float64
    args, kw = _mf_args(dtype)

    def fn(*a):
        return E.dmp_mf_columns(*a, **kw)
    with jax.enable_x64(not native):
        out = jax.jit(fn)(*args)
    for name, (_, _, flux) in NAMES.items():
        got, ref = np.asarray(out["s_awq" + name], np.float64), _a(W, flux, np.float64)
        assert np.isfinite(got).all()
        for c in range(len(ref)):
            scale = np.abs(ref[c]).max()
            assert np.abs(got[c] - ref[c]).max() <= (0.05 * scale if scale else 1e-12), (name, c)
    # the QNI lane of the same evaluations stays the QNI flux
    np.testing.assert_allclose(np.asarray(out["s_awqni"], np.float64), _a(W, "s_awqni", np.float64),
                               atol=0.05 * np.abs(_a(W, "s_awqni")).max())


def test_dmp_passive_entry_reuses_the_qni_lane_exactly():
    # passive=... must equal separate QNI-lane evaluations bit for bit (same code path).
    args, kw = _mf_args(jnp.float64)
    passive = kw.pop("passive")
    with jax.enable_x64(True):
        joint = E.dmp_mf_columns(*args, **kw, passive=passive)
        for name, values in passive.items():
            alone = E.dmp_mf_columns(*args, **dict(kw, qni=values))
            np.testing.assert_array_equal(np.asarray(joint["s_awq" + name]), np.asarray(alone["s_awqni"]))
        base = E.dmp_mf_columns(*args, **kw)
        for key in ("s_aw", "s_awqv", "s_awqc", "s_awqni"):
            np.testing.assert_array_equal(np.asarray(joint[key]), np.asarray(base[key]))


@pytest.mark.parametrize("mp,expect", [(8, ()), (0, ()), (6, ()), (17, ()), (22, ()), (9, ("Nc",)),
                                        (16, ("Nc",)), (28, ("Nc", "nwfa", "nifa")), (38, ("Nc", "nwfa", "nifa")),
                                        (40, ("Nc",)), (51, ("Nc",)), (56, ("Nc",))])
def test_number_scalar_set_follows_wrf_registry(mp, expect):
    from gpuwrf.runtime.operational_mode import _pbl_mynn_number_scalars
    assert _pbl_mynn_number_scalars(mp) == expect


@pytest.mark.parametrize("entry", ["native", "retained"])
def test_whole_entry_persists_passive_numbers(monkeypatch, entry):
    from gpuwrf.kernels import phys_mynn_columns as N
    s = _state().replace(passive={"Nc": jnp.asarray(_a(C, "qnc")), "nwfa": jnp.asarray(_a(C, "qnwfa"))})
    z = jnp.zeros_like(s.qv)
    monkeypatch.setattr(P, "_MYNN_SGS_CLOUD", False)
    monkeypatch.setattr(P, "_mym_turbulence", lambda *a, **kw: {"el": s.el, "pblh": z[..., 0]})
    monkeypatch.setattr(P, "_mym_predict_qke", lambda *a, **kw: (2 * s.tke, z, z, z))
    monkeypatch.setattr(P, "_retrieve_exchange_coeffs", lambda *a, **kw: (z, z))
    changed = {k: v + 7.0 for k, v in s.passive.items()}
    monkeypatch.setattr(P, "_apply_mean_tendencies_with_clouds",
                        lambda *a, **kw: (s.u, s.v, s.theta, s.qv, s.qc, s.qi, changed))
    fn = N._advance_native if entry == "native" else P._step_mynn_pbl_impl_with_pblh
    out, _ = fn(s, DT, False, edmf=False)
    for k in changed:
        np.testing.assert_array_equal(np.asarray(out.passive[k]), np.asarray(changed[k]))


def test_cloudmix_off_keeps_numbers_unmixed(monkeypatch):
    # bl_mynn_cloudmix=0: WRF qnc2=qnc (no RQNCBLTEN); the legacy branch returns them unchanged.
    monkeypatch.setenv("GPUWRF_MYNN_CLOUDMIX", "0")
    s = _state().replace(passive={"Nc": jnp.asarray(_a(C, "qnc"))})
    monkeypatch.setattr(P, "_apply_mean_tendencies_legacy", lambda st, *a, **kw: (st.u, st.v, st.theta, st.qv))
    out = P._apply_mean_tendencies_with_clouds(s, {}, DT, None, None, None)
    assert out[-1] is s.passive


def test_operational_mynn_couples_number_rates_once(monkeypatch):
    """mp=28 + MYNN source-leaf RK branch: the adapter is asked for Nc/nwfa/nifa, their
    rates become coupled scalar sources (pbl_driver.F:1879-1911, module_em.F:2473-2479)
    and the State numbers stay at the PBL-entry value (no second Euler update)."""
    from gpuwrf.coupling.physics_couplers import MynnPBLSourceLeaves
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    ones = jnp.ones_like(state.qv)
    state = state.replace(_cast=False, Nc=1e8 * ones, nwfa=5e8 * ones, nifa=1e6 * ones)
    ns = _namelist(grid, dt_s=18., mp_physics=28, cu_physics=0, bl_pbl_physics=5, sf_sfclay_physics=5,
                   sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=1)
    seen = {}

    def adapter(st, dt, grid, **kw):
        seen.update(kw)
        bump = {"Nc": 3.0e5, "nwfa": -2.0e5, "nifa": 40.0}
        out = st.replace(_cast=False, **{k: getattr(st, k) + v for k, v in bump.items()})
        z = jnp.zeros_like(st.theta)
        return MynnPBLSourceLeaves(state=out, rthblten=z, rqvblten=z, rublten=z, rvblten=z)

    monkeypatch.setattr(om, "mynn_adapter_with_source_leaves", adapter)
    monkeypatch.setattr(om, "_phys_tend_rk_wrf_enabled", lambda: True)
    monkeypatch.setattr(om, "_microphysics_wrf_order_enabled", lambda: True)
    forcing = om._physics_step_forcing(initial_operational_carry(state), ns, 0., run_radiation=False,
                                       first_timestep=True)
    assert seen.get("passive_scalars") == ("Nc", "nwfa", "nifa")
    mass = (np.asarray(ns.metrics.c1h)[:, None, None] * np.asarray(state.mu_total)[None]
            + np.asarray(ns.metrics.c2h)[:, None, None])
    sources = forcing.dry_tendencies.moist_tendf or {}
    assert {"Nc", "nwfa", "nifa"} <= set(sources), sorted(sources)   # scalar_tend(P_QNC/P_QNWFA/P_QNIFA)
    for name, bump in (("Nc", 3.0e5), ("nwfa", -2.0e5), ("nifa", 40.0)):
        np.testing.assert_allclose(np.asarray(sources[name]), mass * bump / 18.0,
                                   rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(getattr(forcing.state, name)), np.asarray(getattr(state, name)))


@pytest.mark.parametrize("mp", [28, 40])
def test_swiss_production_step_carries_number_sources_to_every_rk_stage(monkeypatch, tmp_path, mp):
    """E85 caller chain on the real Swiss root step (eval_shape, CPU) with mp switched and the WRF
    physics->RK coupling ON: MYNN's own number change becomes RQ*BLTEN (object identity), is coupled
    once, ring-masked and handed to the scalar update of EVERY RK stage (PD species at the last stage)."""
    import sys
    sys.path.insert(0, str(ROOT / "tests/v040/rootbdy"))
    from test_root_flow_extras import _swiss_root
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_mode import _pbl_mynn_number_scalars
    for key, value in (("GPUWRF_PHYS_TEND_RK_WRF", "1"), ("GPUWRF_ROOT_SCALAR_BDY_RK1", "1"),
                       ("GPUWRF_MYNN_CLOUDMIX", "1")):
        monkeypatch.setenv(key, value)
    nml, carry = _swiss_root(mp, tmp_path, monkeypatch)
    names = _pbl_mynn_number_scalars(mp)
    assert int(nml.mp_physics) == mp and int(nml.bl_pbl_physics) == 5 and names
    rec = {"mynn": [], "rate": [], "couple": [], "mask": [], "with": [], "pd": []}

    def spy(name, sink, capture):
        real = getattr(op, name)

        def wrapped(*a, **k):
            out = real(*a, **k)
            rec[sink].append(capture(a, k, out))
            return out
        monkeypatch.setattr(op, name, wrapped)

    spy("mynn_adapter_with_source_leaves", "mynn", lambda a, k, out: (a[0], out, k.get("passive_scalars")))
    spy("_pbl_scalar_rate", "rate", lambda a, k, out: (a[0], a[1], out))
    spy("_coupled_physics_moist_tendf", "couple", lambda a, k, out: (a[1], out))
    spy("_physics_sc_outside_spec", "mask", lambda a, k, out: (a[0], out))
    spy("_with_physics_sc", "with", lambda a, k, out: (tuple(a[0]), a[2], tuple(a[3]), tuple(a[1]), tuple(out)))
    spy("_rk_update_scalar_pd", "pd", lambda a, k, out: (tuple(a[1]), a[2][0]))
    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1, cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        jax.clear_caches()
    assert jax.devices()[0].platform == "cpu"
    # producer: MYNN asked for the Registry set; each rate is formed from MYNN's own post/entry leaf
    assert len(rec["mynn"]) == 1 and rec["mynn"][0][2] == names
    entry, mynn_out, _ = rec["mynn"][0]
    assert len(rec["couple"]) == 1
    raw, coupled = rec["couple"][0]
    for name in names:
        post, pre = getattr(mynn_out.state, name), getattr(entry, name)
        hit = [r for p, e, r in rec["rate"] if p is post and e is pre]
        assert len(hit) == 1 and raw[name] is hit[0] and post is not pre, name
        assert name in coupled
    # consumer: every RK stage masks the same coupled dict and feeds it to that stage's scalar update
    assert len(rec["mask"]) == int(nml.rk_order) == len(rec["with"])
    assert all(m[0] is coupled for m in rec["mask"])
    for (species, physics_sc, skip, before, after), (_, masked) in zip(rec["with"], rec["mask"]):
        assert physics_sc is masked and set(names) <= set(species) and set(names) <= set(physics_sc)
        for name in set(names) - set(skip):   # the source was really added to that stage's tendency
            assert after[species.index(name)] is not before[species.index(name)], name
    # PD-family numbers take their source in the final-stage pre-update instead (solve_em.F:1942-1948)
    final_skip = set(rec["with"][-1][2])
    assert len(rec["pd"]) == 1 and rec["pd"][0][1] is rec["mask"][-1][1]
    assert {n for n in names if n in final_skip} <= set(rec["pd"][0][0])
