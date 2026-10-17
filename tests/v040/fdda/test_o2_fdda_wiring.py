"""grid_fdda=1 wiring into the operational step (lane o2-fdda).

Deletion-sensitive checks of the seams between the oracle-verified kernel
(test_o2_fdda_oracle_parity.py) and the RK step:
* phy_prep inputs (th_phy dry, z_at_w = (phb+ph)/g) in WRF REAL form;
* the coupled FDDA sources enter t_tendf / moist_tend(qv) BEFORE
  conv_t_tendf_to_moist and ru/rv_tendf after the PBL fold;
* the driver uses the post-PBL PBLH and mut = mub + mu_2;
* carry seeding, restart pickling of the FDDA cursor.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import fdda_grid as fg
from gpuwrf.coupling import fdda_rk

jax.config.update("jax_enable_x64", True)

NZ, NY, NX = 6, 5, 7


def _rng(seed=0):
    return np.random.default_rng(seed)


def _fields(dtype=np.float32, seed=0):
    r = _rng(seed)
    return dict(
        u=(r.normal(5, 3, (NZ, NY, NX + 1))).astype(dtype), v=(r.normal(-2, 3, (NZ, NY + 1, NX))).astype(dtype),
        theta=(290 + np.arange(NZ)[:, None, None] * 3 + r.normal(0, 1, (NZ, NY, NX))).astype(dtype),
        qv=np.abs(r.normal(4e-3, 1e-3, (NZ, NY, NX))).astype(dtype),
        mu_perturbation=r.normal(500, 50, (NY, NX)).astype(dtype),
        ph_perturbation=r.normal(0, 30, (NZ + 1, NY, NX)).astype(dtype),
        p_perturbation=r.normal(500, 50, (NZ, NY, NX)).astype(dtype),
        pblh=np.abs(r.normal(600, 300, (NY, NX))).astype(dtype),
    )


def test_phy_prep_inputs_are_wrf_real_literal():
    f = _fields()
    phb = (np.arange(NZ + 1)[:, None, None] * np.float32(4000.0) + np.float32(9.81 * 300)).astype(np.float32)
    phb = np.broadcast_to(phb, (NZ + 1, NY, NX)).astype(np.float32)
    state = SimpleNamespace(**{k: jnp.asarray(v) for k, v in f.items()})
    th, zw = fdda_rk._phy_prep_inputs(state, SimpleNamespace(phb=jnp.asarray(phb)), jnp.float32, True)
    rvrd = np.float32(np.float32(461.6) / np.float32(287.0))
    th_ref = f["theta"] / (np.float32(1.0) + rvrd * f["qv"])
    zw_ref = (phb + f["ph_perturbation"]) / np.float32(9.81)
    assert np.array_equal(np.asarray(th), th_ref)
    assert np.array_equal(np.asarray(zw), zw_ref)  # true REAL division, not x*(1/g) (E123)


def _inputs(dtype, **opts):
    f = _fields(np.float32, seed=1)
    pair = {}
    for n, key in (("u", "u"), ("v", "v"), ("q", "qv")):
        pair[f"{n}_old"] = jnp.asarray(f[key][None] * np.float32(1.01))
        pair[f"{n}_new"] = jnp.asarray(f[key][None] * np.float32(0.98))
    pair["t_old"] = jnp.asarray(f["theta"][None] - np.float32(301.0))
    pair["t_new"] = jnp.asarray(f["theta"][None] - np.float32(299.5))
    options = fg.GridFddaOptions(guv=3e-4, gt=3e-4, gq=3e-4, interval_m=60, end_h=6, **opts)
    return fg.GridFddaInputs(**pair, options=options, record_steps=(0,))


def _namelist(inputs, ht):
    metrics = SimpleNamespace(c1h=jnp.linspace(1.0, 0.1, NZ), c2h=jnp.linspace(0.0, 5000.0, NZ))
    return SimpleNamespace(data_assimilation=inputs, metrics=metrics, dt_s=18.0,
                           grid=SimpleNamespace(terrain_height=jnp.asarray(ht)))


def test_step_sources_use_post_pbl_pblh_and_entry_masses(monkeypatch):
    """fdda_step_sources == driver(before, pblh) + couple(mub+mu_2) + phy_fg_ten ownership, step 2."""

    monkeypatch.setattr(fdda_rk, "_real_glue", lambda: False)
    inputs = _inputs(jnp.float64, if_no_pbl_nudging_t=1, if_no_pbl_nudging_uv=1)
    f = _fields(np.float64, seed=2)
    ht = np.full((NY, NX), 100.0)
    phb = np.broadcast_to((np.arange(NZ + 1) * 4000.0 + 981.0)[:, None, None], (NZ + 1, NY, NX)).copy()
    mub = np.full((NY, NX), 9.0e4)
    before = SimpleNamespace(**{k: jnp.asarray(v) for k, v in f.items()},
                             mu_total=jnp.asarray(mub + f["mu_perturbation"]),
                             ph_total=jnp.asarray(phb + f["ph_perturbation"]))
    base = SimpleNamespace(mub=jnp.asarray(mub), phb=jnp.asarray(phb), pb=jnp.full((NZ, NY, NX), 9.e4))
    namelist = _namelist(inputs, ht)
    carry0 = fg.initial_fdda_carry(inputs, jnp.float64)
    pblh_post = jnp.asarray(f["pblh"] * 0.5)  # differs from before.pblh
    got_carry, got = fdda_rk.fdda_step_sources(carry0, before, pblh_post, base, namelist, 18.0,
                                               specified_or_nested=True, periodic_x=False)
    th, zw = fdda_rk._phy_prep_inputs(before, base, jnp.float64, False)
    ref = fg.fdda_driver_tendencies(carry0, inputs, before.u, before.v, th, before.qv, pblh_post,
                                    jnp.asarray(ht), zw, 18.0, 2, jnp.float64)
    ref = fg.couple_fdda_rates(ref, jnp.asarray(mub) + before.mu_perturbation, before.mu_perturbation,
                               jnp.asarray(mub), namelist.metrics.c1h, namelist.metrics.c2h, jnp.float64)
    for a, b in zip(got, fg.owned_fdda_tendf(ref, specified_or_nested=True)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert float(jnp.abs(got[2]).max()) > 0 and float(jnp.abs(got[0]).max()) > 0
    # the PBLH argument is used (not before.pblh): the PBL-top weights change with it
    _c, other = fdda_rk.fdda_step_sources(carry0, before, before.pblh, base, namelist, 18.0,
                                          specified_or_nested=True, periodic_x=False)
    assert not np.array_equal(np.asarray(other[2]), np.asarray(got[2]))
    # specified ownership: outer mass ring and boundary faces carry no FDDA source
    assert float(jnp.abs(got[2][:, 0]).max()) == 0 and float(jnp.abs(got[0][..., 0]).max()) == 0
    assert float(jnp.abs(got[0][..., 1]).max()) > 0


def test_fdda_sources_enter_tendencies_before_moist_theta_conversion():
    from gpuwrf.runtime import operational_mode as om

    r = _rng(3)
    shape = (NZ, NY, NX)
    mu = jnp.asarray(r.normal(9e4, 100, (NY, NX)))
    metrics = SimpleNamespace(c1h=jnp.linspace(1.0, 0.1, NZ), c2h=jnp.linspace(0.0, 5000.0, NZ))
    rthraten = jnp.asarray(r.normal(0, 1e-5, shape))
    rthblten = jnp.asarray(r.normal(0, 1e-4, shape))
    rqvblten = jnp.asarray(r.normal(0, 1e-7, shape))
    rublten = jnp.asarray(r.normal(0, 1e-3, shape))
    rvblten = jnp.asarray(r.normal(0, 1e-3, shape))
    qv = jnp.asarray(np.abs(r.normal(4e-3, 1e-3, shape)))
    theta = jnp.asarray(r.normal(300, 5, shape))
    fdda = (jnp.asarray(r.normal(0, 1.0, (NZ, NY, NX + 1))), jnp.asarray(r.normal(0, 1.0, (NZ, NY + 1, NX))),
            jnp.asarray(r.normal(0, 0.1, shape)), jnp.asarray(r.normal(0, 1e-5, shape)))
    args = (mu, rthraten, rthblten, rqvblten, rublten, rvblten, qv, theta, metrics, jnp.float64)
    base = om._source_leaf_dry_tendencies(*args)
    with_fdda = om._source_leaf_dry_tendencies(*args, fdda_tendf=fdda)
    mass = metrics.c1h[:, None, None] * mu[None] + metrics.c2h[:, None, None]
    t_src = mass * (rthraten + rthblten) + fdda[2]
    q_src = mass * rqvblten + fdda[3]
    factor = 1.0 + om._RVRD * qv
    expect_t = factor * t_src + om._RVRD * theta / factor * q_src
    assert np.allclose(np.asarray(with_fdda.t_tendf), np.asarray(expect_t), rtol=1e-13, atol=0)
    assert not np.allclose(np.asarray(base.t_tendf), np.asarray(expect_t), rtol=1e-6, atol=0)
    assert np.array_equal(np.asarray(with_fdda.ru_tendf), np.asarray(base.ru_tendf + fdda[0]))
    assert np.array_equal(np.asarray(with_fdda.rv_tendf), np.asarray(base.rv_tendf + fdda[1]))


def test_finish_decouples_with_final_mass_and_carry_pickles():
    inputs = _inputs(jnp.float32)
    carry = fg.initial_fdda_carry(inputs, jnp.float32)
    carry = carry.replace(rthndgdten=jnp.full_like(carry.rthndgdten, 2.0))
    namelist = _namelist(inputs, np.zeros((NY, NX)))
    namelist.metrics = SimpleNamespace(c1h=jnp.ones(NZ), c2h=jnp.zeros(NZ), precision="fp64")
    out = fdda_rk.fdda_finish(carry, jnp.full((NY, NX), 4.0), namelist)
    assert np.allclose(np.asarray(out.rthndgdten), 0.5)
    host = jax.tree_util.tree_map(np.asarray, out)  # restart v2 _hostify_tree
    back = pickle.loads(pickle.dumps(host, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(back, fg.GridFddaCarry)
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(out)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_restart_schema_includes_fdda_cursor():
    from gpuwrf.io.restart import _V1_UNSUPPORTED_CARRY_FIELDS
    from gpuwrf.runtime.operational_state import OperationalCarry

    assert "fdda" in _V1_UNSUPPORTED_CARRY_FIELDS  # serialized as an exact host pytree in v2 payloads
    assert OperationalCarry.__dataclass_fields__["fdda"].default is None


def test_legacy_da_hook_does_not_double_apply_grid_fdda():
    from gpuwrf.runtime import operational_mode as om

    inputs = _inputs(jnp.float32)
    namelist = SimpleNamespace(data_assimilation=inputs, metrics=None, dt_s=18.0)
    sentinel = object()
    state, dry, enabled = om._apply_data_assimilation_forcing(sentinel, sentinel, sentinel, namelist, 0.0)
    assert state is sentinel and dry is sentinel and enabled


def _f64_eqns(jaxpr, out):
    """(primitive, shape) of every equation with a non-weak float64 output, nested jaxprs included."""

    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            aval = getattr(var, "aval", None)
            if aval is not None and getattr(aval, "dtype", None) == jnp.float64 and not getattr(aval, "weak_type", False):
                out.append((eqn.primitive.name, tuple(aval.shape)))
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                if hasattr(sub, "jaxpr"):
                    _f64_eqns(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr, out)
                elif hasattr(sub, "eqns"):
                    _f64_eqns(sub, out)
    return out


def test_real_path_f64_census_is_only_the_wrf_double_clock(monkeypatch):
    """S2/E169 (rv-fdda F2): under the REAL glue the whole FDDA seam (driver + fddagd + coupling +
    ownership + decoupling) is REAL except WRF's DOUBLE island curr_mins2 = REAL(curr_secs2_r8/60d0)
    and the scalar clock bookkeeping (integer-clock fraction, step index 1+rint(lead/dt))."""

    from collections import Counter

    monkeypatch.setattr(fdda_rk, "_real_glue", lambda: True)
    metrics = SimpleNamespace(c1h=jnp.linspace(1.0, 0.1, NZ, dtype=jnp.float32),
                              c2h=jnp.linspace(0.0, 5000.0, NZ, dtype=jnp.float32))
    monkeypatch.setattr(fdda_rk, "_metrics", lambda namelist, real_glue: metrics)
    inputs = _inputs(jnp.float32, if_no_pbl_nudging_t=1, if_no_pbl_nudging_uv=1, if_ramping=1, dtramp_min=60.0,
                     fgdtzero=1, stepfg=3)
    namelist = SimpleNamespace(data_assimilation=inputs, metrics=metrics, dt_s=18.0,
                               grid=SimpleNamespace(terrain_height=jnp.zeros((NY, NX), jnp.float32)))
    phb = jnp.asarray(np.broadcast_to((np.arange(NZ + 1) * 4000.0 + 981.0)[:, None, None],
                                      (NZ + 1, NY, NX)).astype(np.float32))
    base = SimpleNamespace(mub=jnp.full((NY, NX), 9.0e4, jnp.float32), phb=phb,
                           pb=jnp.full((NZ, NY, NX), 9.0e4, jnp.float32))
    fields = {k: jnp.asarray(v) for k, v in _fields(np.float32, seed=4).items()}

    def seam(fields, carry, lead):
        before = SimpleNamespace(**fields, mu_total=base.mub + fields["mu_perturbation"],
                                 ph_total=phb + fields["ph_perturbation"])
        carry, owned = fdda_rk.fdda_step_sources(carry, before, fields["pblh"], base, namelist, lead,
                                                 specified_or_nested=True, periodic_x=False)
        return fdda_rk.fdda_finish(carry, base.mub + fields["mu_perturbation"], namelist), owned

    carry = fg.initial_fdda_carry(inputs, jnp.float32, with_history=True)
    jaxpr = jax.make_jaxpr(seam)(fields, carry, jnp.float64(36.0))
    census = _f64_eqns(jaxpr.jaxpr, [])
    assert all(shape == () for _p, shape in census), census  # no f64 field arithmetic at all
    assert Counter(p for p, _s in census) == Counter({
        "div": 2,  # curr_secs2_r8/60d0 (WRF DOUBLE) + step index lead/dt
        "optimization_barrier": 1,  # fenced 60d0 divisor
        "convert_element_type": 1, "sub": 1,  # integer-clock fractional seconds
        "round": 1, "jit": 1,  # step index rint (jnp.rint wrapper)
    }), census
    out_carry, owned = jax.eval_shape(seam, fields, carry, jnp.float64(36.0))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves((out_carry, owned)))
