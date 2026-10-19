"""w42-wdm6perf: WDM6 sedimentation loop restructuring is bitwise neutral and stays restructured.

The rain+Nr fallout used a FIXED 64-trip substep loop (each trip a km-level sequential sweep) although
WRF's mstep is 1..3 on real 3 km cases, and the PLM remap searched kb/kt with data-dependent while
loops nested in its km-level loop (one host round trip per trip on GPU). The port now stops the
substep loop at the column's own mstep (capped at 64 as before) and finds kb/kt with a vectorized
first-hit search. These tests pin (1) bitwise equality with verbatim copies of the v0.4.1
implementations on the frozen WDM6 oracle columns, at the oracle dt, a longer dt (two minor
loops) and squeezed layers (mstep above the 64 cap), in f64 and f32, for WDM6 and WDM7 (which
imports both helpers), and (2) the loop structure itself (fails if the 64-trip loop or the nested
while loops come back).
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics import microphysics_wdm6 as M  # noqa: E402
from gpuwrf.physics import microphysics_wdm5 as M5  # noqa: E402
from gpuwrf.physics import microphysics_wdm7 as M7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.abspath(os.path.join(HERE, "..", "proofs", "v060_wdm6", "savepoints"))
SAVE5 = os.path.abspath(os.path.join(HERE, "..", "proofs", "v013", "savepoints_wdm5"))
SLMSK = {1: 2.0, 2: 1.0, 3: 1.0, 4: 2.0, 5: 1.0, 6: 2.0}
KEYS = ("T_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN", "QG_IN", "NN_IN", "NC_IN", "NR_IN", "DEN", "P", "DELZ")
KEYS5 = ("T_IN", "QV_IN", "QC_IN", "QR_IN", "QI_IN", "QS_IN", "NN_IN", "NC_IN", "NR_IN", "DEN", "P", "DELZ")


# ---- verbatim v0.4.1 references (src/gpuwrf/physics/microphysics_wdm6.py at tag v0.4.1) -------------
def _ref_plm_remap(qa, dza, qmi, qpi, zi, za, km):
    def body(k, carry):
        qn, kb, kt = carry
        kb = jnp.maximum(kb - 1, 0)
        kt = jnp.maximum(kt - 1, 0)
        done = zi[k] >= za[km]

        def find_kb(start):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k] <= za[kk + 1]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, _ = jax.lax.while_loop(cond, step, (start, False))
            return kk0

        kb_new = find_kb(kb)
        kt_entry = kt

        def find_kt(start):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k + 1] <= za[kk]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, found = jax.lax.while_loop(cond, step, (start, False))
            return jnp.where(found, kk0, kt_entry) - 1

        kt_new = find_kt(kt)
        kb_c = kb_new
        kt_c = kt_new

        # kt == kb : piecewise method
        tl = (zi[k] - za[kb_c]) / dza[kb_c]
        th_ = (zi[k + 1] - za[kb_c]) / dza[kb_c]
        tl2 = tl * tl
        th2 = th_ * th_
        qqd = 0.5 * (qpi[kb_c] - qmi[kb_c])
        qqh = qqd * th2 + qmi[kb_c] * th_
        qql = qqd * tl2 + qmi[kb_c] * tl
        qn_eq = (qqh - qql) / (th_ - tl)

        # kt > kb : integrate kb partial + full middle cells + kt partial
        tl_b = (zi[k] - za[kb_c]) / dza[kb_c]
        tl2_b = tl_b * tl_b
        qqd_b = 0.5 * (qpi[kb_c] - qmi[kb_c])
        qql_b = qqd_b * tl2_b + qmi[kb_c] * tl_b
        dql = qa[kb_c] - qql_b
        zsum0 = (1.0 - tl_b) * dza[kb_c]
        qsum0 = dql * dza[kb_c]

        midx = jnp.arange(km)
        mid_mask = (midx >= kb_c + 1) & (midx <= kt_c - 1)
        zsum_mid = jnp.sum(jnp.where(mid_mask, dza[0:km], 0.0))
        qsum_mid = jnp.sum(jnp.where(mid_mask, qa[0:km] * dza[0:km], 0.0))

        th_t = (zi[k + 1] - za[kt_c]) / dza[kt_c]
        th2_t = th_t * th_t
        qqd_t = 0.5 * (qpi[kt_c] - qmi[kt_c])
        dqh = qqd_t * th2_t + qmi[kt_c] * th_t
        zsum = zsum0 + zsum_mid + th_t * dza[kt_c]
        qsum = qsum0 + qsum_mid + dqh * dza[kt_c]
        qn_gt = qsum / zsum

        qn_val = jnp.where(kt_c == kb_c, qn_eq, jnp.where(kt_c > kb_c, qn_gt, 0.0))
        qn_val = jnp.where(done, 0.0, qn_val)
        qn = qn.at[k].set(qn_val)
        return qn, kb_c, kt_c

    qn0 = jnp.zeros(km, dtype=qa.dtype)
    qn, _, _ = jax.lax.fori_loop(0, km, body, (qn0, 0, 0))
    return qn


def _ref_rain_number_fallout(qr, nr, qs, qg, den, denfac, t, delz, dtcld,
                         work1_r0, workn_r0, mstep, slope_rain_vt=M._slope_rain_vt):
    km = qr.shape[0]
    # mstep is a per-column scalar (same value used for all k, = column max).
    # WRF uses per-column mstep(i); for a single column this is one integer.
    mstep_i = jnp.maximum(mstep, 1.0)

    def substep(n, carry):
        qr, nr, fall1, falln, work1_r, workn_r = carry
        do_n = (n < mstep_i)  # scalar mask (n is 0-based here vs WRF 1-based)

        # falk/falkn store the per-cell fluxes computed AT EACH CELL'S PROCESSING
        # TIME (WRF semantics): the dqr(k+1) flux-in uses the STORED falk(k+1)
        # and the already-updated qrs(k+1), NOT a recompute from qrs(k+1).
        falk = jnp.zeros(km, qr.dtype)
        falkn = jnp.zeros(km, qr.dtype)

        # top cell (k = kte = km-1)
        falk_top = den[km - 1] * qr[km - 1] * work1_r[km - 1] / mstep_i
        falkn_top = nr[km - 1] * workn_r[km - 1] / mstep_i
        falk = falk.at[km - 1].set(falk_top)
        falkn = falkn.at[km - 1].set(falkn_top)
        fall1 = fall1.at[km - 1].add(jnp.where(do_n, falk_top, 0.0))
        falln = falln.at[km - 1].add(jnp.where(do_n, falkn_top, 0.0))
        qr = qr.at[km - 1].set(jnp.where(do_n,
            jnp.maximum(qr[km - 1] - falk_top * dtcld / den[km - 1], 0.0), qr[km - 1]))
        nr = nr.at[km - 1].set(jnp.where(do_n,
            jnp.maximum(nr[km - 1] - falkn_top * dtcld, 0.0), nr[km - 1]))

        # interior cells, top-1 down to bottom (k = km-2 .. 0)
        def kbody(j, carry2):
            qr, nr, fall1, falln, falk, falkn = carry2
            k = km - 2 - j  # from km-2 down to 0
            falk_k = den[k] * qr[k] * work1_r[k] / mstep_i
            falkn_k = nr[k] * workn_r[k] / mstep_i
            falk = falk.at[k].set(falk_k)
            falkn = falkn.at[k].set(falkn_k)
            fall1 = fall1.at[k].add(jnp.where(do_n, falk_k, 0.0))
            falln = falln.at[k].add(jnp.where(do_n, falkn_k, 0.0))
            # dqr from this cell + flux in from k+1: WRF uses STORED falk(k+1)
            # and the already-updated qr[k+1]/nr[k+1].
            dqr_k = jnp.minimum(falk_k * dtcld / den[k], qr[k])
            dqr_kp1 = jnp.minimum(falk[k + 1] * delz[k + 1] / delz[k] * dtcld / den[k],
                                  qr[k + 1])
            dnr_k = jnp.minimum(falkn_k * dtcld, nr[k])
            dnr_kp1 = jnp.minimum(falkn[k + 1] * delz[k + 1] / delz[k] * dtcld, nr[k + 1])
            qr_new = jnp.maximum(qr[k] - dqr_k + dqr_kp1, 0.0)
            nr_new = jnp.maximum(nr[k] - dnr_k + dnr_kp1, 0.0)
            qr = qr.at[k].set(jnp.where(do_n, qr_new, qr[k]))
            nr = nr.at[k].set(jnp.where(do_n, nr_new, nr[k]))
            return qr, nr, fall1, falln, falk, falkn

        qr, nr, fall1, falln, falk, falkn = jax.lax.fori_loop(
            0, km - 1, kbody, (qr, nr, fall1, falln, falk, falkn))

        # recompute rain slope/vt from updated qr,nr (slope_rain), /delz
        vt_m, vt_n = slope_rain_vt(qr, nr, den, denfac)
        work1_r = jnp.where(do_n, vt_m / delz, work1_r)
        workn_r = jnp.where(do_n, vt_n / delz, workn_r)
        return qr, nr, fall1, falln, work1_r, workn_r

    fall1 = jnp.zeros(km, qr.dtype)
    falln = jnp.zeros(km, qr.dtype)
    # bound substeps by a static cap (savepoint columns: dt=90, vt small -> few
    # substeps). 64 is a safe ceiling; inactive substeps are masked.
    carry = (qr, nr, fall1, falln, work1_r0, workn_r0)
    qr, nr, fall1, falln, _w1, _wn = jax.lax.fori_loop(0, 64, substep, carry)
    return qr, nr, fall1, falln



# ---- harness ----------------------------------------------------------------------------------------
def _cases(dtype, scheme):
    """Batches of the 6 frozen oracle columns: (oracle dt + squeezed layers) and a long dt (several minor loops)."""
    root, prefix, keys = (SAVE5, "wdm5", KEYS5) if scheme == "wdm5" else (SAVE, "wdm6", KEYS)
    cols, dts, sl = [], set(), []
    for cid in range(1, 7):
        with open(os.path.join(root, f"{prefix}_case_{cid}.json")) as fh:
            d = json.load(fh)
        cols.append([np.asarray(d["columns"][k], dtype=dtype) for k in keys])
        dts.add(float(d["scalars"]["DT"]))
        sl.append(SLMSK[cid])
    (dt,) = dts
    batch = [np.stack([c[i] for c in cols]) for i in range(len(keys))]
    squeezed = list(batch)
    squeezed[-1] = batch[-1] * dtype(0.004)                               # delz squeezed: mstep far above the 64 cap
    both = [np.concatenate([a, b]) for a, b in zip(batch, squeezed)]
    sl = np.asarray(sl, dtype=dtype)
    cases = [("oracle+squeezed", both, dt, np.concatenate([sl, sl]))]
    if scheme == "wdm6":
        cases.append(("dt240", batch, 240.0, sl))                           # 2 minor loops (compile-heavy: WDM6 only)
    return cases


def _run(module, run_name, use_ref, cols, dt, sl, monkeypatch, calls):
    jax.clear_caches()  # E161: never reuse a trace across patched/unpatched arms
    if use_ref:
        def ref_remap(*a):
            calls["remap"] += 1
            return _ref_plm_remap(*a)

        def ref_fallout(*a):
            calls["fallout"] += 1
            return _ref_rain_number_fallout(*a)

        def ref_fallout5(qr, nr, den, denfac, delz, dtcld, w1, wn, mstep):
            calls["fallout"] += 1
            return _ref_rain_number_fallout(qr, nr, None, None, den, denfac, None, delz, dtcld, w1, wn, mstep,
                                            slope_rain_vt=M5._slope_rain_vt)

        monkeypatch.setattr(M, "_plm_remap", ref_remap)
        monkeypatch.setattr(M, "_rain_number_fallout", ref_fallout)
        if module is M5:
            monkeypatch.setattr(M5, "_plm_remap", ref_remap)
            monkeypatch.setattr(M5, "_rain_number_fallout", ref_fallout5)
        if module is M7:
            monkeypatch.setattr(M7, "_plm_remap", ref_remap)
            monkeypatch.setattr(M7, "_rain_number_fallout", ref_fallout)
    try:
        if module is M7:
            cols = cols[:7] + [cols[6] * 0.3] + cols[7:]                  # qh: real-structured share of qg
        if module is M5:
            fn = jax.jit(lambda *a: module.wdm5_run(*a, dt))
        else:
            fn = jax.jit(lambda *a: getattr(module, run_name)(*a, dt, jnp.asarray(sl)))
        res = fn(*[jnp.asarray(c) for c in cols])
        return {k: np.asarray(v) for k, v in res.items()}
    finally:
        monkeypatch.undo()


def _subjaxprs(value):
    if hasattr(value, "eqns"):
        yield value
    elif hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _subjaxprs(item)


def _count(jaxpr, acc):
    for eqn in jaxpr.eqns:
        acc.setdefault(eqn.primitive.name, []).append(eqn.params.get("length"))
        for value in eqn.params.values():
            for sub in _subjaxprs(value):
                _count(sub, acc)
    return acc


@pytest.mark.parametrize("scheme", ["wdm6", "wdm5"])
def test_loop_structure_stays_bounded(scheme):
    root, prefix, keys = (SAVE5, "wdm5", KEYS5) if scheme == "wdm5" else (SAVE, "wdm6", KEYS)
    with open(os.path.join(root, f"{prefix}_case_2.json")) as fh:
        d = json.load(fh)
    cols = [jnp.asarray(np.asarray(d["columns"][k])[None, :]) for k in keys]
    if scheme == "wdm5":
        jaxpr = jax.make_jaxpr(lambda *a: M5.wdm5_run(*a, 90.0))(*cols)
    else:
        jaxpr = jax.make_jaxpr(lambda *a: M.wdm6_run(*a, 90.0, jnp.asarray([1.0])))(*cols)
    acc = _count(jaxpr.jaxpr, {})
    # one data-dependent loop: the rain substeps (until mstep); no nested search loops in the remap
    assert len(acc.get("while", [])) == 1, acc.get("while")
    assert 64 not in acc.get("scan", []), acc.get("scan")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_remap_search_matches_sequential_search_on_nonmonotone_departures(dtype):
    """WRF's DO WHILE searches start at max(kb-1, 0) / max(kt-1, 0). With nondecreasing interface heights zi
    (dz > 0, every physical column) each level's hit set is a subset of the previous one, so the start can never
    bind and dropping it is an equivalent mutant there. Pin the exact search semantics where it does bind:
    non-monotone zi and za."""
    rng = np.random.default_rng(42)
    km, ncol = 12, 64
    zi = np.cumsum(rng.uniform(-1.5, 2.0, (ncol, km + 1)), axis=1).astype(dtype)  # steps of both signs
    za = (zi + rng.normal(0.0, 3.0, (ncol, km + 1))).astype(dtype)                # departures with dips
    qa = np.concatenate([rng.uniform(0.0, 1.0, (ncol, km)), np.zeros((ncol, 1))], axis=1).astype(dtype)
    dza = np.concatenate([za[:, 1:] - za[:, :-1], (zi[:, -1:] - za[:, -1:])], axis=1).astype(dtype)
    qmi = (qa * dtype(0.9)).astype(dtype)
    qpi = (qa * dtype(1.1)).astype(dtype)
    args = [jnp.asarray(a) for a in (qa, dza, qmi, qpi, zi, za)]
    new = jax.jit(jax.vmap(lambda *a: M._plm_remap(*a, km)))(*args)
    ref = jax.jit(jax.vmap(lambda *a: _ref_plm_remap(*a, km)))(*args)
    new, ref = np.asarray(new), np.asarray(ref)
    assert new.dtype == ref.dtype and new.tobytes() == ref.tobytes()
    assert np.isfinite(ref).mean() > 0.5


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("scheme", ["wdm6", "wdm7", "wdm5"])
def test_restructured_loops_bitwise_equal_v041(dtype, scheme, monkeypatch):
    module, run_name = {"wdm6": (M, "wdm6_run"), "wdm7": (M7, "wdm7_run"), "wdm5": (M5, "wdm5_run")}[scheme]
    active = 0
    for name, cols, dt, sl in _cases(dtype, scheme):
        calls = {"remap": 0, "fallout": 0}
        new = _run(module, run_name, False, cols, dt, sl, monkeypatch, calls)
        ref = _run(module, run_name, True, cols, dt, sl, monkeypatch, calls)
        assert calls["remap"] > 0 and calls["fallout"] > 0, (name, calls)  # E114: reference really traced
        assert new.keys() == ref.keys()
        for k in ref:
            assert new[k].dtype == ref[k].dtype and new[k].tobytes() == ref[k].tobytes(), (scheme, name, k)
        active += int(np.sum(ref["rainncv"] > 0))
    assert active >= 10  # sedimentation really active in most columns

