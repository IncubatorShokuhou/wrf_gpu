"""Grid FDDA (grid_fdda=1) port vs the staged pristine-WRF oracle (lane o2-fdda).

The savepoints (proofs/o2_fdda/savepoints/*.npz, generator + Fortran driver in
proofs/o2_fdda/oracle/) hold real-structured CPU-WRF crops (Swiss winter, PROD
summer day/night, land+sea+terrain) and the oracle's per-call outputs of the WRF
chain: auxinput10 read -> fddagd_driver control -> FDDAGD (unmodified pristine
source) -> calculate_phy_tend coupling -> phy_fg_ten ownership -> phy_prep_part2
decoupling.  Each case replays the same call sequence through the port.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.assimilation import fdda_grid as fg

jax.config.update("jax_enable_x64", True)

SAVEPOINTS = Path(__file__).resolve().parents[3] / "proofs" / "o2_fdda" / "savepoints"
MANIFEST = json.loads((SAVEPOINTS / "MANIFEST.json").read_text())
CASES = sorted(MANIFEST["cases"])
NAMES = ("run", "rvn", "rth", "rqv")
OPTION_KEYS = ("guv", "gt", "gq", "interval_m", "end_h", "end_h_sfc", "stepfg", "fgdtzero",
               "if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q", "if_zfac_uv",
               "k_zfac_uv", "if_zfac_t", "k_zfac_t", "if_zfac_q", "k_zfac_q", "if_ramping", "dtramp_min")


def _load(name):
    meta = MANIFEST["cases"][name]
    data = dict(np.load(SAVEPOINTS / name))
    dtype = jnp.float32 if name.endswith("_r4.npz") else jnp.float64
    opts = meta["opts"]
    options = fg.GridFddaOptions(**{k: opts[k] for k in OPTION_KEYS})
    calls = data["calls"]
    rec_steps = []
    for itimestep, secs, rec, _frame in calls:
        if rec > 0:
            assert rec == len(rec_steps) + 1
            rec_steps.append(int(itimestep) - 1)
    nrec = len(rec_steps)
    stacked = {f"{n}_{s}": jnp.stack([jnp.asarray(data[f"rec{r + 1}_{n}_{s}"]) for r in range(nrec)])
               for n in fg.FDDA_FIELDS for s in ("old", "new")}
    inputs = fg.GridFddaInputs(**stacked, options=options, record_steps=tuple(rec_steps))
    return meta, data, dtype, inputs


def _replay(name, *, mutate=None):
    """Run the port over the oracle's call sequence; return per-call port arrays + error-model scales."""

    meta, data, dtype, inputs = _load(name)
    if mutate is not None:
        inputs = mutate(inputs)
    carry = fg.initial_fdda_carry(inputs, dtype)
    # fddagdinit zeroes the tendencies; record 1 is read at call 0 (before the first fddagd).
    c1h, c2h = jnp.asarray(data["c1h"]), jnp.asarray(data["c2h"])
    specified = bool(meta["opts"]["specified"])
    out = []
    for c, (itimestep, secs, _rec, frame) in enumerate(data["calls"]):
        f = lambda n: jnp.asarray(data[f"frame{frame}_{n}"])  # noqa: E731
        carry = fg.fdda_driver_tendencies(carry, inputs, f("u"), f("v"), f("th_phy"), f("qv"), f("pblh"),
                                          jnp.asarray(data["ht"]), f("z_at_w"), float(secs), int(itimestep),
                                          dtype)
        port = {f"unc_{n}": a for n, a in zip(NAMES, carry.tendencies)}
        # FMA error model: the only contractible products are old*(1-coef)+new*coef (|val| <= A) and
        # c1h*mu+c2h; scale = g*A per field (u, v, t, q), the operand magnitude of WRF's own expression.
        amp = [np.maximum(np.abs(np.asarray(getattr(carry, f"{n}_old"))),
                          np.abs(np.asarray(getattr(carry, f"{n}_new")))) for n in fg.FDDA_FIELDS]
        gfac = [meta["opts"]["guv"], meta["opts"]["guv"], meta["opts"]["gt"], meta["opts"]["gq"]]
        # theta: (val - th3d) + 300. rounds at |th3d| ~ 300 K, so a 1-ulp change of val can move it by ulp(th3d).
        amp[2] = amp[2] + np.abs(np.asarray(f("th_phy")))
        scale = {n: g * a for n, g, a in zip(NAMES, gfac, amp)}
        mu_2, mub = f("mu_2"), f("mub")
        mass = float(np.max(np.abs(np.asarray(c1h)[:, None, None] * np.asarray(mub + mu_2)[None]
                                   + np.asarray(c2h)[:, None, None])))
        carry = fg.couple_fdda_rates(carry, mub + mu_2, mu_2, mub, c1h, c2h, dtype)
        port.update({f"cpl_{n}": a for n, a in zip(NAMES, carry.tendencies)})
        if c == 0:
            port.update({f"own_{n}": a for n, a in zip(
                NAMES, fg.owned_fdda_tendf(carry, specified_or_nested=specified))})
            port.update(u_old=carry.u_old, v_old=carry.v_old, t_old=carry.t_old, q_old=carry.q_old)
        carry = fg.finish_fdda_rates(carry, f("muts"), c1h, c2h, dtype)
        port.update({f"dec_{n}": a for n, a in zip(NAMES, carry.tendencies)})
        out.append((c, port, scale, mass))
    return data, out


def _compare(data, out, *, ulps=4):
    """|port - WRF| <= ulps*eps*(|WRF| + M*g*A): FMA-contraction of the two products (XLA CPU and GPU
    contract a*b+c; the WRF build has no -march, i.e. no FMA) changes val by <= 1 ulp of A."""

    worst = {}
    held_scale = None
    for c, port, scale, mass in out:
        for key, value in port.items():
            ref = data[f"call{c}_{key}"]
            got = np.asarray(value)
            assert got.dtype == ref.dtype, (c, key, got.dtype, ref.dtype)
            eps = np.finfo(ref.dtype).eps
            if key in ("u_old", "v_old", "t_old", "q_old"):
                # itimestep==1 overwrite incl. the untouched east/north boundary faces (record value)
                assert np.array_equal(got, ref), (c, key)
                continue
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), (c, key)
            name = key[4:]
            m = {"unc": 1.0, "dec": 2.0}.get(key[:3], mass)
            base = scale[name]
            if held_scale is not None:
                base = np.maximum(base, held_scale[name])
            tol = ulps * eps * (np.abs(ref) + m * base)
            err = np.abs(got - ref)
            worst[(c, key)] = float((err / np.maximum(tol, np.finfo(ref.dtype).tiny)).max())
            assert (err <= tol).all(), (c, key, int((err > tol).sum()), float(err.max()))
            # Structural zeros (masked levels, early return, boundary faces) must stay exact zeros; only a
            # cancellation zero inside the bound (FMA flips (val-th)+300 to exactly 0) may differ.
            flips = (got == 0) != (ref == 0)
            assert int(flips.sum()) <= max(2, flips.size // 1000), (c, key, int(flips.sum()))
        held_scale = scale  # a held (non-recomputed) rate carries the previous call's error scale
    return worst


@pytest.mark.parametrize("name", CASES)
def test_port_matches_pristine_wrf_oracle(name):
    """Every call of every case: pair overwrite, uncoupled, coupled, owned and decoupled rates."""

    data, out = _replay(name)
    worst = _compare(data, out)
    assert worst  # non-vacuous


@pytest.mark.parametrize("name", CASES)
def test_port_is_bitwise_when_nothing_contracts(name):
    """coef == 0 recompute calls (record-read times, step 1, dtramp>0 ramp): val = old exactly, so the
    uncoupled rates must be bit-identical to the gfortran REAL / REAL(8) oracle."""

    meta = MANIFEST["cases"][name]
    data, out = _replay(name)
    opts = meta["opts"]
    checked = 0
    for c, port, _scale, _mass in out:
        itimestep, secs = (int(x) for x in data["calls"][c][:2])
        if not (itimestep == 1 or itimestep % int(opts["stepfg"]) == 0):
            continue
        minutes = secs / 60.0
        ramp_start = opts["end_h"] * 60.0
        coef_zero = (minutes % opts["interval_m"] == 0) or (
            opts["if_ramping"] == 1 and opts["dtramp_min"] > 0 and ramp_start <= minutes <= ramp_start
            + opts["dtramp_min"])
        if not coef_zero:
            continue
        for n in NAMES:
            ref, got = data[f"call{c}_unc_{n}"], np.asarray(port[f"unc_{n}"])
            assert np.array_equal(got, ref), (name, c, n, float(np.abs(got - ref).max()))
            checked += 1
    assert checked >= 4, checked


def test_savepoints_are_non_vacuous():
    """E154: every case has non-zero rates in most calls, varied PBL top levels and land points."""

    for name, meta in MANIFEST["cases"].items():
        nonzero_calls = sum(1 for c in meta["nonzero_unc"] if c["run"] and c["rth"] and c["rqv"])
        assert nonzero_calls >= 3, name
        for frame in ("frame1", "frame2"):
            assert meta["census"][frame]["kpbl_distinct"] >= 3, (name, frame)
        assert meta["census"]["land_fraction"] > 0.3, name
    regimes = {(m["opts"]["if_ramping"], np.sign(m["opts"]["dtramp_min"]), m["opts"]["stepfg"] > 1,
                m["opts"]["fgdtzero"], m["opts"]["specified"]) for m in MANIFEST["cases"].values()}
    assert len(regimes) == 4


def test_mutant_coef_is_killed():
    """Deletion check: dropping the time interpolation (target = NEW analysis) fails the oracle."""

    name = "swiss_basic_r4.npz"

    def target_new(inputs):
        return fg.GridFddaInputs(**{f"{n}_{s}": getattr(inputs, f"{n}_new") for n in fg.FDDA_FIELDS
                                    for s in ("old", "new")},
                                 options=inputs.options, record_steps=inputs.record_steps)

    data, out = _replay(name, mutate=target_new)
    with pytest.raises(AssertionError):
        _compare(data, out)


def test_mutant_pbl_weight_is_killed(monkeypatch):
    """Deletion check: no PBL weights (wpbl = 1) fails the if_no_pbl_nudging case."""

    monkeypatch.setattr(fg, "_kpbl_weights", lambda zpbl, zagl, dtype: jnp.asarray(1.0, dtype))
    data, out = _replay("swiss_pbl_zfac_hold_r4.npz")
    with pytest.raises(AssertionError):
        _compare(data, out)


def test_mutant_decouple_is_killed(monkeypatch):
    """Deletion check: holding the UNcoupled rate across steps (no phy_prep_part2) fails the hold chain."""

    monkeypatch.setattr(fg, "finish_fdda_rates", lambda carry, muts, c1h, c2h, dtype: fg.GridFddaCarry(
        **{**{f: getattr(carry, f) for f in ("u_old", "u_new", "v_old", "v_new", "t_old", "t_new",
                                             "q_old", "q_new")},
           "rundgdten": carry.rundgdten, "rvndgdten": carry.rvndgdten,
           "rthndgdten": carry.rthndgdten, "rqvndgdten": carry.rqvndgdten}))
    data, out = _replay("swiss_pbl_zfac_hold_r4.npz")
    with pytest.raises(AssertionError):
        _compare(data, out)
