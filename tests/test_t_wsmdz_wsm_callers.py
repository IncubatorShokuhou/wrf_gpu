"""t-wsmdz: WSM3/5/6 and WDM5/6/7 caller wiring against pristine WRF.

1. Post-RK microphysics caller operands (solve_em.F:3750 moist_physics_prep_em):
   DELZ = dz8w from (ph+phb)/g with WRF g=9.81 in REAL (:5480/:5488) and
   DEN = 1./(al+alb) (:5542, DRY; al from calc_p_rho_phi hypsometric_opt=2
   :1045-1048).  Expected values come from the compiled untouched statements
   (proofs/t_wsmdz/moist_prep_oracle.json, REAL-structured CPU-WRF c05 column).
2. physics_init.F:1003-1021 has_reqc/i/s=1 for WSM3/5/6 and WDM5/6/7 with
   RRTMG and use_mp_re=1 (compiled flags proofs/t_wsmdz/radius_flags.json): the WSM
   radii must be seeded, held by the adapter and forwarded to both RRTMG calls.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.physics import rrtmg_mp_re as R
from test_v014_dry_source_leaf_wiring import _grid, _state

ROOT = Path(__file__).resolve().parents[1]
PREP = json.loads((ROOT / "proofs/t_wsmdz/moist_prep_oracle.json").read_text())
FLAGS = json.loads((ROOT / "proofs/t_wsmdz/radius_flags.json").read_text())["flags"]
NAMES = ("re_cloud", "re_ice", "re_snow")
SCHEME = {3: "wsm3", 4: "wsm5", 6: "wsm6", 14: "wdm5", 16: "wdm6", 26: "wdm7"}
# DEN/DELZ/DELT positions in each *_physics_tendency call
ARG = {3: dict(den=6, delz=8, dt=9), 4: dict(den=7, delz=9, dt=10), 6: dict(den=8, delz=10, dt=11),
       14: dict(den=10, delz=12, dt=13), 16: dict(den=11, delz=13, dt=14), 26: dict(den=12, delz=14, dt=15)}
RADIUS = (3, 4, 6, 14, 16, 26)
PHASE = json.loads((ROOT / "proofs/t_wsmdz/wsm3_phase_oracle.json").read_text())


def _prep_case(ny, nx, dtype, mp):
    x = {k: np.asarray(v, np.float32) for k, v in PREP["inputs"].items()}
    grid = _grid(nz=3, ny=ny, nx=nx)
    metrics = dataclasses.replace(
        grid.metrics, c3f=jnp.asarray(x["c3f"]), c4f=jnp.asarray(x["c4f"]),
        c3h=jnp.asarray(x["c3h"]), c4h=jnp.asarray(x["c4h"]), p_top=jnp.asarray(x["ptop"][0]))
    grid = dataclasses.replace(grid, metrics=metrics)
    col = lambda v, n: jnp.broadcast_to(jnp.asarray(v, dtype).reshape(n, 1, 1), (n, ny, nx))  # noqa: E731
    # WRF holds ph_2 and phb; its z_at_w adds them in REAL before the division.
    state = _state(grid).replace(
        _cast=False, theta=col([281., 282., 283.5], 3), qv=col([3.7e-3, 3.6e-3, 3.4e-3], 3),
        p=col([75500., 74900., 74150.], 3), ph_total=col(x["ph"] + x["phb"], 4),
        mu_total=jnp.full((ny, nx), x["muts"][0], dtype))
    return grid, state.ensure_conditional_leaves(mp_physics=mp, use_mp_re=0)


@pytest.mark.parametrize("mp", tuple(SCHEME))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("ny,nx", ((1, 1), (2, 3)))
def test_wsm_caller_den_delz_are_pristine_post_rk_prep(monkeypatch, mp, dtype, ny, nx):
    grid, state = _prep_case(ny, nx, dtype, mp)
    seen = {}

    def tendency(*args):
        def observe(den, delz, delt):
            seen.update(den=np.asarray(den).copy(), delz=np.asarray(delz).copy(), dt=float(delt))
        # Test-only boundary observation; absent from production code.
        jax.debug.callback(observe, args[ARG[mp]["den"]], args[ARG[mp]["delz"]], jnp.asarray(args[ARG[mp]["dt"]]))
        return PhysicsTendency(state_replacements={"theta": args[0]}, accumulator_increments={}, diagnostics={})

    monkeypatch.setattr(sa, SCHEME[mp] + "_physics_tendency", tendency)
    adapter = getattr(sa, SCHEME[mp] + "_adapter")
    jax.block_until_ready(jax.jit(lambda s: adapter(s, 18.0, grid))(state))
    assert seen["dt"] == 18.0
    assert seen["delz"].shape == seen["den"].shape == (ny * nx, 3), "flat (ncol, nlev) column batch"
    # Compiled division by the constant g is not the literal REAL quotient
    # (XLA rewrites x/c, E123; CUDA f32 divide is div.full, E84): allow one
    # rounding of each z_at_w quotient and of the difference, nothing more.
    dz8w = np.asarray(PREP["dz8w"], np.float64)
    phi32 = np.asarray(PREP["inputs"]["ph"], np.float32) + np.asarray(PREP["inputs"]["phb"], np.float32)
    z = (phi32 / np.float32(9.81)).astype(np.float64)
    dz_bound = np.spacing(np.abs(z[1:]).astype(np.float32)) + np.spacing(np.abs(z[:-1]).astype(np.float32)) \
        + np.spacing(np.abs(dz8w).astype(np.float32))
    dz_err = np.abs(seen["delz"].astype(np.float64) - dz8w)
    assert np.all(dz_err <= dz_bound), (mp, dz_err.max(axis=0), dz_bound)
    # The port diagnoses al+alb from the stored REAL total geopotential (one
    # rounding of ph+phb per face) instead of WRF's separate ph/phb
    # differences and the (x-alb)+alb round trip: bound by those input ulps.
    phi = np.asarray(PREP["inputs"]["ph"], np.float32) + np.asarray(PREP["inputs"]["phb"], np.float32)
    ulp = np.spacing(np.abs(phi)).astype(np.float64)
    bound = (ulp[1:] + ulp[:-1]) / np.abs(np.diff(phi.astype(np.float64))) + 8 * np.finfo(np.float32).eps
    rel = np.abs(seen["den"].astype(np.float64) / np.asarray(PREP["rho"], np.float64) - 1.0)
    assert np.all(rel <= bound), (mp, rel.max(axis=0), bound)


@pytest.mark.parametrize("row", [r for r in FLAGS if r[0] in RADIUS], ids=lambda r: f"mp{r[0]}_use{r[1]}")
def test_wsm_radius_selection_matches_compiled_pristine_physics_init(monkeypatch, row):
    monkeypatch.setattr(R, "_MP_RE", True)
    mp, use, *flags = row
    assert len(set(flags)) == 1
    assert R.mp_re_active(mp_physics=mp, use_mp_re=use) is bool(flags[0])
    # The has_req block also requires the RRTMG LW/SW pair (physics_init.F:988-1000).
    assert not R.mp_re_active(mp_physics=mp, use_mp_re=use, ra_lw_physics=1)
    assert not R.mp_re_active(mp_physics=mp, use_mp_re=use, ra_sw_physics=1)


@pytest.mark.parametrize("mp", RADIUS)
def test_wsm_radii_are_seeded_held_and_forwarded_to_both_rrtmg_calls(monkeypatch, mp):
    monkeypatch.setattr(R, "_MP_RE", True)
    grid = _grid(nz=3, ny=1, nx=2)
    state = _state(grid).ensure_conditional_leaves(mp_physics=mp, use_mp_re=int(R.mp_re_active(mp_physics=mp)))
    assert R.mp_re_active(mp_physics=mp)
    # physics_init seeds RE_QC/QI/QS_BG before the first radiation call.
    for name, bg in zip(NAMES, (R.RE_CLOUD_BG, R.RE_ICE_BG, R.RE_SNOW_BG)):
        assert getattr(state, name) is not None, name
        np.testing.assert_array_equal(getattr(state, name), jnp.full_like(state.qc, bg, jnp.float32))
    marker = dict(zip(NAMES, (13e-6, 60e-6, 300e-6)))

    def tendency(theta, qv, *args):
        return PhysicsTendency(state_replacements={"theta": theta, "qv": qv},
                               diagnostics={n: jnp.full_like(qv, v) for n, v in marker.items()})

    monkeypatch.setattr(sa, SCHEME[mp] + "_physics_tendency", tendency)
    held = getattr(sa, SCHEME[mp] + "_adapter")(state, 18.0, grid)
    for name, value in marker.items():
        np.testing.assert_array_equal(getattr(held, name), jnp.full_like(state.qc, value, jnp.float32))
    # Next step's radiation reads the held radii (radiation_driver -> RRTMG RE_*).
    sw, lw, *_ = C._rrtmg_column_inputs(held, None, time_utc="2023-01-15_01:00:00",
                                        use_mp_re=int(R.mp_re_active(mp_physics=mp)), mp_physics=mp)
    for column in (sw, lw):
        for name, value in marker.items():
            np.testing.assert_array_equal(getattr(column, name), C._to_columns(getattr(held, name)))


@pytest.mark.parametrize("mp", RADIUS)
def test_wsm_radius_leaves_stay_absent_when_wrf_disables_them(monkeypatch, mp):
    monkeypatch.setattr(R, "_MP_RE", True)
    grid = _grid(nz=3, ny=1, nx=2)
    state = _state(grid).ensure_conditional_leaves(mp_physics=mp, use_mp_re=0)
    assert all(getattr(state, n) is None for n in NAMES)

    def tendency(theta, qv, *args):
        return PhysicsTendency(state_replacements={"theta": theta, "qv": qv},
                               diagnostics={n: jnp.full_like(qv, 1e-5) for n in NAMES})

    monkeypatch.setattr(sa, SCHEME[mp] + "_physics_tendency", tendency)
    out = getattr(sa, SCHEME[mp] + "_adapter")(state, 18.0, grid)
    assert all(getattr(out, n) is None for n in NAMES)


@pytest.mark.parametrize("mp", (3, 4))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_wsm3_rrtmg_inputs_use_pristine_cold_phase_view(monkeypatch, mp, dtype):
    # WSM3 keeps ice in qc and snow in qr; RRTMG LW/SW restage them below
    # 273.15 K (rrtmg_lw.F:12069-12085 / rrtmg_sw.F:10670-10686, compiled
    # verbatim in proofs/t_wsmdz/wsm3_phase_oracle.json). WSM5 (mp4) has F_QI:
    # no restaging. T comes from the fixture (threshold adversaries included).
    x = {k: np.asarray(v, np.float32) for k, v in PHASE["inputs"].items()}
    nk = x["T1D"].size
    grid = _grid(nz=nk, ny=1, nx=2)
    col = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype).reshape(nk, 1, 1), (nk, 1, 2))  # noqa: E731
    zero = jnp.zeros((nk, 1, 2), dtype)
    state = _state(grid).replace(_cast=False, qc=col(x["QC1D"]), qr=col(x["QR1D"]), qi=zero, qs=zero)
    monkeypatch.setattr(C, "_temperature_from_theta", lambda theta, p: col(x["T1D"]))
    sw, lw, *_ = C._rrtmg_column_inputs(state, None, time_utc="2023-01-15_01:00:00", use_mp_re=0,
                                        mp_physics=mp, use_mynn_sgs_cloud=False, cloud_mp_physics=mp)
    expected = PHASE["outputs"] if mp == 3 else {"QC1D": x["QC1D"], "QI1D": 0 * x["QC1D"], "QS1D": 0 * x["QC1D"]}
    for column in (sw, lw):
        for name, key in (("qc", "QC1D"), ("qi", "QI1D"), ("qs", "QS1D")):
            got = np.asarray(getattr(column, name)).reshape(-1, nk)
            want = np.broadcast_to(np.asarray(expected[key], np.float32), (2, nk)).astype(got.dtype)
            np.testing.assert_array_equal(got, want, err_msg=f"mp={mp} {name}")
