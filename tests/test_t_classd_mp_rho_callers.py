"""NSSL (mp17/18/19/21/22), Ferrier (mp5/15) and Goddard 4-ice (mp7) microphysics caller density.

WRF solve_em.F:3750 calls moist_physics_prep_em immediately before microphysics_driver; it
overwrites grid%rho with the DRY 1./(al+alb) (module_big_step_utilities_em.F:5542) and forms
dz8w from (ph+phb)/g (:5480/:5488, g=9.81, no clamp). microphysics_driver passes them as NSSL
DN/DZ (module_microphysics_driver.F:2223+), Ferrier RHO_PHY/DZ8W (:2938, :2971) and Goddard
4-ice RHO/DZ8W (:2353). The (1+qv)/alt form belongs to the earlier phy_prep (:4856).
Expected values: compiled untouched statements (proofs/t_wsmdz/moist_prep_oracle.json, a
REAL-structured original CPU-WRF column); bounds identical to tests/test_t_wsmdz_wsm_callers.py.
Empirical cross-check (lane t-classd, proofs/t_classd/REFL_CPU_HISTORY_c14_{dry,moist}.json):
original CPU-WRF NSSL REFL_10CM is reproduced to 7.6e-6 dBZ with the dry density, 0.043 dBZ off
with (1+qv)/alt.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_t_wsmdz_wsm_callers import _prep_case

ROOT = Path(__file__).resolve().parents[1]
PREP = json.loads((ROOT / "proofs/t_wsmdz/moist_prep_oracle.json").read_text())


class _Seen(Exception):
    pass


def _capture(monkeypatch, mp):
    """Run the operational adapter eagerly and capture the kernel's DEN/DZ (ncol, nlev)."""
    seen = {}

    def stop(rho, dz):
        seen.update(den=np.asarray(rho), delz=np.asarray(dz))
        raise _Seen

    if mp in (17, 18, 19, 21, 22):
        from gpuwrf.physics.nssl2mom import adapter
        monkeypatch.setattr(adapter, "nssl2mom_column", lambda fields, *a, **k: stop(fields["rho"], fields["dz"]))
        variant = {19: dict(variant=19), 21: dict(variant=21), 22: dict(hail_enabled=False)}.get(mp, {})
        run = lambda s, g: adapter.nssl2mom_adapter(s, 18.0, g, first_step=True, **variant)  # noqa: E731
    elif mp in (5, 15):
        from gpuwrf.physics.ferrier import adapter
        monkeypatch.setattr(adapter, "native_column", lambda th, qv, qc, qr, qi, p, pii, rho, dz, *a, **k: stop(rho, dz))
        run = lambda s, g: adapter.ferrier_adapter(s, 18.0, g, option=mp)  # noqa: E731
    else:
        from gpuwrf.physics.goddard4ice import native
        monkeypatch.setattr(native, "native_column",
                            lambda th, qv, qc, qr, qi, qs, qg, qh, rho, exner, p, dz, *a, **k: stop(rho, dz))
        from gpuwrf.coupling import goddard4ice_adapter as adapter
        run = lambda s, g: adapter.goddard4ice_adapter(s, 18.0, g)  # noqa: E731
    return seen, run


def _columns(a, nz=3):
    a = np.asarray(a, np.float64)
    if a.shape[0] == nz and a.ndim == 3:          # (nz, ny, nx) -> (ncol, nz)
        a = np.moveaxis(a, 0, -1).reshape(-1, nz)
    return a.reshape(-1, nz)


@pytest.mark.parametrize("mp", (17, 18, 19, 21, 22, 5, 15, 7))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_mp_caller_den_delz_are_pristine_post_rk_prep(monkeypatch, mp, dtype):
    if mp in (5, 15, 7) and dtype == jnp.float64:
        pytest.skip("Ferrier/Goddard native kernels are REAL-only; the caller casts to float32")
    grid, state = _prep_case(2, 3, dtype, mp)
    seen, run = _capture(monkeypatch, mp)
    with pytest.raises(_Seen):
        run(state, grid)
    den, delz = _columns(seen["den"]), _columns(seen["delz"])
    assert den.shape == delz.shape == (6, 3)
    dz8w = np.asarray(PREP["dz8w"], np.float64)
    phi = np.asarray(PREP["inputs"]["ph"], np.float32) + np.asarray(PREP["inputs"]["phb"], np.float32)
    z = (phi / np.float32(9.81)).astype(np.float64)
    dz_bound = np.spacing(np.abs(z[1:]).astype(np.float32)) + np.spacing(np.abs(z[:-1]).astype(np.float32)) \
        + np.spacing(np.abs(dz8w).astype(np.float32))
    assert np.all(np.abs(delz - dz8w) <= dz_bound), (mp, np.abs(delz - dz8w).max(axis=0), dz_bound)
    ulp = np.spacing(np.abs(phi)).astype(np.float64)
    bound = (ulp[1:] + ulp[:-1]) / np.abs(np.diff(phi.astype(np.float64))) + 8 * np.finfo(np.float32).eps
    rel = np.abs(den / np.asarray(PREP["rho"], np.float64) - 1.0)
    assert np.all(rel <= bound), (mp, rel.max(axis=0), bound)
    # Deletion sensitivity: the phy_prep total density would be (1+qv) larger.
    qv = np.asarray([3.7e-3, 3.6e-3, 3.4e-3])
    assert np.all(qv > 10 * bound)


@pytest.mark.parametrize("mp", (18, 5, 7))
def test_isolated_callers_without_metrics_use_dry_eos_density(monkeypatch, mp):
    """No grid (isolated callers): the dry EOS density, never (1+qv) times it."""
    from gpuwrf.coupling.physics_couplers import _rho_from_state
    _grid, state = _prep_case(1, 2, jnp.float32, mp)
    seen, run = _capture(monkeypatch, mp)
    with pytest.raises(_Seen):
        run(state, None)
    expected = _columns(jnp.asarray(_rho_from_state(state.ensure_conditional_leaves(mp_physics=mp)), jnp.float32))
    np.testing.assert_allclose(_columns(seen["den"]), expected, rtol=2e-7, atol=0)


def test_nssl_dz_has_no_one_metre_clamp(monkeypatch):
    """WRF dz8w is the raw (ph+phb)/g difference; the old NSSL caller clamped it at 1 m."""
    grid, state = _prep_case(1, 2, jnp.float32, 18)
    phi = np.array(state.ph_total, np.float32)
    phi[2] = phi[1] + np.float32(0.5 * 9.81)  # a 0.5 m layer
    state = state.replace(_cast=False, ph_total=jnp.asarray(phi))
    seen, run = _capture(monkeypatch, 18)
    with pytest.raises(_Seen):
        run(state, grid)
    delz = _columns(seen["delz"])
    z = (phi / np.float32(9.81)).astype(np.float32)
    np.testing.assert_array_equal(delz[:, 1], np.float64((z[2] - z[1])[0, 0]))
    assert np.all(delz[:, 1] < 0.6)
