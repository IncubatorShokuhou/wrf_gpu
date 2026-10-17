"""v0.3.4 o1-grell: operational wiring of cu=5 (Grell-3D) / cu=93 (Grell-Devenyi).

Coupled smoke through the real operational physics step (``_physics_step_forcing``)
on a 12x12 convective grid (G3/GD act only >= 4 points from a non-periodic
boundary; the ideal smoke grid is periodic, so both schemes compute everywhere
and G3's conv_grell_spread3d writes the 4..7 interior): finite state, convective
rain, theta/qv tendencies applied, untouched outside the WRF write region; cudt>0
fails closed with a named reason; the adapter routes to the WRF kernels.
CPU only.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.coupling import scan_adapters  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    UnsupportedSchemeSelection,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402

import test_v013_operational_smoke as smoke  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proofs" / "v034"))
import grell_oracle_io as gio  # noqa: E402

NZ = 44


def _grell_state(grid):
    """Smoke State carrying realistic deep/dry-upper/marginal soundings (the
    oracle generator's), sub-saturated and with a stratosphere -- the shallow
    supersaturated b2 smoke column gives G3/GD no cloud top / no valid k22.

    The Grell callers hand WRF phy_prep ``P=p_hyd``/``P8W=p_hyd_w`` (from MU and the hybrid
    coefficients); ``DycoreMetrics.flat`` sets ``c1h=eta`` (WRF terrain-following: c1=1, c2=0).
    So the soundings sit on sigma metrics: per-column MU from the sounding surface pressure,
    p from WRF's hydrostatic recursion, T from the sounding theta, hypsometric heights."""
    nz, ny, nx = NZ, grid.ny, grid.nx
    grid = dataclasses.replace(grid, metrics=dataclasses.replace(
        grid.metrics, c1h=jnp.ones((nz,)), c2h=jnp.zeros((nz,)), c1f=jnp.ones((nz + 1,)), c2f=jnp.zeros((nz + 1,))))
    state = smoke._convective_state(grid)
    tile = gio.make_tile(11, nx=grid.nx, ny=grid.ny, kx=NZ,
                         regimes=("deep", "deep_dryupper", "marginal"))
    lev = lambda n: tile["f3"][n].transpose(1, 2, 0)  # (nx,k,ny) -> (k,ny,nx)
    t, q, pi = (lev(n)[:NZ] for n in ("t", "q", "pi"))
    theta = t / pi
    p_top = float(np.asarray(grid.metrics.p_top))
    dnw = np.asarray(grid.metrics.dnw, np.float64)
    psfc = lev("p8w")[0]
    mu = (psfc - p_top) / np.sum((1.0 + q) * (-dnw)[:, None, None], axis=0)
    faces = [np.full((ny, nx), p_top)]
    for k in range(nz - 1, -1, -1):  # phy_prep :4950-4957
        faces.append(faces[-1] - (1.0 + q[k]) * mu * dnw[k])
    pw = np.stack(faces[::-1])
    p = 0.5 * (pw[:-1] + pw[1:])
    tv = theta * (p / 1.0e5) ** (287.0 / 1004.5) * (1.0 + 0.608 * q)
    zf = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(287.0 * tv / 9.81 * np.log(pw[:-1] / pw[1:]), axis=0)])
    theta_m = theta * (1.0 + 461.6 / 287.0 * q)
    ph = jnp.asarray(9.81 * zf)
    return grid, state.replace(theta=jnp.asarray(theta_m), qv=jnp.asarray(q), p=jnp.asarray(p),
                               p_total=jnp.asarray(p), ph=ph, ph_total=ph, mu=jnp.asarray(mu),
                               qc=jnp.zeros_like(state.qc), w=jnp.asarray(lev("w")))


def _setup(cu):
    grid, state = _grell_state(smoke._grid(nz=NZ, ny=12, nx=12))
    nml = smoke._namelist(grid, dt_s=60.0, mp_physics=0, bl_pbl_physics=0,
                          sf_sfclay_physics=0, cu_physics=cu)
    return grid, state, nml


@pytest.mark.parametrize("cu", (5, 93))
def test_grell_operational_step_triggers_and_rains(cu):
    _grid, state, nml = _setup(cu)
    _resolve_operational_suite(nml)
    from gpuwrf.coupling.cu_forcing import initial_cu_forcing
    # itimestep 1: WRF's held RTHFTEN/RQVFTEN are zero (t-cfix-cu F7 carry leaf).
    carry = initial_operational_carry(state).replace(cu_forcing=initial_cu_forcing(state.theta))
    after = _physics_step_forcing(carry, nml, 0.0, run_radiation=False).state
    assert smoke._all_finite(after), f"cu={cu} produced a non-finite field"
    rain = np.asarray(after.rainc_acc) - np.asarray(state.rainc_acc)
    assert rain.max() > 0.0, f"cu={cu} did not rain"
    assert (rain >= 0.0).all()
    dth = np.abs(np.asarray(after.theta) - np.asarray(state.theta)).max(axis=0)
    assert dth.max() > 0.0, f"cu={cu} applied no theta tendency"
    if cu == 5:
        # conv_grell_spread3d writes RTHCUTEN only on ids+4..ide-5 (even periodic).
        ring = np.ones((12, 12), bool)
        ring[4:8, 4:8] = False
        assert dth[ring].max() == 0.0, "G3 theta tendency leaked outside WRF's write region"


def test_grell_cudt_cadence_fails_closed():
    _grid, _state, nml = _setup(5)
    nml = dataclasses.replace(nml, cumulus_cadence_steps=5, cudt_minutes=5.0)
    with pytest.raises(UnsupportedSchemeSelection) as excinfo:
        _resolve_operational_suite(nml)
    assert "cudt>0" in str(excinfo.value)


@pytest.mark.parametrize("cu, kernel", ((5, "g3drv_tile"), (93, "grelldrv_tile")))
def test_adapter_routes_to_wrf_kernel(cu, kernel, monkeypatch):
    """Deletion-sensitive routing: the adapter must call the WRF-port kernel."""
    from gpuwrf.physics import _grell_cup_jax as G

    calls = []
    real = getattr(G, kernel)

    def spy(**kw):
        calls.append(kernel)
        return real(**kw)

    monkeypatch.setattr(G, kernel, spy)
    grid, state, _nml = _setup(cu)
    scan_adapters.CU_SCAN_ADAPTERS[cu](state, 60.0, grid)
    assert calls == [kernel]
