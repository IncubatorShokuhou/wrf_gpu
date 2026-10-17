"""cu=6 modified Tiedtke on REAL c06 columns vs the UNMODIFIED WRF module_cu_tiedtke.F.

Fixture ``proofs/t_cfix_cu/c06_tiedtke/c06_tiedtke_rows.npz`` (build_fixture.py): WRF phy_prep inputs of the
pristine serial c06 run at 2023-01-15_15:00 (WRF's own held RQVFTEN, this step's RQVBLTEN/QFX) for 9 real
rows + 1 assembled row of 42 real columns whose own first ascent has no convection; reference outputs of
CU_TIEDTKE called per tile row (``row_*``, = serial WRF) and per single column (``one_*``).

Two port defects this pins (the 5-column synthetic savepoint oracle exercised neither):
  * cuasc_new 3.1 (module_cu_tiedtke.F:1633-1649) resets ldcum for every column before the ascent and
    starts zbuoy at the cloud-base buoyancy (ktype 1) -> mid-level convection (ktype 3) can start in any
    column (c06 15:00: 1349 of 1764 WRF columns; the port had 0);
  * :959 ``if(icum.eq.0) return`` is decided per tile row -> tiedtke_tile_rows masks by any() over x.

Pre-registered gates (t-cfix-cu notes 2026-10-16, before the port ran on the fixture): raining-column set
identical; RAINCV sum within 1 % and rel RMS <= 0.10; every R*CUTEN rel RMS <= 0.02; the assembled
icum = 0 row exactly zero (WRF: RQV round trip <= 1e-9).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "proofs" / "t_cfix_cu" / "c06_tiedtke" / "c06_tiedtke_rows.npz"
# Pinned in manifest/reproducibility_assets.json (t-cfix-cu-c06-tiedtke-rows-v1).
FIXTURE_SHA256 = "95ca642f31cbc6f464b1a2bea171717d3c3766a563ea7e9ab999e08b731e4dda"
NAMES = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN")
DT = 18.0


@pytest.fixture(scope="module")
def fx():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    return dict(np.load(FIXTURE))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b * b)))


def _cols(a):
    a = np.asarray(a, np.float64)
    return np.moveaxis(a, 0, -1).reshape(a.shape[1] * a.shape[2], a.shape[0])


def _gate(port, ref, raincv, ref_raincv):
    assert np.array_equal(raincv > 0, ref_raincv > 0), (int((raincv > 0).sum()), int((ref_raincv > 0).sum()))
    assert abs(raincv.sum() / ref_raincv.sum() - 1.0) <= 0.01
    assert _rel(raincv, ref_raincv) <= 0.10
    for name in NAMES:
        assert _rel(port[name], ref[name]) <= 0.02, (name, _rel(port[name], ref[name]))


def test_tile_rows_match_wrf_row_calls_on_real_c06(fx):
    import jax
    import jax.numpy as jnp
    from gpuwrf.coupling.scan_adapters import tiedtke_tile_rows

    jax.config.update("jax_enable_x64", True)
    nz, ny, nx = fx["in_t_phy"].shape
    c = lambda n: jnp.asarray(_cols(fx["in_" + n]))  # noqa: E731
    out = tiedtke_tile_rows(
        c("t_phy"), c("qv"), c("qc"), c("qi"), c("p_hyd"), c("p_hyd_w"), c("dz8w")[:, :nz], c("rho"),
        c("pi_phy"), c("u_phy"), c("v_phy"), c("w"), c("qvften"), c("qvpblten"),
        jnp.asarray(fx["in_qfx"].reshape(-1), jnp.float64), jnp.asarray(fx["in_xland"].reshape(-1), jnp.float64),
        jnp.asarray(np.broadcast_to(fx["in_znu"][None, :], (ny * nx, nz)), jnp.float64), DT, ny=ny, nx=nx)
    back = lambda a: np.moveaxis(np.asarray(a).reshape(ny, nx, nz), -1, 0)  # noqa: E731
    port = dict(zip(("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN", "RUCUTEN", "RVCUTEN"),
                    (back(a) for a in out[:8])))
    raincv = np.asarray(out[8]).reshape(ny, nx)
    ref = {n: fx["row_" + n] for n in NAMES}
    _gate(port, ref, raincv, fx["row_RAINCV"])
    # the assembled icum = 0 row: WRF returns before any tendency although each column convects in its row
    assert np.max(np.abs(fx["row_RQVCUTEN"][:, -1])) <= 1.0e-9
    for name in NAMES:
        assert not np.any(port[name][:, -1]), name
    assert not np.any(raincv[-1])
    # mid-level convection is what carries the real rows (ktype 3 needs the ldcum reset)
    assert int((fx["row_RAINCV"][:-1] > 0).sum()) >= 15


def test_numpy_reference_matches_wrf_single_column_calls(fx):
    from gpuwrf.physics.cumulus_tiedtke import tiedtke_column

    nz, ny, nx = fx["in_t_phy"].shape
    port = {n: np.zeros((nz, ny, nx)) for n in NAMES}
    raincv = np.zeros((ny, nx))
    keys = {"RTHCUTEN": "RTHCUTEN", "RQVCUTEN": "RQVCUTEN", "RQCCUTEN": "RQCCUTEN", "RQICUTEN": "RQICUTEN",
            "RUCUTEN": "RUCUTEN", "RVCUTEN": "RVCUTEN"}
    for j in range(ny):
        for i in range(nx):
            col = lambda n: np.asarray(fx["in_" + n][:, j, i], np.float64)  # noqa: E731
            out = tiedtke_column(col("t_phy"), col("qv"), col("qc"), col("qi"), col("p_hyd"), col("p_hyd_w"),
                                 col("dz8w")[:nz], col("rho"), col("pi_phy"), col("u_phy"), col("v_phy"), col("w"),
                                 col("qvften"), col("qvpblten"), float(fx["in_qfx"][j, i]),
                                 float(fx["in_xland"][j, i]), np.asarray(fx["in_znu"], np.float64), DT, stepcu=1)
            for name, key in keys.items():
                port[name][:, j, i] = np.asarray(out[key])
            raincv[j, i] = float(out["RAINCV"])
    _gate(port, {n: fx["one_" + n] for n in NAMES}, raincv, fx["one_RAINCV"])


def test_adapter_runs_the_tile_row_kernel():
    from gpuwrf.coupling import scan_adapters

    tree = ast.parse(inspect.getsource(scan_adapters.tiedtke_adapter))
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "tiedtke_tile_rows" in called and "tiedtke_column_jax" not in called
