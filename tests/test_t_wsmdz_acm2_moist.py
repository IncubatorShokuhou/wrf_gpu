"""t-wsmdz: ACM2 (bl_pbl_physics=7) moist caller wiring against pristine WRF.

pbl_driver.F:1592 passes QC3D/QI3D to ACMPBL: they switch EDDYX's moist
Richardson number (module_bl_acm.F imvdif=1, condensate > 1e-5 kg/kg) and are
mixed into RQCBLTEN/RQIBLTEN.  The pristine fixtures run the UNMODIFIED
module_bl_acm.F (proofs/t_wsmdz/build_acm2_moist_oracle.sh) on a stable
mountain-night column with an elevated ice (case 7) or cloud-water (case 8)
layer above the PBL -- the c05 hour-1 hotspot class (EXCH_H 25-91 m2/s there).
Tolerances are the frozen v0.6.0 ACM2 savepoint tolerances.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.physics.pbl_acm2 import acm2_columns
from test_v014_dry_source_leaf_wiring import _grid, _state

ROOT = Path(__file__).resolve().parents[1]
TEND_ABS, TEND_REL, EXCH_ABS, EXCH_REL = 3.0e-6, 2.0e-3, 5.0e-4, 2.0e-3  # test_v060_pbl_acm2.py
FIELDS = (("RUBLTEN", "u", TEND_ABS, TEND_REL), ("RVBLTEN", "v", TEND_ABS, TEND_REL),
          ("RTHBLTEN", "theta", TEND_ABS, TEND_REL), ("RQVBLTEN", "qv", TEND_ABS, TEND_REL),
          ("RQCBLTEN", "qc", TEND_ABS, TEND_REL), ("RQIBLTEN", "qi", TEND_ABS, TEND_REL),
          ("EXCH_H", "exch_h", EXCH_ABS, EXCH_REL))


def _run(case, with_condensate):
    d = json.loads((ROOT / f"proofs/t_wsmdz/acm2_moist_case_{case}.json").read_text())
    c, s = d["columns"], d["scalars"]
    col = lambda n: jnp.asarray(np.asarray(c[n], np.float64)[None, :])  # noqa: E731
    one = lambda n: jnp.asarray([float(s[n])], jnp.float64)  # noqa: E731
    kw = dict(qc=col("QC"), qi=col("QI")) if with_condensate else {}
    with jax.enable_x64(True):
        out = acm2_columns(col("U"), col("V"), col("THETA"), col("T"), col("QV"), col("RR"), col("DZ"),
                           pblh_initial=one("PBLH_INITIAL"), ust=one("UST"), hfx=one("HFX"), qfx=one("QFX"),
                           wspd=one("WSPD"), mut=one("MUT"), dt=float(s["DT"]), xtime=float(s["XTIME"]), **kw)
        out = {k: np.asarray(v) for k, v in out.items()}
    return c, out


def _misses(c, out):
    bad = {}
    for wrf, port, atol, rtol in FIELDS:
        ref = np.asarray(c[wrf], np.float64)
        got = np.asarray(out.get(port, np.zeros_like(ref[None, :])), np.float64)[0][: ref.size]
        err = np.abs(got - ref)
        ok = (err <= atol) | (err <= rtol * np.maximum(np.abs(ref), 1e-12))
        if not (np.isfinite(got).all() and ok.all()):
            bad[wrf] = float(err.max())
    return bad


@pytest.mark.parametrize("case", (7, 8))
def test_acm2_with_condensate_matches_pristine_moist_eddyx(case):
    c, out = _run(case, True)
    assert {"qc", "qi"} <= set(out), "ACM2 must return RQCBLTEN/RQIBLTEN"
    assert _misses(c, out) == {}, _misses(c, out)


@pytest.mark.parametrize("case", (7, 8))
def test_acm2_without_condensate_misses_the_pristine_elevated_mixing(case):
    # Discrimination: the zero-condensate (pre-fix caller) solution cannot
    # reproduce WRF's moist-Ri exchange coefficient in these cases.
    c, out = _run(case, False)
    assert "EXCH_H" in _misses(c, out)


def test_acm2_adapter_passes_state_condensate_and_applies_its_rates(monkeypatch):
    grid = _grid(nz=4, ny=1, nx=2)
    state = _state(grid)
    qc = jnp.broadcast_to(jnp.asarray([0., 2e-4, 0., 0.], state.qc.dtype).reshape(4, 1, 1), (4, 1, 2))
    qi = jnp.broadcast_to(jnp.asarray([0., 0., 1e-4, 3e-5], state.qi.dtype).reshape(4, 1, 1), (4, 1, 2))
    state = state.replace(_cast=False, qc=qc, qi=qi)
    seen = {}
    rate = {"qc": 1e-6, "qi": -2e-6}

    def kernel(u, *args, **kw):
        seen.update(qc=np.asarray(kw.get("qc")), qi=np.asarray(kw.get("qi")))
        z = jnp.zeros_like(u)
        return dict(u=z, v=z, theta=z, qv=z, qc=z + rate["qc"], qi=z + rate["qi"],
                    pblh=jnp.zeros(u.shape[0]), exch_h=z)

    monkeypatch.setattr(sa, "acm2_columns", kernel)
    out = sa.acm2_pbl_adapter(state, 18.0, grid)
    cols = lambda a: np.moveaxis(np.asarray(a), 0, -1).reshape(2, 4)  # noqa: E731
    assert seen["qc"] is not None and seen["qi"] is not None, "QC3D/QI3D must reach ACMPBL"
    np.testing.assert_array_equal(seen["qc"], cols(qc).astype(seen["qc"].dtype))
    np.testing.assert_array_equal(seen["qi"], cols(qi).astype(seen["qi"].dtype))
    np.testing.assert_allclose(np.asarray(out.qc), np.asarray(qc) + 18.0 * rate["qc"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.qi), np.asarray(qi) + 18.0 * rate["qi"], rtol=1e-6, atol=1e-12)
