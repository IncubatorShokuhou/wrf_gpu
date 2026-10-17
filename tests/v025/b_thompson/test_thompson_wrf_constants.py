"""GPUWRF_THOMPSON_WRF_CONSTANTS (BD94 constant audit): WRF's own REAL Thompson constants on the native REAL path.

WRF folds PARAMETERs and runs thompson_init and the constant sub-expressions of mp_thompson in REAL, one rounding per
operation, and builds every gamma constant with its own WGAMMA = EXP(GAMMLN(x)) (Numerical-Recipes Lanczos, GAMMLN
rounded to REAL; module_mp_thompson.F:5325/:5371). The port folded the same expressions in Python f64 with the exact
gamma function. Exact WRF bits: proofs/thompson/wrf_constants/wrf_constants.txt (gfortran 14.3 with the pristine
FCOPTIM compiling WRF's declarations/statements and GAMMLN/WGAMMA verbatim, wrf_constants.F90).
"""
import ast
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics import thompson_constants as k

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "proofs" / "thompson" / "wrf_constants" / "wrf_constants.txt"
WRF = k.WRF_REAL_CONSTANTS
# the port's f64 fold of each selected constant / constant sub-expression (what the key-off path keeps)
LEGACY = {name: getattr(k, name) for name in WRF if hasattr(k, name)}
LEGACY.update({
    "LAMC_PREFIX": k.NT_C * k.AM_R * k.CCG2_NU12 * k.OCG1_NU12,  # lamc = (nc*am_r*ccg(2)*ocg1/rc)**obmr
    "DC_G_PREFIX": (k.CCG3_NU12 * k.OCG2_NU12) ** k.OBMR,  # Dc_g = ((ccg(3)*ocg2)**obmr/lamc)*1.E6
    "D0R_X075": k.D0R * 0.75,
    "D0S_X01": 0.1 * k.D0S,
    "PNR_WAU_DEN": k.AM_R * k.NU_C_MP8 * 10.0 * k.D0R**3,  # DOUBLE prr_wau / REAL denominator: differs as DOUBLE
})
# REAL view of the legacy fold minus the WRF bits (gamma constants: exact gamma vs WGAMMA; the rest: f64 vs per-op REAL)
LEGACY_ULPS = {"AM_R": -1, "AM_I": -1, "D0I": 5, "CRG7": -1, "CIG6": -1, "CCG5_NU12": -1, "OCG2_NU12": -5,
               "T1_QR_QC": -1, "T1_QR_QI": -1, "T2_QR_QI": -2, "T1_QS_QC": -1, "T1_QS_QI": -1, "T1_MELT_QS": -1,
               "T2_MELT_QS": -1, "CGE9_OLD": 1, "T1_QG_QC_OLD": 2, "T2_MELT_QG_OLD": 1, "LAMC_PREFIX": 8,
               "DC_G_PREFIX": -6, "D0R_X075": 1, "D0S_X01": -1, "PNR_WAU_DEN": 0}
# release-consumed port constants that already carry WRF's REAL bits (guarded, not selected)
ALREADY_WRF = ("AM_G_MP8", "RHO_NOT", "LFUS", "SC3", "OBMR", "OBMI", "OBMG", "CRE3", "CRE6", "CRE7", "CRE8", "CRE12",
               "CIE2", "CGE11_OLD", "CRG2", "CRG3", "CRG6", "CRG9", "CRG10", "CRG11", "CRG12", "ORG1", "ORG2", "ORG3",
               "CIG1", "CIG2", "CIG3", "CIG5", "CIG7", "OIG1", "OIG2", "T1_QR_EV", "T2_QR_EV", "T1_SUBL_QS",
               "T2_SUBL_QS", "T1_SUBL_QG", "T1_MELT_QG", "T2_SUBL_QG_OLD")


def _receipt():
    rows = (line.split() for line in RECEIPT.read_text().splitlines() if not line.startswith("#"))
    result = {}
    for name, bits, value in rows:
        assert np.isfinite(float(value)), (name, "non-finite receipt value")
        storage = np.int64 if name == "NR_MAX_LAMR3" else np.int32
        real = np.float64 if name == "NR_MAX_LAMR3" else np.float32
        assert np.isfinite(np.asarray(int(bits), dtype=storage).view(real)), (name, "non-finite receipt bits")
        result[name] = int(bits)
    return result


def _bits(value):
    assert np.isfinite(value), "non-finite candidate value"
    real = np.float32(value)
    assert np.isfinite(real), "non-finite candidate REAL cast"
    return int(real.view(np.int32))


def test_table_is_the_gfortran_receipt_and_exact_real():
    receipt = _receipt()
    for name, value in WRF.items():
        assert _bits(value) == receipt[name], (name, _bits(value), receipt[name])
        assert float(np.float32(value)) == value, name  # exactly REAL: DOUBLE contexts see DBLE(REAL) as in WRF


def test_unselected_release_constants_already_carry_wrf_bits():
    receipt = _receipt()
    assert {name: _bits(getattr(k, name)) - receipt[name] for name in ALREADY_WRF} == dict.fromkeys(ALREADY_WRF, 0)
    # the rain-number cap (3.+.672)/(D0r*.75) then **bm_r in DOUBLE (:1896-1897): the f64 fold is already WRF's
    assert np.float64(((3.0 + 0.672) / (k.D0R * 0.75)) ** 3.0).view(np.int64) == receipt["NR_MAX_LAMR3"]


def test_legacy_folds_are_not_wrf():
    """The divergence the key removes, per constant (REAL ulps; PNR_WAU_DEN differs only as DOUBLE)."""
    assert {name: _bits(LEGACY[name]) - _bits(WRF[name]) for name in LEGACY} == LEGACY_ULPS
    assert all(LEGACY[name] != WRF[name] for name in LEGACY)
    cgg63 = np.float32(5.0) * np.float32(k.CGG6_OVER_CGG3_OLD)
    assert cgg63 != np.float32(np.float32(5.0) * np.float32(WRF["CGG6_OLD"])) * np.float32(WRF["OGG3"])


@pytest.mark.parametrize("key,native,expect_wrf", [("1", "1", True), ("0", "1", False), ("1", "0", False),
                                                   (None, "1", False)])
def test_selector_is_native_real_only(monkeypatch, key, native, expect_wrf):
    """Key ON + native REAL -> WRF's REAL values; the f64 legacy path (NATIVE_REAL=0) and key OFF keep the old ones."""
    if key is None:
        monkeypatch.delenv("GPUWRF_THOMPSON_WRF_CONSTANTS", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_THOMPSON_WRF_CONSTANTS", key)
    monkeypatch.setenv("GPUWRF_THOMPSON_NATIVE_REAL", native)
    for name, legacy in LEGACY.items():
        assert tc._wrf(name, legacy) == (WRF[name] if expect_wrf else legacy), name
    assert (tc._am_r(), tc._am_i(), tc._d0i()) == tuple(WRF[n] if expect_wrf else LEGACY[n] for n in ("AM_R", "AM_I", "D0I"))
    x = jnp.float32(5.0)
    want = x * WRF["CGG6_OLD"] * WRF["OGG3"] if expect_wrf else x * k.CGG6_OVER_CGG3_OLD
    assert tc._times_cgg63(x, k.CGG6_OVER_CGG3_OLD) == want


# constants that only occur inside the folded sub-expressions LAMC_PREFIX / DC_G_PREFIX, and the folded REAL products
CHAIN_PARTS = {"CCG2_NU12", "OCG1_NU12", "CCG3_NU12"}
PRODUCTS = {"D0R*0.75": "D0R_X075", "0.1*D0S": "D0S_X01"}


def _bare_uses(path):
    """Selected names / folded products in function bodies outside the legacy argument of a _wrf() call."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    shielded = {id(node) for call in ast.walk(tree)
                if isinstance(call, ast.Call) and getattr(call.func, "attr", getattr(call.func, "id", None)) == "_wrf"
                for node in ast.walk(call.args[1])}
    def own_nodes(fn):  # the function's own nodes; nested defs are scanned under their own name
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            if not isinstance(node, ast.FunctionDef):
                stack.extend(ast.iter_child_nodes(node))
                yield node

    uses = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in own_nodes(fn):
            if id(node) in shielded:
                continue
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in WRF or name in CHAIN_PARTS:
                uses.append((fn.name, name))
            elif isinstance(node, ast.BinOp):
                text = re.sub(r"\s|tc\.", "", ast.get_source_segment(src, node)).replace(".75", "0.75").replace("00.75", "0.75")
                if text in PRODUCTS:
                    uses.append((fn.name, text))
    return sorted(set(uses))


def test_every_release_use_goes_through_the_selector():
    """Deletion guard over all consumers: removing any _wrf(...) wrapper (or adding a bare use) fails here."""
    allowed = {("_am_r", "AM_R"), ("_am_i", "AM_I"), ("_d0i", "D0I"),
               # graupel NUMBER fall speed feeds only the Ng carry, which the native path re-diagnoses (B56, :3287)
               ("_fall_speeds", "CRG7"),
               # rain-number caps (3.+.672)/(D0r*.75) then **bm_r in DOUBLE: the f64 fold is already WRF's (NR_MAX_LAMR3)
               ("_apply_warm_rain_rates", "D0R*0.75"), ("_ice_sources_with_process_flags", "D0R*0.75"),
               ("inactive", "D0R*0.75")}
    src = ROOT / "src" / "gpuwrf"
    uses = set(_bare_uses(src / "physics" / "thompson_column.py")) | set(_bare_uses(src / "kernels" / "phys_thompson_full.py"))
    assert uses <= allowed, sorted(uses - allowed)
    assert {("_fall_speeds", "CRG7"), ("inactive", "D0R*0.75")} <= uses  # the scanner sees both forms (sanity)


C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0",
           GPUWRF_THOMPSON_IMPLICIT_SED="0", GPUWRF_THOMPSON_MIXED_PHASE_WRF="1")


def _mixed_phase_state(ncol=2, nz=12):
    f = lambda v: jnp.full((ncol, nz), v, jnp.float32)  # noqa: E731
    T, p = f(263.0), f(7.0e4)
    qv = (0.95 * tc.saturation_mixing_ratio_liquid(p, T)).astype(jnp.float32)
    return tc.ThompsonColumnState(qv=qv, qc=f(2.0e-4), qr=f(1.0e-4), qi=f(1.0e-5), qs=f(2.0e-4), qg=f(1.0e-4),
                                  Ni=f(1.0e4), Nr=f(1.0e3), Ns=f(0.0), Ng=f(0.0), T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p, T, qv), dz=f(250.0), w=f(0.0))


@pytest.mark.parametrize("key", ("0", "1"))
def test_release_full_column_traces_every_wrf_constant(monkeypatch, key):
    """Consumer-level deletion check: the release C24 full-column program (mixed phase on, sedimentation included)
    carries every selected WRF REAL value with the key and none of them without it."""
    from gpuwrf.kernels import phys_thompson_full as full
    for name, value in {**C24, "GPUWRF_THOMPSON_WRF_CONSTANTS": key}.items():
        monkeypatch.setenv(name, value)
    jax.clear_caches()
    text = str(jax.make_jaxpr(lambda s: full.full_column(s, 18.0, interpret=True))(_mixed_phase_state()))
    jax.clear_caches()
    present = {name for name, value in WRF.items() if repr(value) in text or f"{np.float32(value)}" in text}
    distinct = set(WRF) - {"OGG3"}  # ogg3 = 1/6 also enters as org3/oig2 elsewhere
    assert present & distinct == (distinct if key == "1" else set()), sorted(present & distinct)


@pytest.mark.parametrize("key", ("0", "1"))
def test_mp_re_radii_trace_the_selected_constants(monkeypatch, key):
    """MP_RE calc_effectRad (REAL in WRF, :5647/:5655): the traced cloud/ice lambdas carry the selected am_r / am_i."""
    from gpuwrf.physics import rrtmg_mp_re as mp_re
    monkeypatch.setenv("GPUWRF_THOMPSON_WRF_CONSTANTS", key)
    x = jnp.full((2, 3), 1.0, jnp.float32)
    jx = str(jax.make_jaxpr(lambda T, p, qv, qc, qi, Ni, qs: mp_re.calc_thompson_effective_radii(
        T, p, qv, qc, qi, Ni, qs))(270.0 * x, 8.0e4 * x, 1e-3 * x, 1e-4 * x, 1e-5 * x, 1e4 * x, 1e-4 * x))
    wrf = {f"{np.float32(WRF['AM_R'])}", f"{np.float32(WRF['AM_I'])}"}
    old = {f"{np.float32(k.AM_R)}", f"{np.float32(k.AM_I)}"}
    lits = set(re.findall(r"(?<![\d.])(?:523|466)\.\d+", jx))
    assert lits == (wrf if key == "1" else old), (key, lits)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("role", ("candidate", "receipt_value", "receipt_bits", "coincident"))
def test_table_gate_rejects_nonfinite(monkeypatch, tmp_path, bad, role):
    """E200: reject before bit comparisons, including matching non-finite payloads."""
    receipt = RECEIPT.read_text()
    if role in ("candidate", "coincident"):
        monkeypatch.setitem(WRF, "AM_R", float(bad))
    if role != "candidate":
        rows = receipt.splitlines()
        i = next(i for i, line in enumerate(rows) if line.startswith("AM_R "))
        name, bits, value = rows[i].split()
        if role in ("receipt_bits", "coincident"):
            bits = str(int(np.float32(bad).view(np.int32)))
        if role in ("receipt_value", "coincident"):
            value = str(bad)
        rows[i] = " ".join((name, bits, value))
        p = tmp_path / "bad-receipt.txt"
        p.write_text("\n".join(rows))
        monkeypatch.setitem(test_table_is_the_gfortran_receipt_and_exact_real.__globals__, "RECEIPT", p)
    with pytest.raises(AssertionError, match="non-finite"):
        test_table_is_the_gfortran_receipt_and_exact_real()
