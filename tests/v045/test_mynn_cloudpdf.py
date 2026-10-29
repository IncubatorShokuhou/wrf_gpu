"""bl_mynn_cloudpdf 0/1 plus default2 control versus pristine WRF 4.7.1 MYNN.

The frozen fixture holds 384 real CPU-WRF restart columns (night 2026-05-04 03Z d01/d02/d03,
day Swiss 2023-01-15 12Z with ice) and the outputs of the unedited pristine mym_condensation
and mym_predict objects (generator proofs/o5_n333/oracle/make_fixture.py, provenance JSON next
to the npz).  Gates were pre-registered in proofs/o5_n333/oracle/cloudpdf_oracle.py:

* sigma (formed before erf, a WRF output argument): <= 2 REAL ulps;
* cldfra: |d| <= 2.4e-7 + 1e-6 cf;  qc_bl/qi_bl: |d| <= alp*sgm*(|q1|+1)*2.4e-7 + 1e-5 |q|
  (WRF's qll = cld*q1 + eq1 cancels for q1 << 0, so one libm erf/exp ulp is amplified);
* REAL subnormal oracle values against a CPU-flushed 0 are reported, not failed (E117).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_sgs_cloud as S

FIXTURE = Path(__file__).parents[1] / "fixtures" / "o5_n333" / "mynn_cloudpdf_pristine_v1.npz"
ERF_ULP_TOL = 2.4e-7
SGM_REL_TOL = 2.4e-7
TINY = float(np.finfo(np.float32).tiny)
_ORIGINAL_CASE1_SIGMA = S._case1_sigma


@pytest.fixture(scope="module")
def fx():
    data = np.load(FIXTURE)
    meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    return data, meta


def _inputs(data, dtype):
    return {k[3:]: jnp.asarray(data[k], dtype) for k in data.files if k.startswith("in_")}


def _port01(a, cloudpdf):
    kw = dict(theta=a["th"], p=a["p"], exner=a["exner"], dz=a["dz"], thl=a["thl"], qw=a["qw"],
              tsq=a["tsq"], qsq=a["qsq"], cov=a["cov"], el=a["el"], sh=a["sh"])
    out = S.mym_condensation_cloudpdf01(cloudpdf, **kw)
    _qc, _qi, _cf, sgm = S._cloudpdf01_columns(abs(cloudpdf), **kw)
    t = a["th"] * a["exner"]
    qsl, alp, _bet = S._alp_bet(t, a["p"], a["exner"])
    q1 = (a["qw"][..., :-1] - qsl[..., :-1]) / sgm
    return [np.asarray(x, np.float64) for x in out], np.asarray(sgm, np.float64), np.asarray(alp[..., :-1], np.float64), np.asarray(q1, np.float64)


def _violations(data, cloudpdf, out, sgm, alp, q1):
    bad = {}
    if cloudpdf in (0, 1, -1):
        rs = data[f"cond{cloudpdf}_sgm"][:, :-1].astype(np.float64)
        rel = np.abs(sgm - rs) / np.maximum(np.abs(rs), 1e-30)
        bad["sgm"] = int((rel > SGM_REL_TOL).sum())
    for name, x in zip(("qc_bl", "qi_bl", "cldfra_bl"), out):
        r = data[f"cond{cloudpdf}_{name}"].astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(r).all(), name
        if name == "cldfra_bl":
            tol = ERF_ULP_TOL + 1e-6 * np.abs(r)
        else:
            body = alp * sgm * (np.abs(q1) + 1.0) * ERF_ULP_TOL
            tol = np.concatenate((body, np.zeros_like(body[:, :1])), axis=1) + 1e-5 * np.abs(r)
        subnormal = (np.abs(r) < TINY) & (r != 0) & (x == 0)
        bad[name] = int(((np.abs(x - r) > tol) & ~subnormal).sum())
    return bad


def test_fixture_provenance_and_coverage(fx):
    data, meta = fx
    assert meta["columns"] == data["in_dz"].shape[0] == 384
    labels = list(data["labels"])
    assert {lab: labels.count(lab) for lab in set(labels)} == {"night_d01": 96, "night_d02": 96,
                                                                "night_d03": 96, "day_swiss": 96}
    # Real-structured coverage (E88/E102/E154): BL cloud, ice and previous-step variances present.
    assert (data["cond0_cldfra_bl"] >= 0.5).sum() > 500
    assert (data["cond1_qi_bl"] > 0).sum() > 1000
    assert (data["in_qi"] > 1e-8).any(axis=1).sum() > 20
    assert data["in_tsq"].max() > 1e-3 and np.abs(data["in_cov"]).max() > 1e-6 and data["in_sh"].max() > 0.1


@pytest.mark.parametrize("cloudpdf", [0, 1])
def test_native_real32_matches_pristine_mym_condensation(fx, cloudpdf):
    data, _ = fx
    with jax.enable_x64(False):
        out, sgm, alp, q1 = _port01(_inputs(data, jnp.float32), cloudpdf)
    bad = _violations(data, cloudpdf, out, sgm, alp, q1)
    assert bad == {"sgm": 0, "qc_bl": 0, "qi_bl": 0, "cldfra_bl": 0}, bad
    if cloudpdf < 0:
        assert all(np.all(x == 0) for x in out)
    else:
        assert (out[2] > 0).sum() > 1000 and np.all(out[2][:, -1] == 0)


def _compiled_gate(data, cloudpdf):
    with jax.enable_x64(False):
        a = _inputs(data, jnp.float32)
        kw = dict(theta=a["th"], p=a["p"], exner=a["exner"], dz=a["dz"], thl=a["thl"], qw=a["qw"],
                  tsq=a["tsq"], qsq=a["qsq"], cov=a["cov"], el=a["el"], sh=a["sh"])
        actual = jax.jit(lambda values: S.mym_condensation_cloudpdf01(cloudpdf, **values))(kw)
        _, sigma, alp, q1 = _port01(a, cloudpdf)
        bad = _violations(data, cloudpdf, [np.asarray(value, np.float64) for value in actual], sigma, alp, q1)
    assert not any(bad.values()), bad


def _isolated_compiled_gate(tmp_path, option, *, delete_barrier=False):
    """Do not inherit pytest's legacy constants or the host's ambient CPU ISA."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GPUWRF_") and key not in ("XLA_FLAGS", "JAX_DISABLE_JIT")}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_FAST_DEFAULTS="1", GPUWRF_JAX_CACHE="0",
               PYTHONPATH=str(root / "src"), XLA_FLAGS="--xla_cpu_max_isa=AVX", JAX_DISABLE_JIT="0")
    report = tmp_path / "compiled.json"
    command = [sys.executable, str(root / "proofs/v45_mynnopts/check_compiled.py"),
               "--option", str(option), "--out", str(report)]
    if delete_barrier:
        command.append("--delete-barrier")
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout[-1000:] + run.stderr[-3000:]
    result = json.loads(report.read_text())
    assert result["healthy_passed"] and result["release_real"] and result["backend"] == "cpu"
    assert result["xla_flags"] == "--xla_cpu_max_isa=AVX"
    return result


@pytest.mark.parametrize("cloudpdf", [0, 1])
def test_whole_compiled_operator_matches_pristine(tmp_path, cloudpdf):
    _isolated_compiled_gate(tmp_path, cloudpdf)


def test_compiled_gate_rejects_reciprocal_fold_mutant(tmp_path):
    result = _isolated_compiled_gate(tmp_path, 0, delete_barrier=True)
    assert result["mutant_rejected"] and result["mutant_violations"]["cldfra_bl"] > 0


@pytest.mark.parametrize("cloudpdf", [2])
def test_registry_default_case2_matches_pristine(fx, cloudpdf):
    """Regression of the untouched CASE(2) port on the same real columns (bitwise at night)."""
    data, _ = fx
    night = data["labels"] != "day_swiss"
    with jax.enable_x64(False):
        a = _inputs(data, jnp.float32)
        qc, qi, cf = S.mym_condensation_cloudpdf2(theta=a["th"], p=a["p"], exner=a["exner"], dz=a["dz"],
                                                  qw=a["qw"], qc=a["qc"], qi=a["qi"], qs=a["qs"],
                                                  qsq=a["qsq"], pblh=a["pblh"])
    if cloudpdf == -2:
        assert all(np.all(data[f"cond-2_{n}"] == 0) for n in ("qc_bl", "qi_bl", "cldfra_bl"))
        return
    for name, x in zip(("qc_bl", "qi_bl", "cldfra_bl"), (qc, qi, cf)):
        np.testing.assert_array_equal(np.asarray(x)[night], data[f"cond2_{name}"][night])


def _mutant_sigma0(tsq, qsq, cov, bet):
    """Mutant: the WRF DOUBLE t3sq/r3sq/c3sq island evaluated in REAL."""
    t3 = jnp.maximum(tsq, 0.0)
    r3 = jnp.maximum(qsq, 0.0)
    c3 = jnp.copysign(jnp.minimum(jnp.abs(cov), jnp.sqrt(t3 * r3)), cov)
    return jnp.sqrt(jnp.maximum((r3 + bet * bet * t3) - 2.0 * bet * c3, 1.0e-10))


def _mutant_alp_bet(t, p, exner):
    """Mutant: r_d*t*t association instead of WRF r_d*t**2 (the bug the gate first caught)."""
    esat = S.esat_blend(t)
    qsl = S._EP_2_4 * esat / jnp.maximum(np.float32(1.0e-4), p - S._EP_3_4 * esat)
    dqsl = qsl * S._EP_2_4 * S._XLV4 / (S._R_D4 * t * t)
    return qsl, np.float32(1.0) / (np.float32(1.0) + dqsl * S._XLVCP4), dqsl * exner


@pytest.mark.parametrize("cloudpdf,target,mutant", [
    (0, "_case0_sigma", _mutant_sigma0),
    (1, "_alp_bet", _mutant_alp_bet),
    # Mutant: el and sh swapped (caller-wiring error class, E11).
    (1, "_case1_sigma", lambda thl, qw, dz, el, sh, alp, bet: _ORIGINAL_CASE1_SIGMA(thl, qw, dz, sh, el, alp, bet)),
])
def test_gate_is_deletion_sensitive(fx, monkeypatch, cloudpdf, target, mutant):
    """E39/E202: each mutant must be rejected by the same gate (assertion-level kill)."""
    data, _ = fx
    monkeypatch.setattr(S, target, mutant)
    with jax.enable_x64(False):
        out, sgm, alp, q1 = _port01(_inputs(data, jnp.float32), cloudpdf)
    assert sum(_violations(data, cloudpdf, out, sgm, alp, q1).values()) > 0


def test_level2_tsq_cov_matches_pristine_mym_predict(fx):
    from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, _mym_predict_tsq_cov

    data, _ = fx
    with jax.enable_x64(False):
        f32 = lambda k: jnp.asarray(data[k], jnp.float32)  # noqa: E731
        z = jnp.zeros_like(f32("in_th"))
        st = MynnPBLColumnState(z, z, z, f32("in_th"), z, 0.5 * f32("pred_in_qkep"), f32("in_p"),
                                f32("pred_in_rho"), f32("in_dz"), z, z, f32("in_el"))
        tsq, cov = _mym_predict_tsq_cov(st, {"el": f32("in_el"), "pdt": f32("pred_in_pdt"),
                                             "pdc": f32("pred_in_pdc")})
    for name, x in (("tsq", tsq), ("cov", cov)):
        r = data[f"pred_{name}"].astype(np.float64)
        x = np.asarray(x, np.float64)
        assert np.isfinite(x).all()
        assert np.count_nonzero(r) > 10000
        np.testing.assert_allclose(x, r, rtol=1e-6, atol=1e-6 * np.abs(r).max())


def _step_state(nz=20, ncol=4):
    from gpuwrf.physics.mynn_pbl import MynnPBLColumnState

    dz = np.full((ncol, nz), 200.0)
    zmid = np.cumsum(dz, axis=1) - 0.5 * dz
    p = 1.0e5 * np.exp(-zmid / 8000.0)
    theta = 289.0 + 0.002 * zmid
    qv = np.maximum(0.0095 * np.exp(-zmid / 2500.0), 1.0e-6)
    rho = p / (287.0 * theta * (p / 1e5) ** 0.2857)
    rng = np.random.default_rng(3)
    z = np.zeros((ncol, nz))
    st = MynnPBLColumnState(*(jnp.asarray(x) for x in (np.full((ncol, nz), 6.0), z, z, theta, qv,
                                                        np.full((ncol, nz), 0.2), p, rho, dz, z, z,
                                                        rng.random((ncol, nz)) * 50.0)))
    return st.replace(qsq=jnp.asarray(rng.random((ncol, nz)) * 1e-7),
                      tsq=jnp.asarray(rng.random((ncol, nz)) * 1e-2),
                      cov=jnp.asarray((rng.random((ncol, nz)) - 0.5) * 1e-5),
                      sh3d=jnp.asarray(rng.random((ncol, nz)) * 0.5))


@pytest.mark.parametrize("cloudpdf", [0, 1])
def test_column_step_carries_wrf_previous_step_state(cloudpdf):
    from gpuwrf.physics.mynn_pbl import _step_mynn_pbl_impl_with_pblh

    st = _step_state()
    out, pblh = _step_mynn_pbl_impl_with_pblh(st, 30.0, False, None, True, 3000.0, cloudpdf)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(out))
    if cloudpdf == 0:
        tsq, cov = np.asarray(out.tsq), np.asarray(out.cov)
        assert not np.array_equal(tsq, np.asarray(st.tsq)) and (tsq >= 0).all()
        np.testing.assert_array_equal(tsq[:, -1], tsq[:, -2])
        np.testing.assert_array_equal(cov[:, -1], cov[:, -2])
        np.testing.assert_array_equal(np.asarray(out.sh3d), np.asarray(st.sh3d))
    elif cloudpdf in (1, -1):
        sh = np.asarray(out.sh3d)
        assert np.all(sh[:, 0] == 0) and sh[:, 1:].max() > 0
        np.testing.assert_array_equal(np.asarray(out.tsq), np.asarray(st.tsq))
    if cloudpdf < 0:
        assert float(jnp.max(out.qi_bl)) == 0.0


def test_column_step_refuses_missing_inputs_and_unknown_option():
    from gpuwrf.physics.mynn_pbl import _step_mynn_pbl_impl_with_pblh

    st = _step_state().replace(tsq=None, cov=None, sh3d=None)
    for cloudpdf, message in ((0, "tsq/cov"), (1, "sh3d"), (3, "unsupported")):
        with pytest.raises(ValueError, match=message):
            _step_mynn_pbl_impl_with_pblh(st, 30.0, False, None, True, 3000.0, cloudpdf)


def test_namelist_leaves_and_catalog():
    from gpuwrf.contracts.precision import mynn_cloudpdf_leaves
    from gpuwrf.io.scheme_catalog import classify_control

    assert mynn_cloudpdf_leaves({"bl_pbl_physics": 5, "bl_mynn_cloudpdf": 0}) == ("tsq", "cov")
    assert mynn_cloudpdf_leaves({"bl_pbl_physics": 5, "bl_mynn_cloudpdf": 1}) == ("sh3d",)
    assert mynn_cloudpdf_leaves({"bl_pbl_physics": 5, "bl_mynn_cloudpdf": 2}) == ()
    assert mynn_cloudpdf_leaves({"bl_pbl_physics": 1, "bl_mynn_cloudpdf": 0}) == ()
    for value in (0, 1, 2):
        assert classify_control("bl_mynn_cloudpdf", value).status.name == "IMPLEMENTED"
    assert classify_control("bl_mynn_cloudpdf", 3).status.name != "IMPLEMENTED"


@pytest.mark.parametrize("cloudpdf,leaf", [(0, "tsq"), (0, "cov"), (1, "sh3d")])
def test_restart_readers_refuse_missing_previous_step_state(cloudpdf, leaf):
    """E78: an ON run cannot resume from a checkpoint without the held MYNN TSQ/COV/SH3D."""
    from gpuwrf.contracts.state import State
    from gpuwrf.io import restart, wrfrst_netcdf

    namelist = {"bl_pbl_physics": 5, "bl_mynn_cloudpdf": cloudpdf}
    recorded = tuple(f for f in State.__slots__ if f != leaf)
    for validate in (restart._validate_state_field_order, wrfrst_netcdf._validate_state_field_order):
        with pytest.raises(ValueError, match="TSQ/COV/SH3D"):
            validate(recorded, namelist=namelist)
        # The default program may omit them (OFF checkpoints stay readable).
        try:
            validate(recorded, namelist={"bl_pbl_physics": 5, "bl_mynn_cloudpdf": 2})
        except ValueError as exc:
            assert "TSQ/COV/SH3D" not in str(exc)
