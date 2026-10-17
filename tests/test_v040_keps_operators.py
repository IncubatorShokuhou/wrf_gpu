"""Deterministic EEPS/KEPS substrate vs unchanged WRF REAL32 helper oracles.

No whole-PBL claim: the fixture boundary excludes the upstream undefined
variables reproduced in proofs/s2_keps/source_probe.py. Fixed 16-ulp field-scale
gate was declared before producing these fixtures. Both arms must be finite.
"""
from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import pbl_eeps as eeps
from gpuwrf.physics import pbl_keps as keps


FIXTURE = Path(__file__).resolve().parents[1]/"proofs/s2_keps/operators_real32.json"
ULPS = 16


def test_oracle_fixture_provenance():
    fixture = json.loads(FIXTURE.read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "c656e0edd9342af2a4bba19e6cedce47e97041804ce9f0547e0a99e6d7bb2b35"
    assert hashlib.sha256(FIXTURE.with_name("operators_oracle.f90").read_bytes()).hexdigest() == fixture["driver_sha256"]
    assert fixture["source_sha256"] == {
        "share/module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
        "phys/module_bl_keps.F": "a79058405affcd293740b15137002e788d168a5eba51586d14cee41342ffc204",
        "phys/module_bl_eepsilon.F": "b1ea6db55866af15cb3c6962b1a15fa512eda803078f7304e6a133462d7e12de",
    }
    assert len(fixture["records"]) == 18
    assert all(len(r["output"]) == 18 for r in fixture["records"])


def inputs(record):
    data = {k: np.asarray(v, np.float32) for k, v in record["input"].items()}
    data.update({k: np.float32(record[k]) for k in ("dt", "roughness", "ustar", "phi_epsilon", "phi_m")})
    return data


def evaluate(x):
    result = {
        "shear": keps.shear(x["u"], x["v"], x["dz"]),
        "buoyancy": keps.buoyancy(jnp.float32(9.81), x["theta"], x["theta0"], x["dz"],
                                   x["tke"], x["tpe"], x["prandtl"]),
        "invert": keps.invert(x["lower"], x["diagonal"], x["upper"], x["rhs"]),
        "countergradient": keps.countergradient(x["dz"], jnp.float32(9.81), x["theta0"], x["tke"],
                                                 x["epsilon"], x["tpe"], x["gradient"]),
    }
    result["exchange_m"], result["exchange_h"] = keps.exchange(x["tke"], x["epsilon"], x["dz"], x["prandtl_face"])
    for first in (1, 2):
        result[f"diffusion_{first}"], result[f"flux_{first}"] = keps.diffusion(
            x["dt"], x["rhs"], x["rho"], x["rho_face"], x["diffusivity"], x["epsilon_source"],
            x["gradient"], x["surface"], x["volume"], x["dz"], first=first)
    result["tpe_implicit"], result["tpe_explicit"], result["tpe_coeff"] = keps.tpe_terms(
        x["gradient"], jnp.float32(9.81), x["theta0"], x["tke"], x["epsilon"], x["tpe"], x["prandtl"])
    result["analytic_tke"], result["analytic_epsilon"], result["analytic_source"] = keps.analytic_tke(
        x["dt"], x["dz"], x["roughness"], x["analytic_shear"], x["analytic_buoy"], x["tke"],
        x["epsilon"], x["prandtl"], x["length"], x["ustar"], x["phi_epsilon"], x["phi_m"],
        x["richardson"], x["nsquare"], x["epsilon_source"])
    result["eeps_solution"], result["eeps_modified_c"] = eeps.tridiag(x["upper"], x["diagonal"], x["lower"], x["rhs"])
    return result


def check_fields(got, expected, context):
    assert set(got) == set(expected), (context, "missing output")
    for field, ref in expected.items():
        actual = np.asarray(got[field])
        reference = np.asarray(ref, np.float32)
        assert actual.dtype == np.float32, (context, field, actual.dtype)
        assert actual.shape == reference.shape, (context, field, actual.shape, reference.shape)
        assert np.isfinite(actual).all(), f"{context} {field}: nonfinite candidate"
        assert np.isfinite(reference).all(), f"{context} {field}: nonfinite reference"
        scale = np.max(np.abs(reference))
        error = np.max(np.abs(actual.astype(np.float64)-reference.astype(np.float64)))
        tolerance = ULPS*float(np.spacing(scale)) if scale else 0.
        assert error <= tolerance, f"{context} {field}: error={error:.9g} tolerance={tolerance:.9g}"


@pytest.mark.parametrize("levels", [8, 12, 44])
def test_operators_match_pristine_wrf_real32(levels):
    """Literal eager REAL32 reference, including the two FMA-sensitive operators."""
    assert jax.default_backend() == "cpu"
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["precision"] == "WRF REAL32"
    assert fixture["tolerance"]["field_scale_float32_ulps"] == ULPS
    records = [r for r in fixture["records"] if r["n"] == levels]
    with jax.disable_jit():
        for rec in records:
            result = evaluate(jax.tree.map(jnp.asarray, inputs(rec)))
            check_fields(result, rec["output"], rec["name"])


FMA_SENSITIVE = {"shear", "analytic_tke", "analytic_epsilon", "analytic_source"}


@pytest.mark.parametrize("levels", [8, 12, 44])
def test_seven_operators_match_pristine_wrf_native_jit(levels):
    """Six KEPS helpers plus EEPS TRIDIAG; exclude shear and analytic_tke."""
    records = [r for r in json.loads(FIXTURE.read_text())["records"] if r["n"] == levels]
    batch = {k: np.stack([inputs(r)[k] for r in records]) for k in inputs(records[0])}
    def compiled(x):
        return {k: v for k, v in evaluate(x).items() if k not in FMA_SENSITIVE}
    result = jax.jit(jax.vmap(compiled))(batch)
    for i, rec in enumerate(records):
        check_fields({k: v[i] for k, v in result.items()},
                     {k: v for k, v in rec["output"].items() if k not in FMA_SENSITIVE}, rec["name"])


def test_active_regimes_and_finite_reference_census():
    records = json.loads(FIXTURE.read_text())["records"]
    signs, rescues, floor_hits = set(), 0, 0
    for rec in records:
        x = inputs(rec)
        c = np.float32(.09)*((np.float32(1.44)-1)*x["analytic_shear"][1:]
                             +(np.float32(1.44)-1)*x["analytic_buoy"][1:]/x["prandtl"][1:])
        signs.update(np.sign(c).tolist())
        rescues += np.count_nonzero((x["tke"][1:]/x["epsilon"][1:] > 1.e4)&(x["epsilon"][1:] <= 1.e-6))
        floor_hits += np.count_nonzero(np.asarray(rec["output"]["exchange_m"])[1:] == np.float32(.09))
        assert all(np.isfinite(v).all() for v in map(np.asarray, rec["output"].values()))
    assert signs == {-1., 0., 1.}
    assert rescues > 0 and floor_hits > 0
    assert all(any(np.count_nonzero(r["output"][field]) for r in records) for field in (
        "shear", "buoyancy", "tpe_explicit", "countergradient", "flux_1", "analytic_source"))


def test_helper_hlo_has_no_fp64():
    rec = json.loads(FIXTURE.read_text())["records"][0]
    hlo = jax.jit(evaluate).lower(inputs(rec)).compiler_ir("hlo").as_hlo_text()
    assert "f64[" not in hlo


@pytest.mark.parametrize("which", ["candidate", "reference"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_numeric_gate_rejects_nonfinite(which, value):
    got, ref = {"field": np.ones(3, np.float32)}, {"field": np.ones(3, np.float32)}
    (got if which == "candidate" else ref)["field"][1] = value
    with pytest.raises(AssertionError, match="nonfinite"):
        check_fields(got, ref, "nonfinite mutant")


def test_oracle_gate_kills_removed_tpe_buoyancy(monkeypatch):
    """Genuine source mutant: remove KEPS's phi contribution, same numeric gate."""
    source = inspect.getsource(keps.buoyancy)
    term = "+(g/theta0[1:-1])*phi"
    assert source.count(term) == 1
    namespace = {"jnp": jnp, "_real32": keps._real32}
    exec(compile(source.replace(term, ""), "<removed-tpe-buoyancy>", "exec"), namespace)
    monkeypatch.setattr(keps, "buoyancy", namespace["buoyancy"])
    rec = json.loads(FIXTURE.read_text())["records"][0]
    x = inputs(rec)
    got = jax.jit(keps.buoyancy)(jnp.float32(9.81), x["theta"], x["theta0"], x["dz"],
                                x["tke"], x["tpe"], x["prandtl"])
    with pytest.raises(AssertionError, match="buoyancy: error="):
        check_fields({"buoyancy": got}, {"buoyancy": rec["output"]["buoyancy"]}, rec["name"])
