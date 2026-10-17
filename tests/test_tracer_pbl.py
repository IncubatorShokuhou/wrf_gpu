"""Passive-tracer mixing against extracted pristine WRF REAL32 routines."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.tracers.pbl import tracer_pbl_tendency


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "proofs/u2_tracers/pbl_fixture.json"
MANIFEST = FIXTURE.with_suffix(".manifest.json")
CASES = json.loads(FIXTURE.read_text())
FMA_FIXTURE = ROOT / "proofs/u2_tracers/pbl_fma_fixture.json"


def inputs(record):
    return [np.asarray(record[k], dtype=np.float32)
            for k in ("q", "rho", "dz", "exch_h", "exch_m", "dt")]


def assert_wrf(record, actual):
    q, *_, dt = inputs(record)
    expected = np.asarray(record["tendency"], dtype=np.float32)
    budget = 16 * np.finfo(np.float32).eps * (np.abs(q) / dt + np.abs(expected))
    assert np.isfinite(actual).all()
    assert actual.dtype == np.float32
    assert np.all(np.abs(actual - expected) <= budget), (
        float(np.max(np.abs(actual - expected))), float(np.max(budget)))


def test_pristine_fixture_seal():
    manifest = json.loads(MANIFEST.read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    assert manifest["source_sha256"]
    assert manifest["extracted_routines_sha256"]
    assert set(manifest["cases"]) == set(CASES)
    assert "16*eps32" in manifest["tolerance"]


@pytest.mark.parametrize("case", CASES)
def test_tracer_mixing_original_wrf(case):
    record = CASES[case]
    with jax.disable_jit():
        actual = np.asarray(tracer_pbl_tendency(*inputs(record)))
    assert_wrf(record, actual)
    np.testing.assert_array_equal(actual,np.asarray(record["tendency"],np.float32))
    # Original top mass-level row is identity, for every active tracer/column.
    np.testing.assert_array_equal(actual[:, -1], np.float32(0))


@pytest.mark.parametrize("case", CASES)
def test_compiled_tracer_mixing_within_original_wrf_fma_spread(case):
    fma_cases=json.loads(FMA_FIXTURE.read_text())
    fma_manifest=json.loads(FMA_FIXTURE.with_suffix(".manifest.json").read_text())
    assert hashlib.sha256(FMA_FIXTURE.read_bytes()).hexdigest()==fma_manifest["fixture_sha256"]
    original=CASES[case]
    fused=fma_cases[case]
    for key in ("q","rho","dz","exch_h","exch_m","dt"):
        np.testing.assert_array_equal(original[key],fused[key])
    literal=np.asarray(original["tendency"],np.float32)
    fma=np.asarray(fused["tendency"],np.float32)
    actual=np.asarray(jax.jit(tracer_pbl_tendency)(*inputs(original)))
    assert np.isfinite(actual).all() and np.isfinite(literal).all() and np.isfinite(fma).all()
    assert actual.dtype==np.float32
    assert np.all(actual>=np.minimum(literal,fma))
    assert np.all(actual<=np.maximum(literal,fma))
    np.testing.assert_array_equal(actual[:,-1],np.float32(0))


def test_tracer_mixing_uses_heat_exchange_and_ignores_surface_face():
    record = CASES["heterogeneous"]
    q, rho, dz, kh, km, dt = inputs(record)
    original = np.asarray(tracer_pbl_tendency(q,rho,dz,kh,km,dt))
    altered = kh.copy()
    altered[0] *= np.float32(100)
    altered[-1] *= np.float32(100)
    np.testing.assert_array_equal(original, tracer_pbl_tendency(q,rho,dz,altered,km*0,dt))
    assert np.count_nonzero(original) > original.size // 2


def test_tracer_mixing_genuine_exchange_deletion_is_killed():
    record = CASES["heterogeneous"]
    q, rho, dz, kh, km, dt = inputs(record)
    # Delete actual diffusion, retaining the complete solver and valid inputs.
    mutant = np.asarray(tracer_pbl_tendency(q,rho,dz,kh*0,km,dt))
    with pytest.raises(AssertionError):
        assert_wrf(record, mutant)


def test_tracer_mixing_contract_rejects_wrong_shape_and_dtype():
    q,rho,dz,kh,km,dt = inputs(CASES["heterogeneous"])
    with pytest.raises(TypeError, match="float32"):
        tracer_pbl_tendency(q.astype(np.int32),rho,dz,kh,km,dt)
    with pytest.raises(TypeError, match="float32"):
        tracer_pbl_tendency(q.astype(np.float64),rho,dz,kh,km,dt)
    with pytest.raises(ValueError, match="exchange"):
        tracer_pbl_tendency(q,rho,dz,kh[:-1],km,dt)
    with pytest.raises(ValueError, match="tracers"):
        tracer_pbl_tendency(q[0],rho,dz,kh,km,dt)


def test_tracer_mixing_has_no_fp64_or_host_callback():
    hlo = jax.jit(tracer_pbl_tendency).lower(*inputs(CASES["heterogeneous"])).as_text()
    assert "f64" not in hlo
    assert "xla_python_cpu_callback" not in hlo
