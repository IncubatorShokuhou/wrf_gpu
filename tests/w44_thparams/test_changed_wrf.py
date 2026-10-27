"""Retained changed-PARAMETER/source-literal WRF oracles on real columns.

The experimental names are exercised only through the low-level kernel for
qualification; the public runtime binder refuses them until their GPU gates.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics.thompson_parameters import PARAMETERS, validate_parameters

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
NAMES = ("tno", "ato", "av_c", "nt50", "nt150", "av_i", "av_r",
         "autoconversion_threshold", "autoconversion_rate")
spec = importlib.util.spec_from_file_location("thompson_wrf_increment_gate",
                                             ROOT / "proofs/w44_thparams/oracle_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def raw_outputs(result):
    raw = {str(i): np.asarray(value).copy() for i, value in enumerate(jax.tree.leaves(result))}
    assert all(np.isfinite(value).all() for value in raw.values())
    return raw


def assert_wrf_gate(case, reference, output):
    z, idx, _, _, _, _ = case
    scores = gate.score_arrays(z, idx, reference, output)
    assert all(row["failed_columns"] == 0 for row in scores.values()), f"WRF increment gate: {scores}"


@pytest.fixture(scope="module")
def native_environment():
    with pytest.MonkeyPatch.context() as patch:
        for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS", "COLUMN_LAYOUT", "FULL_COLUMN_EARLY_EXIT"):
            patch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
        patch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
        patch.setenv("GPUWRF_THOMPSON_SED_PREP_FUSED", "0")
        patch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
        jax.clear_caches()
        yield
        jax.clear_caches()


@pytest.fixture(scope="module")
def case(native_environment):
    assert jax.devices()[0].platform == "cpu"
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    fixture = ROOT / manifest["fixture"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(fixture, allow_pickle=False) as archive:
        z = dict(archive)
    assert all(np.isfinite(value).all() for name, value in z.items() if name.startswith(("in_", "wrf_")))
    idx = np.array([z["names"].tolist().index(name) for name in manifest["selected_columns"]])
    assert len(idx) == 24 and np.all(z["dt"][idx] == 18.)
    f = lambda name: jnp.asarray(z["in_" + name][idx], jnp.float32)
    temp, pressure, vapor = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(vapor)
    state = tc.ThompsonColumnState(qv=vapor, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"),
        Ni=f("ni"), Nr=f("nr"), Ns=zero, Ng=zero, T=temp, p=pressure,
        rho=tc.density_from_pressure_temperature(pressure, temp, vapor), dz=f("dz8w"), w=f("w"))
    defaults = {name: row[0] for name, row in PARAMETERS.items()}
    compiled = jax.jit(lambda state, controls: full.full_column(state, 18., interpret=True, parameters=controls))
    default = raw_outputs(jax.block_until_ready(compiled(state, validate_parameters(defaults))))
    yield z, idx, state, compiled, defaults, (manifest, default)
    jax.clear_caches()


def reference(case, name):
    manifest = case[-1][0]
    row = manifest["references"][name]
    for field in ("npz", "receipt"):
        assert hashlib.sha256((FIXTURES / row[field]).read_bytes()).hexdigest() == row[field + "_sha256"]
    receipt = json.loads((FIXTURES / row["receipt"]).read_text())
    assert receipt["selection"] == manifest["selected_columns"]
    assert receipt["values"] == row["values"]
    with np.load(FIXTURES / row["npz"], allow_pickle=False) as archive:
        ref = dict(archive)
    assert all(np.isfinite(value).all() for value in ref.values())
    return ref, {**case[4], **row["values"]}


@pytest.mark.parametrize("name", NAMES)
def test_changed_value_matches_pristine_wrf_and_kills_default_omission(case, name):
    ref, controls = reference(case, name)
    output = raw_outputs(jax.block_until_ready(case[3](case[2], validate_parameters(controls))))
    assert_wrf_gate(case, ref, output)
    with pytest.raises(AssertionError, match="WRF increment gate"):
        assert_wrf_gate(case, ref, case[-1][1])


@pytest.mark.parametrize("name", ("tno", "av_i"))
def test_genuine_kernel_default_hardcode_is_killed(case, name, monkeypatch):
    ref, controls = reference(case, name)
    original = full._parameter_namespace
    def hardcoded(*args, **kwargs):
        env = original(*args, **kwargs)
        env[name.upper()] = getattr(tc, name.upper())
        return env
    monkeypatch.setattr(full, "_parameter_namespace", hardcoded)
    jax.clear_caches()
    mutant = jax.jit(lambda state, controls: full.full_column(state, 18., interpret=True, parameters=controls))
    output = raw_outputs(jax.block_until_ready(mutant(case[2], validate_parameters(controls))))
    with pytest.raises(AssertionError, match="WRF increment gate"):
        assert_wrf_gate(case, ref, output)
    jax.clear_caches()
