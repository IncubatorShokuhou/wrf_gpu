"""Admission/key/zero-rate controls; WRF changed-column gates are separate."""
from pathlib import Path
from types import FunctionType
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics.thompson_parameters import validate_parameters
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime_parameters import bind_domain_parameters, restart_parameter_digests
from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash


@pytest.fixture(autouse=True)
def native(monkeypatch):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")


def test_rate_zero_is_valid_real_but_nonzero_underflow_is_not():
    result = validate_parameters({"autoconversion_rate": 0})["autoconversion_rate"]
    assert result.shape == () and result.dtype == jnp.float32 and float(result) == 0.
    for bad in (False, 1e-100):
        with pytest.raises(ValueError):
            validate_parameters({"autoconversion_rate": bad})


def test_threshold_respects_original_inactive_cloud_guard():
    result = validate_parameters({"autoconversion_threshold": 1e-12})["autoconversion_threshold"]
    assert float(result) == float(np.float32(tc.R1))
    for bad in (0., .5e-12):
        with pytest.raises(ValueError):
            validate_parameters({"autoconversion_threshold": bad})


def test_autoconversion_values_share_keys_and_preserve_restart_identity(record_property):
    original = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True, mp_physics=8)
    def bind(values):
        # CPU qualification can exercise the experimental shape without
        # admitting the unqualified family names through the runtime binder.
        return replace(original, thompson_parameters=validate_parameters(values))
    a = bind({"autoconversion_threshold": 1e-5, "autoconversion_rate": .027})
    b = bind({"autoconversion_threshold": 1e-3, "autoconversion_rate": .054})
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(original)
    assert carry_aval_hash((a,), {}) == carry_aval_hash((b,), {})
    assert restart_parameter_digests(a) != restart_parameter_digests(b)
    assert static_config_hash(bind({"autoconversion_rate": .027})) != static_config_hash(bind({"autoconversion_threshold": 1e-5}))
    record_property("autoconversion_static_key", static_config_hash(a))
    record_property("autoconversion_aval_key", carry_aval_hash((a,), {}))


def test_zero_rate_removes_only_auto_terms_on_a_real_cloudy_column():
    path = Path(__file__).resolve().parents[1] / "v025/b_thompson/fixtures/mixed_phase_columns.npz"
    with np.load(path, allow_pickle=False) as fixture:
        index = int(np.argmax(np.max(fixture["in_qc"], axis=1)))
        data = {name: np.asarray(fixture["in_" + name][[index]])
                for name in ("th", "pii", "p", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "dz8w", "w")}
        dt = float(fixture["dt"][index])
    assert all(np.isfinite(value).all() for value in data.values())
    f = lambda name: jnp.asarray(data[name], jnp.float32)
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"),
                                  Ni=f("ni"), Nr=f("nr"), T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p,T,qv), dz=f("dz8w"), w=f("w"))
    original = tc._wrf_warm_rain_rates
    def rates(state, rate):
        fn = FunctionType(original.__code__, {**original.__globals__, "AU_RT": rate},
                          original.__name__, original.__defaults__, original.__closure__)
        fn.__kwdefaults__ = original.__kwdefaults__
        return fn(state, dt, tc.THOMPSON_TABLES, jnp.float32(1.) / jnp.float32(dt))
    run = jax.jit(rates)
    baseline, zero = ([np.asarray(x) for x in run(state, jnp.float32(value))] for value in (.027, 0.))
    assert all(np.isfinite(x).all() for x in (*baseline, *zero))
    assert np.count_nonzero(baseline[0]) > 0 and np.count_nonzero(baseline[1]) > 0
    assert not np.count_nonzero(zero[0]) and not np.count_nonzero(zero[1])
    assert all(a.tobytes() == b.tobytes() for a,b in zip(baseline[2:], zero[2:], strict=True))
