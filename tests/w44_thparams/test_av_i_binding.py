"""AV_I admission/key checks only; original WRF physics evidence is AI02."""
import numpy as np
from dataclasses import replace
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash
from gpuwrf.runtime_parameters import bind_domain_parameters, restart_parameter_digests
from gpuwrf.physics.thompson_parameters import validate_parameters


def test_experimental_av_i_is_a_real_operand_with_value_independent_keys(monkeypatch, record_property):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_" + suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    original = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True, mp_physics=8)
    with pytest.raises(ValueError, match="experimental and refused"):
        bind_domain_parameters(original, {"d01": {"thompson_parameters": {"av_i": 1493.9}}}, "d01")
    def bind(name, value):
        # Low-level qualification only; the public binder remains closed.
        return replace(original, thompson_parameters=validate_parameters({name: value}))
    default, changed = bind("av_i", 1493.9), bind("av_i", 2987.8)
    a, b = (np.asarray(n.thompson_parameters["av_i"]) for n in (default, changed))
    assert a.shape == b.shape == () and a.dtype == b.dtype == np.float32
    assert float(a) == 1493.9000244140625 and float(b) == 2987.800048828125
    static_key = static_config_hash(default)
    aval_key = carry_aval_hash((default,), {})
    assert static_key == static_config_hash(changed) != static_config_hash(original)
    assert aval_key == carry_aval_hash((changed,), {})
    assert static_key != static_config_hash(bind("av_c", 31694600.)), "same-count names must not collide"
    assert restart_parameter_digests(default) != restart_parameter_digests(changed)
    record_property("av_i_static_key", static_key)
    record_property("av_i_aval_key", aval_key)
