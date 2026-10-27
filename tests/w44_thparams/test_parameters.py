"""Host admission must reject values that cannot be valid REAL operands."""
import numpy as np
import pytest

from gpuwrf.physics.thompson_parameters import PARAMETERS, COEFFICIENT_BOUNDS, validate_parameters


@pytest.fixture(autouse=True)
def native(monkeypatch):
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_"+suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")


@pytest.mark.parametrize("name,bad", [(name,bad) for name in PARAMETERS
    for bad in (float("nan"),float("inf"),float("-inf"),0.,-1.,1e100,1e-100,True,"5",[5])
    if not (name=="autoconversion_rate" and isinstance(bad,(int,float)) and bad==0.)])
def test_reject_invalid(name,bad):
    with pytest.raises(ValueError):
        validate_parameters({name:bad})


@pytest.mark.parametrize("name", ["ccn_conc","typo"])
def test_refuse_unimplemented_or_wrong_physics(name):
    with pytest.raises(ValueError):
        validate_parameters({name:1e8})


def test_sorted_real_operands():
    result=validate_parameters({k:v[0] for k,v in PARAMETERS.items()})
    assert tuple(result)==tuple(sorted(PARAMETERS))
    assert all(a.shape==() and a.dtype==np.float32 for a in result.values())


def test_unsupported_native_configuration(monkeypatch):
    monkeypatch.setenv("GPUWRF_THOMPSON_FULL_COLUMN","0")
    with pytest.raises(ValueError,match="FULL_COLUMN"):
        validate_parameters({"tno":5})


@pytest.mark.parametrize("value", [1., 2.e9])
def test_nt_c_admission_avoids_original_integer_overflow_and_number_cap(value):
    with pytest.raises(ValueError,match="2..1999e6"):
        validate_parameters({"nt_c":value})


@pytest.mark.parametrize("name", tuple(COEFFICIENT_BOUNDS))
def test_bounded_coefficient_operands_reject_unbounded_substep_inputs(name):
    lower, upper = map(np.float32, COEFFICIENT_BOUNDS[name])
    for value in (lower, np.float32(PARAMETERS[name][0]), upper):
        assert float(validate_parameters({name: value})[name]) == float(value)
    for value in (np.nextafter(lower, np.float32(-np.inf)),
                  np.nextafter(upper, np.float32(np.inf)), np.float32(1e9)):
        with pytest.raises(ValueError, match="0.5..2 times WRF default"):
            validate_parameters({name: value})
