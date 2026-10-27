"""The reporting gate must reject nonfinite values and larger deviations."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("thparams_roundoff",Path(__file__).resolve().parents[2]/"proofs/w44_thparams/roundoff.py")
gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def outputs():
    return [np.array([1.],np.float32) for _ in range(15)]+[np.array([1.],np.float64) for _ in range(5)]


@pytest.mark.parametrize("index",range(20))
@pytest.mark.parametrize("side",(0,1))
@pytest.mark.parametrize("value",(np.nan,np.inf,-np.inf))
def test_each_nonfinite_output_and_reference_refused(index,side,value):
    pair=[outputs(),outputs()]
    pair[side][index][0]=value
    with pytest.raises(AssertionError,match="nonfinite"):
        gate.compare(*pair)


def test_measured_class_and_larger_error():
    a,b=outputs(),outputs()
    b[1]=np.array([1.],np.float32).view(np.uint32).__add__(2).view(np.float32)
    b[15]=b[1].astype(np.float64)
    result=gate.compare(a,b)
    assert not result["byte_exact"] and result["within_roundoff"]
    assert result["differences"][1]["max_storage_ulp"]==2**30
    b[1]=np.array([1.],np.float32).view(np.uint32).__add__(3).view(np.float32)
    assert not gate.compare(a,b)["within_roundoff"]


def test_other_double_fields_not_silently_treated_as_REAL():
    a,b=outputs(),outputs()
    b[16]=np.nextafter(b[16],np.inf)
    assert not gate.compare(a,b)["within_roundoff"]


def test_nt_number_class_is_scoped_to_Nr():
    a,b=outputs(),outputs()
    b[7]=np.array([1.],np.float32).view(np.uint32).__add__(3).view(np.float32)
    assert not gate.compare(a,b)["within_roundoff"]
    assert gate.compare(a,b,nt_c=True)["within_roundoff"]
    b[7]=np.array([1.],np.float32).view(np.uint32).__add__(4).view(np.float32)
    assert not gate.compare(a,b,nt_c=True)["within_roundoff"]
