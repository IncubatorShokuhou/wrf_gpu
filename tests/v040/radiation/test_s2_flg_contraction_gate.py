from pathlib import Path
import importlib.util

import numpy as np
import pytest

path=Path(__file__).resolve().parents[3]/'proofs/s2_flggfdl/contraction_gate.py'
spec=importlib.util.spec_from_file_location('flg_contraction_gate',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('field',range(3))
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_rejects_nonfinite_reference_and_candidate(field,bad):
    arrays=[np.ones((2,3,2),np.float32) for _ in range(3)]
    arrays[field][0,0,0]=bad
    with pytest.raises(ValueError,match='Non-finite'):
        module.evaluate(*arrays)


def test_spread_is_source_only_and_sensitive_to_candidate():
    ref=np.ones((2,3,2),np.float32);fma=ref.copy();fma[0,2,0]+=.001
    candidate=ref.copy();candidate[0,0,0]+=.0009
    assert module.evaluate(candidate,ref,fma)['pass_']
    candidate[1,0,0]+=.0009  # other row has no WRF spread
    assert not module.evaluate(candidate,ref,fma)['pass_']
    with pytest.raises(ValueError,match='geometry'):
        module.evaluate(candidate[:1],ref,fma)
