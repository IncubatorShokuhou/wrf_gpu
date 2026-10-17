from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from proofs.s1_km5 import upstream_oracle as ora


@pytest.mark.parametrize('key',ora.CASES)
def test_sms_upstream_pristine(key):
    with np.load(ROOT/'data/fixtures/sms3dtke-upstream-wrf-v1.npz') as z:
        names,got,expected=ora.execute(z,key)
    for n,a,r in zip(names,got,expected):
        a=np.asarray(a)
        assert np.isfinite(a).all(),(key,n)
        assert np.isfinite(r).all(),(key,n)
        np.testing.assert_allclose(a,r,rtol=32*np.finfo(np.float32).eps,
            atol=32*np.finfo(np.float32).eps*abs(r).max(),err_msg=key+'/'+n)
