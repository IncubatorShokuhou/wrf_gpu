"""Byte gate for public REAL4 wrapper outputs with CPU FMA/algsimp disabled.

Separate from the normal compiled tolerance gates; the explicit DOUBLE solvers
retain their pristine-WRF native-output comparisons. No GPU execution claim.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

source,out=map(Path,sys.argv[1:3])
assert os.environ['JAX_PLATFORMS']=='cpu'
fma_path=Path(sys.argv[3]) if len(sys.argv)>3 else None
if fma_path is None:
    assert '--xla_cpu_max_isa=AVX' in os.environ.get('XLA_FLAGS','')
    assert '--xla_disable_hlo_passes=algsimp' in os.environ.get('XLA_FLAGS','')
sys.path[:0]=[str(source/'src'),str(source/'tests')]
records=[];current=[''];group_index={}
original=np.testing.assert_allclose
fma={} if fma_path is None else dict(np.load(fma_path))


def check(actual,reference,*args,**kwargs):
    a=np.asarray(actual);e=np.asarray(reference,dtype=a.dtype)
    assert a.dtype==np.float32 and a.shape==e.shape
    assert np.isfinite(a).all() and np.isfinite(e).all()
    bits_a=a.view(np.uint32);bits_e=e.view(np.uint32)
    def ordered(bits):
        b=bits.astype(np.int64)
        return np.where((b&0x80000000)!=0,0x80000000-(b&0x7fffffff),0x80000000+b)
    record={'node':current[0],'values':int(a.size),
        'unequal_bits':int(np.count_nonzero(bits_a!=bits_e)),
        'max_abs':float(np.max(np.abs(a.astype(np.float64)-e.astype(np.float64)))),
        'max_ulp':int(np.max(np.abs(ordered(bits_a)-ordered(bits_e))))}
    if fma:
        case=int(current[0].rsplit('[',1)[1].rstrip(']'))
        g=group_index.get(case,0);group_index[case]=g+1
        sw=fma[f'{case}_swflux'];lw=fma[f'{case}_lwflux']
        groups=(fma[f'{case}_lw'],fma[f'{case}_sw'],
                np.array((sw[2],-sw[3],lw[2],-lw[3],sw[0],-sw[1],lw[0],-lw[1]),np.float32),
                fma[f'{case}_swextra'])
        spread=np.abs(groups[g].astype(np.float64)-e.astype(np.float64))
        excess=np.maximum(np.abs(a.astype(np.float64)-e.astype(np.float64))-spread,0)
        record.update(group=('lw_rate','sw_rate','flux','native_history')[g],
                      wrf_fma_spread_max=float(spread.max()),
                      compiled_excess_max=float(excess.max()),compiled_excess_values=int(np.count_nonzero(excess)))
    records.append(record)
    return original(actual,reference,*args,**kwargs)


class Context:
    def pytest_runtest_setup(self,item):current[0]=item.nodeid


np.testing.assert_allclose=check
try:
    rc=pytest.main([str(source/'tests/test_goddard_wrapper_parity.py'),'-q',
        '--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/pytest'],plugins=[Context()])
finally:
    np.testing.assert_allclose=original
byte=bool(records) and all(x['unequal_bits']==0 for x in records)
result={'platform':'cpu','gpu_executed':False,'cache':'NONE','pytest_returncode':int(rc),
    'literal_real4_bytes_equal':byte,'compared_values':sum(x['values'] for x in records),
    'flags':os.environ.get('XLA_FLAGS',''),'wrf_fma_reference':str(fma_path) if fma_path else None,
    'comparisons':records}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}),flush=True)
raise SystemExit(0 if rc==0 and (byte or fma_path is not None) else 1)
