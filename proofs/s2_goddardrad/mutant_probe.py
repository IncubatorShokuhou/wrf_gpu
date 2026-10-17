"""Remove the approved taux=0 site by using taux=1; require a real oracle assertion.

The immutable production source is untouched. The in-process mutant replaces
only the explicit reduction_tau argument at the existing solver boundary.
"""
import json
import os
from pathlib import Path
import sys

source,out=map(Path,sys.argv[1:3])
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(source/'scripts/v025'))
import cpu_guard  # Must precede JAX; pytest's shared conftest verifies ordering.
sys.path[:0]=[str(source/'src'),str(source/'tests')]
import jax.numpy as jnp
import pytest
from gpuwrf.physics import ra_goddard_sw

original=ra_goddard_sw.solve_sw_column


def wrong_tau(*args):
    args=(*args[:-1],jnp.ones_like(args[-1]))
    return original(*args)


class Capture:
    def __init__(self):self.assertions=[];self.setup_errors=[]
    def pytest_runtest_makereport(self,item,call):
        if call.excinfo is not None:
            if call.when=='call' and call.excinfo.errisinstance(AssertionError):
                self.assertions.append({'node':item.nodeid,'message':str(call.excinfo.value)[:1200]})
            else:self.setup_errors.append({'node':item.nodeid,'phase':call.when,'error':str(call.excinfo.value)[:500]})


capture=Capture();ra_goddard_sw.solve_sw_column=wrong_tau
try:
    rc=pytest.main([str(source/'tests/test_goddard_sw_parity.py')+'::test_native_swrad_pristine[2]',
        '-q','--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/pytest'],plugins=[capture])
finally:
    ra_goddard_sw.solve_sw_column=original
killed=rc==1 and bool(capture.assertions) and not capture.setup_errors
result={'platform':'cpu','gpu_executed':False,'cache':'NONE','pytest_returncode':int(rc),
        'mutant_killed_by_assertion':killed,'assertions':capture.assertions,'setup_errors':capture.setup_errors}
out.write_text(json.dumps(result,indent=2)+'\n')
print('MUTANT_KILLED' if killed else 'MUTANT_GATE_FAILED',flush=True)
raise SystemExit(0 if killed else 1)
