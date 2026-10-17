import sys
sys.path.insert(0,'scripts/v025')
import cpu_guard
import pytest
import jax.numpy as jnp
from gpuwrf.physics import flg_transfer

original=flg_transfer.infrared_transfer
def mutant(*a,**k):
    return tuple(jnp.zeros_like(x) for x in original(*a,**k))
flg_transfer.infrared_transfer=mutant
sys.exit(pytest.main([
    'tests/v040/radiation/test_s2_flg_column.py::test_infrared_transport_compiled[7]',
    'tests/v040/radiation/test_s2_flg_column.py::test_transport_compiled_regression_is_finite_and_unchanged[7]',
    '-q','-rxX','--basetemp','<USER_HOME>/wrf_gpu2_lanes/rv-s2-flggfdl/repair-mutant-pytest',
    '--junitxml=<USER_HOME>/wrf_gpu2_lanes/rv-s2-flggfdl/repair_mutant.xml',
]))
