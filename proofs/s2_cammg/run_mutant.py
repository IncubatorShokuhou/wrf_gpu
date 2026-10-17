"""Deliberately remove MG's total ground precipitation; original gate must fail."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'proofs/s2_cammg')]
# Same guard ordering as tests/conftest.py, before any model/JAX import.
import cpu_guard
from gpuwrf.physics.cammg import micro
original=micro.mmicro_pcond
index=micro.OUTPUTS['mmicro_pcond'].index('prect')


def without_precipitation(*args,**kwargs):
    values=list(original(*args,**kwargs))
    values[index]=values[index]*0
    return tuple(values)


micro.mmicro_pcond=without_precipitation
import pytest
status=pytest.main([str(ROOT/'tests/v040/test_cammg.py'),'-k','original_public_byte','-q',
    '--basetemp','<DATA_ROOT>/wrf_gpu2/scratch/s2-cammg/mutant_pytest',
    '-p','proofs.s2_cammg.pytest_receipts'])
assert status==1,f'physical mutant must produce genuine test failures, got {status}'
