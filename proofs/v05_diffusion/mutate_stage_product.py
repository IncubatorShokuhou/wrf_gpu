"""Run the numeric stage-product gate with the precomputed operand deleted."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root/'scripts/v025'), str(root/'src')]
import cpu_guard  # Must precede JAX, including an in-process pytest invocation.
import pytest
from gpuwrf.kernels import dyn_acoustic_fp32 as native
from gpuwrf.contracts.real_constfold import real_scalar_product

# Patch before importing the test's by-name alias (E228).
native._dampmag = lambda dts, coef, *, precomputed=None: real_scalar_product(dts, coef)
rc = pytest.main(['-q', str(root/'tests/w44_params/test_diffusion_stage_coefficients.py') +
                 '::test_bound_stage_strengths_are_default_exact_and_value_live[False-54.0]',
                 '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/v05-diffusion/pytest-mutant'])
assert rc == 1, f'expected numeric assertion failure, got {rc}'
print('EXPECTED_NUMERIC_MUTANT_FAILURE; check log has array AssertionError, zero setup errors')
