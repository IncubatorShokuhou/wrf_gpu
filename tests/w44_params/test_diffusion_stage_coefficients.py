"""Default scalar-folding identity and live stage operands (CPU, research only)."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf import runtime_parameters as rp
from gpuwrf.contracts import real_constfold as rc
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.kernels.dyn_acoustic_fp32 import _dampmag
from gpuwrf.runtime.operational_mode import OperationalNamelist


def base(dt):
    grid = GridSpec.canary_3km_template()
    grid = replace(grid, bc=replace(grid.bc, source='wrf'))
    return OperationalNamelist(grid, None, None, dt, run_boundary=True,
                              damp_opt=3, dampcoef=.2, acoustic_substeps=4)


@pytest.mark.parametrize('dt', [54.,18.])
@pytest.mark.parametrize('real_fold', [False,True])
def test_bound_stage_strengths_are_default_exact_and_value_live(monkeypatch, dt, real_fold):
    # Deliberately local bypass: no product admission or global test policy change.
    monkeypatch.setenv('GPUWRF_DYN_FP32','1')
    monkeypatch.setattr(rp,'UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES',frozenset())
    monkeypatch.setattr(rc,'ENABLED',real_fold)
    original=base(dt)
    bound=[rp.bind_diffusion_parameters(original,{'dampcoef':x}) for x in (.2,.4,0.)]
    from gpuwrf.runtime.aot_cheap_key import static_config_hash,carry_aval_hash
    assert len({static_config_hash(nl) for nl in bound}) == 1
    assert len({carry_aval_hash((nl,),{}) for nl in bound}) == 1
    traces=[]
    def evaluate(nl):
        traces.append(1)
        return jnp.stack([_dampmag(dts,rp.diffusion_value(nl,'dampcoef'),
                         precomputed=nl.diffusion_parameters['dampmag_rk'][i])
                         for i,dts in enumerate((dt/3.,dt/4.))])
    call=jax.jit(evaluate)
    results=[np.asarray(call(nl)) for nl in bound]
    assert traces == [1], 'changed values retraced the program'
    for value,result in zip((.2,.4,0.),results):
        coefficient = float(original.dampcoef) if value == .2 else np.float32(value)
        expected=np.asarray([rc.real_scalar_product(dts,coefficient) for dts in (dt/3.,dt/4.)],np.float32)
        np.testing.assert_array_equal(result.view(np.uint32),expected.view(np.uint32))
    assert np.any(results[0] != results[1])
    assert np.all(results[2] == 0), 'zero damping must stay exactly zero'
    # A raw traced multiplication is the semantic deletion mutant. It must be
    # distinguished in the legacy dt54 and dt18 second-stage rounding cases.
    if not real_fold:
        raw=np.asarray(jax.jit(lambda c:jnp.array([dt/3.,dt/4.],jnp.float32)*c)(jnp.float32(.2)))
        assert not np.array_equal(raw.view(np.uint32),results[0].view(np.uint32))


def test_bound_stage_strengths_ignore_sub_real_input_bits(monkeypatch):
    monkeypatch.setenv('GPUWRF_DYN_FP32','1')
    monkeypatch.setattr(rp,'UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES',frozenset())
    nl=base(54.)
    for value in (.2,.4):
        variants=[value,np.float32(value),np.nextafter(float(np.float32(value)),np.inf)]
        selected=[rp.bind_diffusion_parameters(nl,{'dampcoef':raw}) for raw in variants]
        assert len({rp.restart_parameter_digests(x)['diffusion_parameters'] for x in selected}) == 1
        for other in selected[1:]:
            for key in selected[0].diffusion_parameters:
                np.testing.assert_array_equal(np.asarray(selected[0].diffusion_parameters[key]).view(np.uint32),
                                              np.asarray(other.diffusion_parameters[key]).view(np.uint32))
