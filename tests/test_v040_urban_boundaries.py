"""Critic E78/E200 controls: actual checkpoint boundaries and corrupted truth."""
from pathlib import Path
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROOF=Path(__file__).resolve().parents[1]/'proofs/v040/urban_bep_bem'
sys.path.insert(0,str(PROOF))


@pytest.fixture
def urban_case(monkeypatch):
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re,'_MP_RE',False)
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    def build(option,empty=False,scalar_angles=False):
        from gpuwrf.config.paths import wrf_run_dir
        from gpuwrf.validation.tier2 import make_ideal_grid
        from gpuwrf.coupling import urban_multilayer_hook as hook
        from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
        from gpuwrf.runtime.operational_mode import OperationalNamelist
        from gpuwrf.runtime.operational_state import initial_operational_carry
        from test_v0110_wrfrst_netcdf import _state
        grid=make_ideal_grid(nz=8,ny=3,nx=4)
        ivg=np.full((3,4),14)
        if not empty:ivg[1,1]=13
        bundle=hook.load_multilayer_bundle(wrf_run_dir(),option=option,ivgtyp=ivg,isurban=13)
        urban=hook.initial_multilayer_carry(bundle,tslb=jnp.full((4,3,4),290,jnp.float32),
            frc_urb2d=jnp.zeros((3,4),jnp.float32),nz=8,smois=jnp.full((4,3,4),.3,jnp.float32))
        zero=jnp.zeros((3,4),jnp.float32)
        angle=jnp.asarray(0,jnp.float32) if scalar_angles else zero
        rad=RRTMGRadiationDiagnostics(**{name:zero for name in RRTMGRadiationDiagnostics._fields
            if name not in RRTMGRadiationDiagnostics._field_defaults},urban_radiation=(zero,zero,angle,angle))
        carry=initial_operational_carry(_state(grid)).replace(urban_bep=urban,radiation_diagnostics=rad)
        nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,sf_urban_physics=option,
            urban_bundle=bundle,bl_pbl_physics=1,sf_sfclay_physics=1)
        return grid,nl,carry
    return build


def damage_cases(carry):
    urban=carry.urban_bep
    for name in urban.state._fields:
        yield 'missing_state_'+name,carry.replace(urban_bep=urban._replace(state=urban.state._replace(**{name:None})))
    for name in urban.sources._fields:
        yield 'missing_source_'+name,carry.replace(urban_bep=urban._replace(sources=urban.sources._replace(**{name:None})))
    for name in urban.surface_layer._fields:
        yield 'missing_surface_'+name,carry.replace(urban_bep=urban._replace(surface_layer=urban.surface_layer._replace(**{name:None})))
    yield 'missing_sources',carry.replace(urban_bep=urban._replace(sources=None))
    yield 'missing_fraction',carry.replace(urban_bep=urban._replace(fraction=None))
    for name,value in [('empty_urban_axis',urban.state.roof[:,:,:0,:]),
                       ('empty_material_axis',urban.state.roof[...,:0]),
                       ('wrong_street_axis',urban.state.roof[:,:1])]:
        yield name,carry.replace(urban_bep=urban._replace(state=urban.state._replace(roof=value)))
    yield 'source_level_mismatch',carry.replace(urban_bep=urban._replace(sources=urban.sources._replace(a_v=urban.sources.a_v[:,:-1])))
    yield 'surface_interface_mismatch',carry.replace(urban_bep=urban._replace(sources=urban.sources._replace(surface=urban.sources.surface[:,:-1])))
    yield 'fraction_columns_mismatch',carry.replace(urban_bep=urban._replace(fraction=np.zeros(2,np.float32)))
    yield 'held_surface_grid_mismatch',carry.replace(urban_bep=urban._replace(surface_layer=urban.surface_layer._replace(br=np.zeros((3,3),np.float32))))
    if hasattr(urban.state,'extra'):
        yield 'nested_extra_none',carry.replace(urban_bep=urban._replace(state=urban.state._replace(extra={'missing':None})))
    rad=carry.radiation_diagnostics
    for bad in (np.nan,np.inf,-np.inf):
        for index in range(4):
            held=list(rad.urban_radiation);value=np.asarray(held[index]).copy();value.flat[0]=bad;held[index]=value
            yield f'radiation_{index}_{bad}',carry.replace(radiation_diagnostics=rad._replace(urban_radiation=tuple(held)))
    for index,value in [(0,np.float32(0)),(1,np.zeros((3,3),np.float32)),
                         (2,np.zeros(3,np.float32)),(3,np.zeros((3,4),np.float64))]:
        held=list(rad.urban_radiation);held[index]=value
        yield f'radiation_shape_dtype_{index}',carry.replace(radiation_diagnostics=rad._replace(urban_radiation=tuple(held)))


@pytest.mark.parametrize('option',[2,3])
@pytest.mark.parametrize('fmt',['pickle','netcdf'])
def test_checkpoint_rejects_every_missing_field_shape_and_nonfinite_on_write_and_read(tmp_path,monkeypatch,urban_case,option,fmt):
    from gpuwrf.io import restart_schema as schema
    from test_restart_schema_held_physics import exact,save,read
    grid,nl,carry=urban_case(option)
    good=tmp_path/'healthy';badpath=tmp_path/'malformed'
    save(good,fmt,grid,nl,carry);exact(carry,read(good,fmt));good_bytes=good.read_bytes()
    for name,bad in damage_cases(carry):
        with pytest.raises(ValueError,match='BEP/BEM'):
            save(good,fmt,grid,nl,bad)
        assert good.read_bytes()==good_bytes,name
        # Manufacture a valid serialization of a malformed payload, bypassing
        # only write-time admission. Read-time checks are restored before read.
        with monkeypatch.context() as bypass:
            bypass.setattr(schema,'validate_urban_bep',lambda *a,**k:None)
            bypass.setattr(schema,'validate_urban_radiation',lambda *a,**k:None)
            save(badpath,fmt,grid,nl,bad)
        with pytest.raises(ValueError,match='BEP/BEM'):
            read(badpath,fmt)


@pytest.mark.parametrize('option',[2,3])
@pytest.mark.parametrize('fmt',['pickle','netcdf'])
@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('scalar_angles',[False,True])
def test_healthy_and_zero_urban_domains_roundtrip_exact(tmp_path,urban_case,option,fmt,empty,scalar_angles):
    from test_restart_schema_held_physics import exact,save,read
    grid,nl,carry=urban_case(option,empty,scalar_angles)
    path=tmp_path/'healthy'
    save(path,fmt,grid,nl,carry);exact(carry,read(path,fmt))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('family',['pbl','bep_driver','bem_driver'])
def test_one_nonfinite_original_output_is_rejected_before_physics(tmp_path,monkeypatch,family,bad):
    import importlib
    gate=importlib.import_module(family+'_parity')
    def forbidden_jit(*args,**kwargs):pytest.fail('corrupted reference reached physics/JIT')
    monkeypatch.setattr(gate.jax,'jit',forbidden_jit)
    if family=='pbl':
        original=Path.read_text
        def read_text(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name=='ysu_fp64_1.json':
                data=json.loads(text);data['scalars']['PBLH']=bad;return json.dumps(data)
            return text
        monkeypatch.setattr(Path,'read_text',read_text)
        run=lambda:gate.run(PROOF/'savepoints/pbl',tmp_path/'out.json',schemes=('ysu',),precisions=('fp64',))
    else:
        original=gate.np.load
        target='urban2_fp64_ac_on.npz' if family=='bep_driver' else 'urban3_fp64_ac_on.npz'
        def load(path,*args,**kwargs):
            data=original(path,*args,**kwargs)
            if Path(path).name==target:
                data=dict(data);data['HFX']=data['HFX'].copy();data['HFX'][0,0]=bad
            return data
        monkeypatch.setattr(gate.np,'load',load)
        run=(lambda:gate.run(PROOF/'savepoints',tmp_path/'out.json')) if family=='bep_driver' else (
            lambda:gate.run(PROOF/'savepoints',tmp_path/'out.json',('fp64',),1))
    with pytest.raises(AssertionError,match='nonfinite.*reference'):run()


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('operand',['actual','reference','raw_limit','extra'])
def test_comparison_rejects_every_nonfinite_operand(operand,bad):
    from conditioned_gate import comparison
    values=dict(actual=np.array([1.]),reference=np.array([1.]),raw_limit=np.array([.01]),extra=np.array([0.]))
    values[operand][0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):comparison(**values)


def test_comparison_preserves_raw_failures_exact_kpbl_and_rejects_overflow():
    from conditioned_gate import comparison
    actual=np.array([0.,.1,2.]);reference=np.array([0.,0.,2.]);raw=np.array([.01]*3);extra=np.array([0.,.2,0.])
    row=comparison(actual,reference,raw,extra)
    assert row==dict(max_abs=.1,max_ratio=float(np.max(abs(actual-reference)/(raw+extra))),raw_violations=1,violations=0)
    assert comparison(np.array([6]),np.array([6]),np.array([0.]),exact=True)['violations']==0
    assert comparison(np.array([7]),np.array([6]),np.array([0.]),exact=True)['violations']==1
    with np.errstate(over='ignore'),pytest.raises(AssertionError,match='nonfinite final limit'):
        comparison(np.array([1.]),np.array([1.]),np.array([np.finfo(float).max]),np.array([np.finfo(float).max]))
