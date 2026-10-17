"""Existing P3 producer oracle and held-output transcription, no radar recompute."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.physics.p3 import adapter as a
from gpuwrf.diagnostics.online import configuration, radar_from_context, validate_options
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_m7_netcdf_writer import synthetic_case, writer_authority
from test_v0110_wrfrst_netcdf import _state
from gpuwrf.validation.tier2 import make_ideal_grid

ROOT=Path(__file__).resolve().parents[1]


def test_existing_mp52_ze_against_original_wrf_three_steps():
    """Existing P360 compiled policy, retaining the strict 3e-4 dBZ residuals."""
    from gpuwrf.physics.p3.column import step
    fixture=ROOT/'proofs/s2_p3/fixtures/columns_mp52.npz'
    manifest=json.loads((fixture.parent.parent/'columns_manifest.json').read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==manifest['files'][fixture.name]['sha256']
    with np.load(fixture) as f:data={k:f[k] for k in f.files}
    # Preserve P348/P360's registered original-WRF contraction spread. Never
    # replace the strict residual with a claim of exact numerical agreement.
    spread_root=ROOT/'proofs/t_writer_p3_radar/P348_reference'
    reference=json.loads((spread_root/'manifest.json').read_text())
    accepted=json.loads((ROOT/'proofs/s2_p3/qualification.json').read_text())
    assert reference['files']['report.json']==accepted['gates']['WRF_contraction_init_control']['sha256']
    for name,sha in reference['files'].items():
        assert hashlib.sha256((spread_root/name).read_bytes()).hexdigest()==sha,name
    with np.load(spread_root/'literal_mp52.npz') as f:literal=f['ze']
    with np.load(spread_root/'fma_mp52.npz') as f:contracted=f['ze']
    np.testing.assert_array_equal(literal,data['ze'])
    limit=max(3e-4,float(np.max(np.abs(contracted[...,1:].astype('f8')-literal[...,1:]))))
    assert reference['strict_atol_dBZ']==3e-4
    assert limit==reference['registered_WRF_contraction_max_dBZ']
    fields={k:v[...,0] for k,v in data.items() if v.ndim==3 and k!='qzi'}
    for it in (1,2,3):
        out=step(fields,18.,it,mp=52)
        actual=np.asarray(out['ze'])
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,data['ze'][...,it],rtol=0,atol=limit)
        fields={k:out.get(k,v) for k,v in fields.items()}


@pytest.mark.parametrize('mp',(50,51,52,53))
def test_p3_unconditional_radar_configuration_and_direct_time_level(mp):
    nl=SimpleNamespace(mp_physics=mp,do_radar_ref=0,nwp_diagnostics=0,run_physics=True)
    assert configuration(nl)['do_radar_ref']==1
    validate_options(nl,specified=True,nested=False,periodic=False)
    old=jnp.full((3,5,6),123.,jnp.float32)
    fresh=jnp.arange(old.size,dtype=jnp.float32).reshape(old.shape)/8-99
    got=radar_from_context(old,fresh,nl,7,1) # Ordinary step, not a history alarm.
    np.testing.assert_array_equal(got[:,1:-1,1:-1],fresh[:,1:-1,1:-1])
    np.testing.assert_array_equal(got[:,:,0],old[:,:,0])


@pytest.mark.parametrize('native',(False,True))
def test_actual_adapter_retains_existing_ze_without_conversion(monkeypatch,native):
    from gpuwrf.backends import pallas as backend
    from gpuwrf.kernels import phys_p3
    grid=make_ideal_grid(4,3,5)
    state=_state(grid,mp_physics=52)
    expected=np.arange(state.qv.size,dtype=np.float32).reshape(state.qv.shape)/4-99
    captured=[]
    def solve(fields,dt,it,**kw):
        if native:assert 'ze' in kw['output_names']
        captured.append(int(it))
        out={name:jnp.zeros_like(fields['qv']) for name in a.output_mapping(52)}
        out.update(th=fields['th'],qv=fields['qv'],rainncv=jnp.zeros(fields['qv'].shape[:-1],jnp.float32),snowncv=jnp.zeros(fields['qv'].shape[:-1],jnp.float32),
                   ze=jnp.moveaxis(jnp.asarray(expected),0,-1))
        return out
    monkeypatch.setattr(backend,'cuda_native',lambda:native)
    monkeypatch.setattr(a,'step',solve);monkeypatch.setattr(phys_p3,'full_column',solve)
    nl=op.OperationalNamelist(grid=grid,metrics=grid.metrics,tendencies=None,mp_physics=52)
    updated,precip,context=op._apply_post_rk_microphysics(state,nl,return_precipitation=True,
        return_radar_context=True,step_index=jnp.int32(7))
    assert captured==[7]
    assert np.asarray(context).tobytes()==expected.tobytes()


@pytest.mark.parametrize('series',('raw','zero'))
def test_c23_cpu_all_frames_held_publication_byte(tmp_path,series):
    """IO transcription only: actual original CPU outputs as held producer values."""
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    root=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c23/cpu' if series=='raw' else
              '<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP32/52a')
    files=sorted(root.glob('wrfout_d01_*'));assert len(files)==25
    state,grid,nl=synthetic_case(nx=42,ny=42,nz=44)
    nl.update(mp_physics=52)
    authority=writer_authority(grid,'d01')
    previous=jnp.zeros(state.qv.shape,jnp.float32)
    cfg=SimpleNamespace(mp_physics=52,do_radar_ref=0,nwp_diagnostics=0)
    active=False
    for it,path in enumerate(files):
        with Dataset(path) as ds:
            expected=np.asarray(ds['REFL_10CM'][0]);assert 'REFD_MAX' not in ds.variables
        if it==0:assert np.asarray(previous).tobytes()==expected.tobytes()
        held=radar_from_context(previous,jnp.asarray(expected),cfg,it*200,1) if it else previous
        assert np.asarray(held).tobytes()==expected.tobytes()
        active |= bool(np.any(expected>0))
        payload=prepare_wrfout_payload(state,grid,nl,tmp_path/f'{series}_{it}',domain='d01',
            domain_authority=authority,valid_time=path.name.removeprefix('wrfout_d01_').replace('_','T'),
            lead_hours=float(it),run_start='2023-01-15T00:00:00',variable_subset=('REFL_10CM',),
            diagnostics={'REFL_10CM':held},full_variable_set=False)
        write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,
                              variable_subset=('REFL_10CM',))
        with Dataset(payload.target) as ds:assert np.asarray(ds['REFL_10CM'][0]).tobytes()==expected.tobytes()
        previous=held
    assert active


def test_nested_writer_p3_held_radar_has_no_fabricated_refd_max(tmp_path):
    from test_initial_history_wrf_t0 import _writer, _wrfinput
    writer,_=_writer(tmp_path,lambda *args,**kw:{})
    state,grid,nl=synthetic_case(nx=4,ny=3,nz=5)
    nl=dict(nl,physics={'mp_physics':52,'do_radar_ref':0},time_control={'nwp_diagnostics':0})
    writer.bundles['d01']=SimpleNamespace(namelist=nl,grid=grid)
    writer.domain_authorities['d01']=writer_authority(grid,'d01')
    _wrfinput(tmp_path/'wrfinput_d01',3,4)
    radar=np.arange(state.qv.size,dtype=np.float32).reshape(state.qv.shape)/4-99
    result=writer('d01',200,SimpleNamespace(state=state,nwp_diagnostics=None,radar_reflectivity=radar))
    with Dataset(result['wrfout']) as ds:
        assert np.asarray(ds['REFL_10CM'][0]).tobytes()==radar.tobytes()
        assert 'REFD_MAX' not in ds.variables
