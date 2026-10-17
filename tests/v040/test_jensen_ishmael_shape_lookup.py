"""Pristine observed Jensen calls; no JAX-vs-JAX fidelity reference."""
from pathlib import Path
import hashlib,importlib.util,json

import jax
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'proofs/s2_mpmore/jensen55/stages'
_spec=importlib.util.spec_from_file_location('jensen_shape_libm',DATA.parent/'libm_numpy.py')
_libm=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_libm)
jax.config.update('jax_enable_x64',True)


def load(mode):
    manifest=json.loads((DATA/'manifest.json').read_text())
    row=next(r for r in manifest['rows'] if r['mode']==mode)
    p=DATA/(mode+'.npz')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    assert row['E26_BYTE_columns']==17 and all(v==0 for v in row['nonfinite_values'].values())
    with np.load(p) as data:
        return manifest,{k:data[k].copy() for k in data.files}


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_shape_actual_original_callers_literal_BYTE(mode):
    from gpuwrf.physics.jensen_ishmael.shape import var_check
    manifest,d=load(mode)
    inputs={k:d['shape_inputs'][:,i] for i,k in enumerate(manifest['shape_input_names'])}
    got=var_check(**inputs,gamma_tab=d['gamma_table'],xp=_libm.backend)
    failures=[]
    for name,value in got.items():
        ref=d['shape_outputs'][:,manifest['shape_output_names'].index(name)]
        assert np.isfinite(value).all() and value.dtype==ref.dtype
        if value.tobytes()!=ref.tobytes():
            bad=value!=ref
            failures.append((name,int(np.count_nonzero(bad)),np.flatnonzero(bad)[:8],value[bad][:8],ref[bad][:8]))
    assert not failures,(mode,failures)
    assert np.count_nonzero(got['nidum']!=inputs['nidum'])>0,'Original number-size check must execute'
    assert np.count_nonzero(got['aidum']!=inputs['aidum'])>0,'Original shape update must execute'


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_lookup_actual_table_corners_literal_BYTE(mode):
    from gpuwrf.physics.jensen_ishmael.lookup import interpolate_corners
    _,d=load(mode);v=d['lookup_values'];i=d['lookup_indices'];R=v.dtype
    assert set(i[:,0])=={2,6},'Actual cloud and rain collection tables both required'
    assert np.count_nonzero(v[:,4])>0
    corners=tuple(v[:,5+j] for j in range(16))
    di=v[:,0]-i[:,3].astype(R);dk=v[:,1]-i[:,4].astype(R)
    dii=v[:,2]-i[:,2].astype(R);djj=v[:,3]-i[:,1].astype(R)
    got=interpolate_corners(corners,di,dk,dii,djj)
    ref=v[:,4]
    assert np.isfinite(got).all() and got.dtype==ref.dtype
    assert got.tobytes()==ref.tobytes(),(mode,np.flatnonzero(got!=ref)[:20],got[got!=ref][:20],ref[got!=ref][:20])


def controls(mode):
    folder=DATA.parent/'lookup_controls'
    m=json.loads((folder/'manifest.json').read_text())
    result=[]
    for contraction in ('noFMA','FMA'):
        row=next(r for r in m['rows'] if r['mode']==mode and r['contraction']==contraction)
        p=folder/(mode+'_'+contraction+'.npz')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        assert (row['FMA_opcode_rows']==0)==(contraction=='noFMA')
        with np.load(p) as data:result.append({k:data[k].copy() for k in data.files})
    rows=[r for r in m['rows'] if r['mode']==mode]
    for name in ('lookup_input_sha256','shape_input_sha256'):
        assert len({r[name] for r in rows})==1,'Original contraction controls must have identical raw inputs'
    for k in ('lookup_inputs','lookup_original_coords','shape_inputs'):
        assert result[0][k].tobytes()==result[1][k].tobytes()
    return m,result


def assert_enclosed(value,original):
    low=np.minimum.reduce([a.min(axis=0) for a in original])
    high=np.maximum.reduce([a.max(axis=0) for a in original])
    assert np.isfinite(value).all()
    assert value.dtype==low.dtype
    bad=(value<low)|(value>high)
    assert not bad.any(),(np.flatnonzero(bad)[:12],value[bad][:12],low[bad][:12],high[bad][:12])


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_lookup_compiled_in_original_actual_coordinate_ULP_FMA_enclosure(mode):
    from gpuwrf.physics.jensen_ishmael.lookup import interpolate_corners
    _,observed=load(mode);_,arms=controls(mode);v=observed['lookup_values'];i=observed['lookup_indices']
    R=v.dtype
    coords=arms[0]['lookup_original_coords']
    assert coords[0].tobytes()==v[:,:4].tobytes()
    for n in range(4):
        for sign,direction in enumerate((-np.inf,np.inf)):
            expected=v[:,:4].copy();expected[:,n]=np.nextafter(expected[:,n],R.type(direction))
            assert expected.tobytes()==coords[1+2*n+sign].tobytes()
    indices=i[:,[3,4,2,1]].astype(R)
    delta=coords-indices
    assert (arms[0]['lookup_inputs'][:,:,:4]-R.type(1)).tobytes()==delta.tobytes()
    assert arms[0]['lookup_outputs'][0].tobytes()==v[:,4].tobytes()
    corners=tuple(v[:,5+j] for j in range(16))
    got=np.asarray(jax.jit(interpolate_corners)(corners,*tuple(delta[0,:,j] for j in range(4))))
    assert_enclosed(got,[a['lookup_outputs'] for a in arms])


@pytest.mark.parametrize('mode',['fp32','fp64'])
def test_shape_observed_callers_compiled_BYTE(mode):
    from gpuwrf.physics.jensen_ishmael.shape import var_check
    m,d=load(mode)
    inputs={k:d['shape_inputs'][:,i] for i,k in enumerate(m['shape_input_names'])}
    lowered=jax.jit(lambda fields,table:var_check(**fields,gamma_tab=table)).lower(inputs,d['gamma_table'])
    if mode=='fp32':assert 'f64' not in str(lowered.compiler_ir('stablehlo'))
    got=jax.tree.map(np.asarray,lowered.compile()(inputs,d['gamma_table']))
    for name,value in got.items():
        ref=d['shape_outputs'][:,m['shape_output_names'].index(name)]
        assert np.isfinite(value).all() and value.dtype==ref.dtype
        assert value.tobytes()==ref.tobytes(),(name,np.flatnonzero(value!=ref)[:12])


@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('compiled',[False,True])
def test_original_shape_lower_delta_branch_and_thresholds(mode,compiled):
    from gpuwrf.physics.jensen_ishmael.shape import var_check
    manifest,d=load(mode);_,arms=controls(mode);x=arms[0]['shape_inputs'][0]
    assert np.count_nonzero(x[:,5]<x.dtype.type(.55))==6
    # Compiler constants are exact; only actual physical arguments get a ULP.
    assert np.all(arms[0]['shape_inputs'][:,:,:4]==x[None,:,:4])
    for n in range(4,12):
        for sign,direction in enumerate((-np.inf,np.inf)):
            expected=x.copy();expected[:,n]=np.nextafter(expected[:,n],x.dtype.type(direction))
            assert expected.tobytes()==arms[0]['shape_inputs'][1+2*(n-4)+sign].tobytes()
    inputs={k:x[:,i] for i,k in enumerate(manifest['shape_input_names'])}
    if compiled:
        got=jax.tree.map(np.asarray,jax.jit(lambda fields,table:var_check(**fields,gamma_tab=table))(inputs,d['gamma_table']))
    else:got=var_check(**inputs,gamma_tab=d['gamma_table'],xp=_libm.backend)
    names=manifest['shape_input_names']+['rni','alphstr','alphv','betam']
    for name,value in got.items():
        index=names.index(name);ref=arms[0]['shape_outputs'][0,:,index]
        if compiled:assert_enclosed(value,[a['shape_outputs'][:,:,index] for a in arms])
        else:
            assert np.isfinite(value).all() and value.dtype==ref.dtype
            assert value.tobytes()==ref.tobytes(),(name,np.flatnonzero(value!=ref))
