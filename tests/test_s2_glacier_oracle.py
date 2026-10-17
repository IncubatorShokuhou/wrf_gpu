"""Independent pristine REAL4 glacier call: all options and active regimes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics.noahmp.glacier import glacier_column

PROOF=Path(__file__).resolve().parents[1]/'proofs/s2_glacier'
MANIFEST=json.loads((PROOF/'oracle_manifest.json').read_text())
DATA=np.load(PROOF/'glacier_oracle.npz')
OPTIONS=sorted(set(map(tuple,DATA['options'])))
spec=importlib.util.spec_from_file_location('glacier_literal_debugger',PROOF/'check_column.py')
debug=importlib.util.module_from_spec(spec);spec.loader.exec_module(debug)


def operands(selected):
    matrix=DATA['inputs'][selected].T
    values={}
    for field in MANIFEST['fields']:
        if field['intent']=='OUT':continue
        a=matrix[field['offset']:field['offset']+field['extent']]
        dtype=np.int32 if field['integer'] else np.float32
        values[field['name']]=list(a.astype(dtype)) if field['extent']>1 else a[0].astype(dtype)
    return values


def options_dict(options):
    return dict(zip(MANIFEST['options'],map(int,options)))


def defined(field,inputs):
    mask=np.ones((field['extent'],len(inputs['tg'])),bool)
    if field['name']=='hcpct':mask[:3]=np.arange(3)[:,None]>=inputs['isnow'][None,:]+3
    return mask


def test_fixture_is_bound_to_unchanged_pristine_source():
    assert hashlib.sha256((PROOF/'glacier_oracle.npz').read_bytes()).hexdigest()==MANIFEST['fixture_sha256']
    assert DATA['reference'].dtype==DATA['inputs'].dtype==np.float32
    assert len(DATA['inputs'])==MANIFEST['case_count']==4896
    assert np.isfinite(DATA['reference']).all() and np.isfinite(DATA['inputs']).all()
    assert MANIFEST['source_sha256']=='a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece'


@pytest.mark.parametrize('options',OPTIONS)
def test_literal_binary32_is_byte_exact_to_complete_pristine_call(options):
    selected=np.flatnonzero(np.all(DATA['options']==options,axis=1))
    # WRF's oracle is scalar. NumPy's vector **.25 selects a different SIMD
    # approximation; use the same scalar literal debugger that generated G09.
    for index in selected:
        inputs={}
        for field in MANIFEST['fields']:
            if field['intent']=='OUT':continue
            dtype=np.int32 if field['integer'] else np.float32
            a=DATA['inputs'][index,field['offset']:field['offset']+field['extent']].astype(dtype)
            inputs[field['name']]=list(a) if field['extent']>1 else a[0]
        actual=glacier_column(inputs,**options_dict(options),xp=debug.Literal())
        for field in MANIFEST['fields']:
            if field['intent']=='IN':continue
            value=np.asarray(actual[field['name']],np.float32).reshape(-1)
            reference=DATA['reference'][index,field['offset']:field['offset']+field['extent']]
            mask=np.ones(field['extent'],bool)
            if field['name']=='hcpct':mask[:int(inputs['isnow'])+3]=False
            assert np.isfinite(value[mask]).all(),field['name']
            np.testing.assert_array_equal(value.view(np.uint32)[mask],reference.view(np.uint32)[mask],err_msg=field['name'])


@pytest.mark.parametrize('gla',(1,2))
def test_compiled_column_is_inside_independent_pristine_input_ulp_envelope(gla):
    options=(2,1,2,1,gla)
    selected=np.flatnonzero(np.all(DATA['options']==options,axis=1))[:29]
    inputs=operands(selected)
    result=jax.device_get(jax.jit(lambda v:glacier_column(v,**options_dict(options)))(jax.tree.map(jnp.asarray,inputs)))
    envelope=np.load(PROOF/'input_ulp.npz');part=slice((gla-1)*29,gla*29)
    for field in MANIFEST['fields']:
        if field['intent']=='IN':continue
        key=field['name'];offset=field['offset'];extent=field['extent']
        low=envelope['lower'][part,offset:offset+extent].T
        high=envelope['upper'][part,offset:offset+extent].T
        low=np.nextafter(low,np.float32(-np.inf));high=np.nextafter(high,np.float32(np.inf))
        value=np.asarray(result[key],np.float32).reshape(extent,-1);mask=defined(field,inputs)
        assert np.isfinite(value[mask]).all(),key
        assert np.all(value[mask]>=low[mask]) and np.all(value[mask]<=high[mask]),key


def test_active_regimes_are_present_in_pristine_outputs():
    options=(2,1,2,1,1);selected=np.flatnonzero(np.all(DATA['options']==options,axis=1))
    rows={str(DATA['names'][i]):i for i in selected}
    offset={f['name']:f['offset'] for f in MANIFEST['fields']}
    assert DATA['reference'][rows['melt'],offset['qmelt']]>0
    assert DATA['reference'][rows['accumulation'],offset['qsnow']]>0
    assert DATA['reference'][rows['drain'],offset['qsnbot']]>0
    assert DATA['reference'][rows['refreeze'],offset['snice']+2]>DATA['inputs'][rows['refreeze'],offset['snice']+2]
    assert DATA['reference'][rows['one_split'],offset['isnow']]<-1
    assert DATA['reference'][rows['three_remove_mid'],offset['isnow']]>-3
