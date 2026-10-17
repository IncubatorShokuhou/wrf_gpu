"""Fitch E78 through actual public native RestartStore/checkpoint APIs."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib,json,pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime.restart_store import RestartStore,nested_identity
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_v013_operational_smoke import _grid,_namelist,_base_state
from test_u2_windfarm import farm_files


def setup(tmp_path):
    grid=_grid(nz=12,ny=3,nx=4)
    n=replace(_namelist(grid),windfarm_opt=1,windfarm=farm_files(tmp_path))
    carry=initial_operational_carry(_base_state(grid)).replace(windfarm_power=jnp.full((3,4),100.,jnp.float32))
    config=NestedPipelineConfig(tmp_path,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    identity=nested_identity(config,{'d01':SimpleNamespace(namelist=n)},datetime(2026,7,25,tzinfo=timezone.utc))
    return grid,n,carry,identity


def test_public_native_restart_identity_and_power_roundtrip(tmp_path):
    grid,n,c,identity=setup(tmp_path)
    assert 'wind-turbine-1.tbl' in identity['inputs']
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d01':c},{'d01':1},{'d01':n.dt_s},identity)
    saved,_=store.read(generation,expected_identity=identity)
    for a,b in zip(jax.tree.leaves(c),jax.tree.leaves(saved['carries']['d01']),strict=True):np.testing.assert_array_equal(a,b)
    (tmp_path/'wind-turbine-1.tbl').write_text((tmp_path/'wind-turbine-1.tbl').read_text().replace('80 80','81 80'))
    cfg=NestedPipelineConfig(tmp_path,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    changed=nested_identity(cfg,{'d01':SimpleNamespace(namelist=n)},datetime(2026,7,25,tzinfo=timezone.utc))
    with pytest.raises(ValueError,match='identity'):store.read(generation,expected_identity=changed)


@pytest.mark.parametrize('damage',['power','config','domains','config_type','selection','factor'])
def test_native_rehashed_corruption_refused(tmp_path,damage):
    grid,n,c,identity=setup(tmp_path)
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d01':c},{'d01':1},{'d01':n.dt_s},identity)
    snapshot=pickle.loads((generation/'snapshot.pkl').read_bytes())
    manifest=json.loads((generation/'manifest.json').read_text())
    if damage=='power':snapshot['carries']['d01']['extra_fields']['windfarm_power']=None
    elif damage=='config':manifest['identity'].pop('windfarm_config')
    elif damage=='domains':manifest['identity']['windfarm_config']={}
    elif damage=='config_type':manifest['identity']['windfarm_config']['d01']=[]
    elif damage=='selection':manifest['identity']['windfarm_config']['d01']['windfarm_opt']=0
    elif damage=='factor':manifest['identity']['windfarm_config']['d01']['tke_factor']=float('nan')
    data=pickle.dumps(snapshot);(generation/'snapshot.pkl').write_bytes(data)
    manifest.update(payload_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    (generation/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Fitch'):store.read(generation)


def test_checkpoint_runtime_required_and_retained(tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    grid,n,c,_=setup(tmp_path);path=tmp_path/'cp.pkl'
    with pytest.raises(ValueError,match='Fitch'):write_checkpoint(c.state,n,grid,5,path)
    write_checkpoint(c.state,n,grid,5,path,runtime_state=c)
    _,_,_,step,out=read_checkpoint_with_runtime_state(path)
    assert step==5
    np.testing.assert_array_equal(out.windfarm_power,c.windfarm_power)
