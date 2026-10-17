"""Real Swiss native pipeline, three CPU steps and an exact resumed fourth."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import time

source,out,gla=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
assert os.environ['JAX_PLATFORMS']=='cpu'
out.mkdir(parents=True,exist_ok=False);case=out/'case';case.mkdir()
reference=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
for name in ('wrfinput_d01','wrfbdy_d01'):(case/name).symlink_to(reference/name)
text=(reference/'namelist.input').read_text()
line=f' opt_gla = {gla},'
if re.search(r'(?im)^\s*opt_gla\s*=',text):text=re.sub(r'(?im)^\s*opt_gla\s*=.*$',line,text)
elif re.search(r'(?i)&noah_mp\b',text):text=re.sub(r'(?i)(&noah_mp\b)',r'\1\n'+line,text,count=1)
else:text+='\n&noah_mp\n'+line+'\n/\n'
(case/'namelist.input').write_text(text)
sys.path.insert(0,str(source/'src'))
import jax
import numpy as np
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime.domain_tree import DomainTree,run_operational_domain_tree
from gpuwrf.runtime.operational_mode import _commit_to_operational_device
from gpuwrf.io.restart import write_restart,read_restart

assert jax.devices()[0].platform=='cpu'
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
receipt=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
             platform='cpu',gpu_executed=False,opt_gla=gla,cache='NONE',steps=[],
             inputs={n:hashlib.sha256((case/n).read_bytes()).hexdigest() for n in ('wrfinput_d01','wrfbdy_d01','namelist.input')})
cfg=NestedPipelineConfig(case,out/'stream',out/'proof',hours=1,max_dom=1)
start=time.perf_counter();hierarchy,bundles,_meta,_start,_dt,carries=_load_domains(cfg,('d01',))
receipt['load_s']=time.perf_counter()-start
nl=bundles['d01'].namelist
assert nl.noahmp_static.glacier.opt_gla==gla and len(nl.noahmp_static.glacier.indices)==22
assert carries['d01'].noahmp_land.glacier_qsnow is not None
receipt['resolved_work']=dict(shape=[nl.grid.nz,nl.grid.ny,nl.grid.nx],dt=float(nl.dt_s),mp=nl.mp_physics,
    ra_lw=nl.ra_lw_physics,ra_sw=nl.ra_sw_physics,bl=nl.bl_pbl_physics,glacier_columns=22,steps=4)
tree=DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
carries={k:_commit_to_operational_device(v) for k,v in carries.items()};own={'d01':0}

def advance(tree,carries,own):
    return run_operational_domain_tree(tree,root_steps=1,carries=carries,initial_own_steps=own,
        block_between=False,root_sync_cadence=0)

for i in range(3):
    start=time.perf_counter();result=advance(tree,carries,own);jax.block_until_ready(result.carries)
    carries,own=result.carries,dict(result.own_steps);c=carries['d01']
    bad=[jax.tree_util.keystr(p) for p,a in jax.tree_util.tree_leaves_with_path(c)
         if np.asarray(a).dtype.kind=='f' and not np.isfinite(np.asarray(a)).all()]
    guards=None if c.census is None else int(np.asarray(c.census.guards).sum())
    ice=np.asarray(nl.noahmp_static.glacier.indices)
    # Driver outputs, not host-writer sentinel rendering: removing the actual
    # operational glacier dispatch must fail this coupled gate.
    assert np.all(np.asarray(c.noahmp_land.tv).reshape(-1)[ice]==np.float32(-1.e36))
    assert np.all(np.asarray(c.noahmp_land.smois).reshape(4,-1)[:,ice]==np.float32(1.))
    receipt['steps'].append(dict(step=i+1,wall_s=time.perf_counter()-start,nonfinite=bad,guards=guards,
        snow_mass_min=float(np.min(np.asarray(c.noahmp_land.sneqv))),glacier_qsnow_dtype=str(c.noahmp_land.glacier_qsnow.dtype)))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not bad and guards in (0,None),receipt['steps'][-1]

path=out/'restart.pkl';write_restart(carries['d01'],nl,nl.grid,3,path)
restored,rnl,rgrid,step=read_restart(path)
assert step==3 and rnl.noahmp_static.glacier==nl.noahmp_static.glacier
r_bundle=replace(bundles['d01'],namelist=rnl)
r_tree=DomainTree.from_domains(hierarchy,{'d01':r_bundle},feedback_enabled=False)
control=advance(tree,carries,own);resumed=advance(r_tree,{'d01':restored},own)
jax.block_until_ready((control.carries,resumed.carries))
a=jax.tree.leaves(jax.device_get(control.carries['d01']));b=jax.tree.leaves(jax.device_get(resumed.carries['d01']))
assert len(a)==len(b)
different=[i for i,(x,y) in enumerate(zip(a,b,strict=True)) if (x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes())]
receipt.update(resume_leaf_count=len(a),resume_byte_mismatches=different,land_bytes=int(carries['d01'].noahmp_land.bytes()))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert not different,different
print(json.dumps(receipt,indent=2))
