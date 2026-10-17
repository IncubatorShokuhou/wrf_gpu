"""Real Swiss native loader/binding -> three coupled CPU steps with Goddard5/5.

No scheme or solver replacement. Uses the release CPU facade (not a GPU claim),
an immutable source checkout and lane-private namelist/output paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('out',type=Path)
parser.add_argument('--input',type=Path,default=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'))
parser.add_argument('--ra',type=int,choices=(4,5),default=5)
parser.add_argument('--diagnose',action='store_true',help='retain guard details and a FAIL verdict rather than stop at the first nonzero guard')
args=parser.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
args.out.mkdir(parents=True,exist_ok=True)
case=args.out/'case';case.mkdir(exist_ok=True)
for name in ('wrfinput_d01','wrfbdy_d01'):
    target=case/name
    if not target.exists():target.symlink_to(args.input/name)
text=(args.input/'namelist.input').read_text()
for key in ('ra_lw_physics','ra_sw_physics'):
    text,count=re.subn(rf'(?im)^\s*{key}\s*=.*$',f' {key} = {args.ra},',text)
    assert count==1,(key,count)
(case/'namelist.input').write_text(text)

import sys
sys.path.insert(0,str(args.source/'src'))
import jax
import numpy as np
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime.domain_tree import DomainTree,run_operational_domain_tree
from gpuwrf.runtime.operational_mode import _commit_to_operational_device

assert jax.devices()[0].platform=='cpu'
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip(),
        'platform':'cpu','gpu_executed':False,'cache':'NONE','input':str(args.input),
        'namelist_sha256':hashlib.sha256((case/'namelist.input').read_bytes()).hexdigest(),
        'steps':[],'affinity':sorted(os.sched_getaffinity(0))}
config=NestedPipelineConfig(case,args.out/'stream',args.out/'proof',hours=1,max_dom=1)
start=time.perf_counter()
hierarchy,bundles,_meta,_start,_dts,carries=_load_domains(config,('d01',))
record['load_s']=time.perf_counter()-start
nl=bundles['d01'].namelist
assert (nl.ra_lw_physics,nl.ra_sw_physics,nl.mp_physics)==(args.ra,args.ra,8)
if args.ra==5:assert type(carries['d01'].radiation_diagnostics).__name__=='GoddardRadiationDiagnostics'
record['resolved_work']={'ra_lw':args.ra,'ra_sw':args.ra,'mp':8,'dt_s':float(nl.dt_s),
                         'radiation_cadence_steps':int(nl.radiation_cadence_steps),'steps':3}
tree=DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
carries={k:_commit_to_operational_device(v) for k,v in carries.items()}
own={'d01':0}
for i in range(3):
    start=time.perf_counter()
    result=run_operational_domain_tree(tree,root_steps=1,carries=carries,initial_own_steps=own,
                                      block_between=False,root_sync_cadence=0)
    jax.block_until_ready(result.carries)
    elapsed=time.perf_counter()-start
    carries,own=result.carries,dict(result.own_steps)
    c=carries['d01']
    bad=[jax.tree_util.keystr(p) for p,x in jax.tree_util.tree_leaves_with_path(c)
         if np.asarray(x).dtype.kind=='f' and not np.isfinite(np.asarray(x)).all()]
    guards=int(np.asarray(c.census.guards).sum()) if c.census is not None else None
    sw_count=getattr(c.radiation_diagnostics,'sw_heating_negative_count',None)
    sw_negative=None if sw_count is None else int(np.asarray(sw_count))
    from gpuwrf.diagnostics.census import GUARDS,EVENTS,F2_PHASES,F2_FIELDS
    guard_counts=({name:{event:int(value) for event,value in zip(EVENTS,row)}
                   for name,row in zip(GUARDS,np.asarray(c.census.guards)) if np.any(row)}
                  if c.census is not None else None)
    minima=[]
    if c.census is not None:
        for p,f in np.argwhere(np.asarray(c.census.f2_min_negative)<0):
            minima.append({'phase':F2_PHASES[p],'field':F2_FIELDS[f],
                'minimum':float(np.asarray(c.census.f2_min_negative)[p,f]),
                'step':int(np.asarray(c.census.f2_min_step)[p,f]),
                'flat_index':int(np.asarray(c.census.f2_min_index)[p,f])})
    row={'step':i+1,'wall_s':elapsed,'nonfinite_leaves':bad,'guards':guards,
         'guard_counts':guard_counts,'f2_negative_minima':minima,'sw_negative_heating_count':sw_negative,'swdown_max':float(np.max(c.radiation_diagnostics.swdown)),
         'glw_min':float(np.min(c.radiation_diagnostics.glw)),
         'rthraten_max':float(np.max(np.abs(c.rthraten)))}
    record['steps'].append(row)
    (args.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    assert not bad and (args.ra!=5 or sw_negative==0)
    if not args.diagnose:assert guards==0
    assert row['glw_min']>0 and row['rthraten_max']>0
record['verdict']='PASS' if all(row['guards']==0 for row in record['steps']) else 'FAIL'
(args.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
