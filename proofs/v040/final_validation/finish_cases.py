"""Complete admitted carbon/VIC selectors and reuse corrected WRF FDDA metadata."""
import json
import os
from pathlib import Path
import sys
from prepare import ROOT,RETAINED,SLOTS,private_executable,sha,write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source=Path(sys.argv[1]);matrix=json.loads((ROOT/'MATRIX.json').read_text())
swiss=RETAINED/'source_inputs/Swiss'


def add(ident,groups,features,links=None):
    if any(r['id']==ident for r in matrix['cases']):return
    row=dict(id=ident,geography='Swiss',hours=24,domains=['d01'],
        options={k:groups['physics'][k] for k in SLOTS},extra={},features=features)
    row=stage(row,swiss,groups);private_executable(ROOT/ident,ROOT/'reference_zero')
    for name,p in (links or {}).items():
        for directory in ('input','cpu'):(ROOT/ident/directory/name).symlink_to(p.resolve())
    row['input_sha256']={p.name:sha(p) for p in (ROOT/ident/'input').iterdir() if p.is_file()}
    row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
    write_json(ROOT/ident/'inputs.json',row)
    matrix['cases'].append(row);write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'id':ident,'errors':row['admission_errors']}),flush=True)


for ident,dveg,inf in (('c68',7,1),('c69',8,2),('c70',9,3)):
    g=parse_namelist(source/'examples/switzerland_d01/namelist.input')
    g['noah_mp']=dict(dveg=dveg,opt_run=8,opt_infdv=inf)
    add(ident,g,[f'Noah-carbon-DVEG{dveg}',f'dynamicVIC8-infiltration{inf}-compiler-zero'])
analysis=Path('<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF02/wrffdda_d01')
with Dataset(analysis) as ds:
    assert 'WRF V4.' in ds.TITLE and len(ds.dimensions['Time'])==24
    assert all(v.getncattr('FieldType')==104 for k,v in ds.variables.items() if k!='Times')
for ident,oldid in (('c77','c52'),('c78','c53'),('c79','c54')):
    old=next(r for r in matrix['cases'] if r['id']==oldid)
    g=parse_namelist(ROOT/oldid/'input/namelist.input')
    links={'wrffdda_d01':analysis}
    sf=ROOT/oldid/'input/wrfsfdda_d01'
    if sf.exists():links['wrfsfdda_d01']=sf
    add(ident,g,old['features']+['corrected-original-WRF-TITLE+FieldType104-ingestion'],links)
    old.update(gpu_eligible=False,control_reason='Original WRF refused old analysis TITLE; superseded by '+ident)
    write_json(ROOT/oldid/'inputs.json',old)
write_json(ROOT/'MATRIX.json',matrix)
