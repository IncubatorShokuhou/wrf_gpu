"""Three selectors on ONE original Madrid WPS/real geography; no map fabrication."""
import json
import os
from pathlib import Path
from prepare import ROOT,SLOTS,sha,write_json,private_executable
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
base=ROOT/'madrid_maps_real'
receipt=json.loads((ROOT/'VP27-NCAR-Madrid-maps.json').read_text())
assert len(receipt['stages'])==3 and all(r['rc']==0 for r in receipt['stages'])
assert all(receipt['map_census'][key]['nonzero'] for key in ('SOILCL1','SOILCL2','SOILCL3','SOILCL4','SOILCOMP','IRFRACT'))
assert receipt['map_census']['CROPTYPE']['nonzero']==0
for ident,soil,irrig in (('c71',2,1),('c72',3,2),('c73',3,3)):
    if any(r['id']==ident for r in matrix['cases']):continue
    g=parse_namelist(base/'namelist.input')
    g['noah_mp']=dict(opt_soil=soil,opt_irr=irrig,opt_irrm=irrig)
    if ident=='c73':g['noah_mp']['opt_snf']=5
    row=dict(id=ident,geography='Madrid-official-NCAR-soil+IRFRACT',hours=24,domains=['d01'],
        options={k:g['physics'][k] for k in SLOTS},extra={},
        features=[f'Noah-soil{soil}',f'explicit-irrigation-trigger{irrig}-method{irrig}']+(['icefree-snow5'] if ident=='c73' else []),
        static_map_receipt=str(ROOT/'VP27-NCAR-Madrid-maps.json'),
        scope='Original Madrid maps; crop/tile/mapped-method qualification unavailable, never synthesized')
    row=stage(row,base,g);private_executable(ROOT/ident,ROOT/'reference_zero')
    row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
    matrix['cases'].append(row)
    write_json(ROOT/ident/'inputs.json',row);write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'id':ident,'errors':row['admission_errors']}),flush=True)
old=next(r for r in matrix['cases'] if r['id']=='c67')
old.update(gpu_eligible=False,control_reason='Snow5 packed into actual Madrid c73; retained separate lowland CPU control')
write_json(ROOT/'c67/inputs.json',old)
write_json(ROOT/'MATRIX.json',matrix)
