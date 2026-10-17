"""Use eight genuine NCAR lowland lake cells instead of inactive Swiss Lake1."""
import json
import os
from pathlib import Path
import numpy as np
from prepare import ROOT, SLOTS, private_executable, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset


assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
base=ROOT/'mmf24_real'
with Dataset(base/'wrfinput_d01') as ds:
    mask=np.asarray(ds['LAKEMASK'][0])>0
    assert int(mask.sum())==8
    assert not np.any(np.asarray(ds['IVGTYP'][0])==int(ds.getncattr('ISICE')))
for ident,options,lake,features in (
    ('c66',{},1,['Lake1-eight-genuine-NCAR-lake-cells']),
    ('c67',dict(opt_snf=5),0,['Noah-snow-partition5-authentic-icefree-lowland']),
):
    if any(r['id']==ident for r in matrix['cases']):continue
    g=parse_namelist(base/'namelist.input')
    g['physics'].update(sf_lake_physics=lake)
    g['physics'].pop('wtddt',None)  # WRF default; no MMF cadence on these rows
    g['noah_mp']=options
    row=dict(id=ident,geography='authentic-NCAR-lowland',hours=24,domains=['d01'],
        options={k:g['physics'][k] for k in SLOTS},extra={},features=features,
        active_lake_cells=int(mask.sum()) if lake else 0)
    row=stage(row,base,g);private_executable(ROOT/ident,ROOT/'reference_zero')
    row['reference_semantics']='compiler-defined zero, inactive canonical Morrison-aero object'
    write_json(ROOT/ident/'inputs.json',row)
    matrix['cases'].append(row)
    write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'case':ident,'errors':row['admission_errors']}),flush=True)
