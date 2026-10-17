"""Explicit WRF-defined default depth on eight genuine lake cells."""
import json
import os
from pathlib import Path
from prepare import ROOT,SLOTS,private_executable,write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
assert not any(r['id']=='c83' for r in matrix['cases'])
g=parse_namelist(ROOT/'c66/input/namelist.input')
g['physics'].update(use_lakedepth=0,lakedepth_default=50.)
row=dict(id='c83',geography='authentic-NCAR-lowland',hours=24,domains=['d01'],
    options={k:g['physics'][k] for k in SLOTS},extra={},active_lake_cells=8,
    features=['Lake1-eight-genuine-NCAR-lake-cells','explicit-WRF-default-depth50m'],
    scope='Declared WRF default depth mode, NOT observed bathymetry; original geog values unchanged')
row=stage(row,ROOT/'mmf24_real',g);private_executable(ROOT/'c83',ROOT/'reference_zero')
matrix['cases'].append(row)
old=next(r for r in matrix['cases'] if r['id']=='c66')
old.update(gpu_eligible=False,control_reason='Original WRF requires lake depth or explicit default mode; superseded by c83 use_lakedepth0/default50m')
write_json(ROOT/'c66/inputs.json',old);write_json(ROOT/'MATRIX.json',matrix)
print(json.dumps({'id':'c83','errors':row['admission_errors']}))
