"""Add admitted Jensen55's authentic Swiss24h original reference on DATA."""
from pathlib import Path
import json
import os
from prepare import ROOT, RETAINED, SLOTS, private_executable, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.scheme_catalog import _IMPLEMENTED

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source=Path(__import__('sys').argv[1]);assert 55 in _IMPLEMENTED['mp_physics']
matrix=json.loads((ROOT/'MATRIX.json').read_text());assert not any(r['id']=='c87' for r in matrix['cases'])
base=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms/c87');(base/'cpu').mkdir(parents=True)
case=ROOT/'c87';case.mkdir();(case/'cpu').symlink_to(base/'cpu',target_is_directory=True)
groups=parse_namelist(source/'examples/switzerland_d01/namelist.input');groups['physics']['mp_physics']=55
row=dict(id='c87',geography='Swiss',hours=24,domains=['d01'],extra={},
    options={k:groups['physics'][k] for k in SLOTS},features=['primary-MP55-Jensen-admitted-batch6'])
row=stage(row,RETAINED/'source_inputs/Swiss',groups)
assert not row['admission_errors'],row['admission_errors']
private_executable(case,ROOT/'reference_zero')
row['reference_semantics']='Unchanged original400bodies, compiler-defined zero; inactive canonical Morrison-aero object'
write_json(case/'inputs.json',row);matrix['cases'].append(row);write_json(ROOT/'MATRIX.json',matrix)
print(json.dumps(row,indent=2))
