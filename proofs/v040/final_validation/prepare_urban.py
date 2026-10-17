"""Original real.exe BEP/BEM inputs, preserving mismatched earlier controls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from prepare import ROOT, RETAINED, PRISTINE, private_executable, render, sha, write_json
from extra_cases import stage
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
base=RETAINED/'urban_madrid/input'
for old_id,new_id,scheme,pbl in (('c34','c56',2,1),('c35','c57',3,8)):
    if any(r['id']==new_id for r in matrix['cases']):continue
    old=next(r for r in matrix['cases'] if r['id']==old_id)
    groups=parse_namelist(ROOT/old_id/'input/namelist.input')
    work=ROOT/f'urban{scheme}_real';work.mkdir()
    (work/'namelist.input').write_text(render(groups))
    for p in base.glob('met_em.*'):(work/p.name).symlink_to(p.resolve())
    for p in (PRISTINE/'run').iterdir():
        if p.is_file() and p.name not in ('namelist.input','wrf.exe','real.exe','ndown.exe','tc.exe'):
            (work/p.name).symlink_to(p.resolve())
    with (work/'real.log').open('w') as log:
        r=subprocess.run(['timeout','300',str(PRISTINE/'main/real.exe')],cwd=work,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0 and 'SUCCESS COMPLETE REAL' in (work/'real.log').read_text(errors='replace')
    with Dataset(work/'wrfinput_d01') as ds:
        assert int(ds.SF_URBAN_PHYSICS)==scheme
        urban=int((ds['LU_INDEX'][:]==13).sum())
        assert urban>0
    row=dict(old,id=new_id,status='ADMITTED',gpu_eligible=True,
             features=[f'BEP/BEM{scheme}-authentic-original-real','actual-urban-geography'])
    row.pop('cpu_reference',None)
    row=stage(row,work,groups)
    private_executable(ROOT/new_id,ROOT/'reference_zero')
    row['reference_semantics']='compiler-defined zero; inactive canonical Morrison-aero object'
    old.update(gpu_eligible=False,status='INPUT_HEADER_REFUSAL_CONTROL',superseded_by=new_id)
    matrix['cases'].append(row)
    write_json(ROOT/old_id/'inputs.json',old);write_json(ROOT/new_id/'inputs.json',row)
    write_json(work/'receipt.json',dict(rc=r.returncode,real_executable_sha256=sha(PRISTINE/'main/real.exe'),
        input_sha256={n:sha(work/n) for n in ('wrfinput_d01','wrfbdy_d01','namelist.input')},
        source_met_em={p.name:sha(p) for p in base.glob('met_em.*')},
        actual_sf_urban_physics=scheme,active_urban_cells=urban,
        provenance='original real.exe from untouched authentic Madrid WPS outputs; no header fabrication'))
    write_json(ROOT/'MATRIX.json',matrix)
    print(json.dumps({'case':new_id,'scheme':scheme,'urban_cells':urban,'errors':row['admission_errors']}),flush=True)
