"""Pack late admitted land controls before their original references start."""
import json
import os
from pathlib import Path
from prepare import ROOT, render, sha, write_json
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.namelist_binding import namelist_honour_report

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
matrix=json.loads((ROOT/'MATRIX.json').read_text())
for ident in ('c60','c61'):
    case=ROOT/ident
    assert not (case/'cpu_run.json').exists() and not list((case/'cpu').glob('wrfout*'))
    row=next(r for r in matrix['cases'] if r['id']==ident)
    groups=parse_namelist(case/'input/namelist.input')
    if ident=='c60':
        groups['physics']['sst_skin']=1
        feature='Noah-SSTskin1-on-active-Atlantic-water'
    else:
        groups['noah_mp']=dict(dveg=6,opt_run=7)
        feature='Noah-dveg6+Xinanjiang7'
    (case/'input/namelist.input').write_text(render(groups))
    errors,warnings=namelist_honour_report(groups,('d01',),replay=False)
    assert not errors, errors
    row.update(admission_errors=errors,admission_warnings=warnings,status='ADMITTED')
    if feature not in row['features']:row['features'].append(feature)
    row['input_sha256']={p.name:sha(p) for p in (case/'input').iterdir() if p.is_file()}
    write_json(case/'inputs.json',row)
    print(json.dumps({'case':ident,'errors':errors,'added':feature}),flush=True)
row=next(r for r in matrix['cases'] if r['id']=='c44')
row.update(gpu_eligible=False,control_reason='LAKEMASK and LAKE_DEPTH all-zero; no actual lake cells; superseded by authentic-lowland c66')
write_json(ROOT/'c44/inputs.json',row)
write_json(ROOT/'MATRIX.json',matrix)
