"""Localize whole-process rounding using original per-level E26 checkpoints."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and not artifact.exists()
artifact.mkdir()
spec=importlib.util.spec_from_file_location('original_jensen',tree/'proofs/s2_mpmore/jensen55/build_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
spec=importlib.util.spec_from_file_location('jensen_levels',tree/'proofs/s2_mpmore/jensen55/port_levels.py')
port=importlib.util.module_from_spec(spec);spec.loader.exec_module(port)
pristine=original.SOURCE.read_text()
checkpoints=port.CHECKPOINTS
point='theta qv qc qr nr rhoair i_rhoair'.split()
scalar='temp i_temp qvs qvi sup sui svpi svpl xxls xxlv xxlf mu dv kt nsch npr gammnu i_gammnu i_dt i_cp fourthirdspi pcc prc nprc nprc1 pre npre pra npra nragg mbiggr nbiggr lamr n0rr'.split()
point_cat='qi ni ai ci vtrmi1 vtrni1 vtrzi1'.split()
local_cat=('ani cni rni deltastr rhobar rimesum rimesumr rimetotal prd nrd ard crd prdr ardr crdr qmlt nmlt amlt cmlt '
           'totalnuc totalnucn qagg nagg dQRfzri dQIfzri dNfzri dQImltri dNmltri nibnuc qmult nmult fv fh').lower().split()
scalar_names=point+scalar;cat_names=point_cat+local_cat
scalars=','.join([n+'(k)' for n in point]+scalar)
categories=','.join([n+'(:,k)' for n in point_cat]+local_cat)
record=lambda stage:f' call s2_process_record({stage},k,[{scalars}],[{categories}])\n'
source=''
for n,line in enumerate(pristine.splitlines(keepends=True),1):
    if n in checkpoints:source+=record(checkpoints[n])
    source+=line
source=source.replace('  !.. Inherent growth ratio data','  integer :: s2_current_case=0\n\n  !.. Inherent growth ratio data',1)
observer=f'''
subroutine s2_case(value)
integer,intent(in) :: value
s2_current_case=value
end subroutine
subroutine s2_process_record(stage,k,scalars,categories)
integer,intent(in) :: stage,k
real,intent(in) :: scalars({len(scalar_names)}),categories({3*len(cat_names)})
integer :: unit
open(newunit=unit,file='process_checkpoints.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) s2_current_case,stage,k
write(unit) scalars,categories
close(unit)
end subroutine
'''
source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:observer+m[0],source);assert count==1
path=artifact/'observed.F';path.write_text(source);original.SOURCE=path
driver=original.caller().replace('only: jensen_ishmael_init,mp_jensen_ishmael','only: jensen_ishmael_init,mp_jensen_ishmael,s2_case')
driver=driver.replace(' call mp_jensen_ishmael',' call s2_case(case_index)\n call mp_jensen_ishmael')
original.caller=lambda:driver
original.main(tree,artifact/'capture')
out=artifact/'fixtures';out.mkdir();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    for case in range(1,18):
        assert (tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json').read_bytes()==(artifact/f'capture/fixtures/{mode}_{case}.json').read_bytes()
    raw=(artifact/f'capture/build/{mode}/process_checkpoints.bin').read_bytes();offset=0;ids=[];values=[];dtype=np.dtype(R).newbyteorder('>')
    count=len(scalar_names)+3*len(cat_names)
    while offset<len(raw):
        ids.append(np.frombuffer(raw,'>i4',count=3,offset=offset).astype(np.int32));offset+=12
        values.append(np.frombuffer(raw,dtype,count=count,offset=offset).astype(R));offset+=count*dtype.itemsize
    assert offset==len(raw)
    ids=np.stack(ids);values=np.stack(values);assert np.isfinite(values).all()
    path=out/(mode+'.npz');np.savez_compressed(path,indices=ids,values=values)
    rows.append(dict(mode=mode,records=len(ids),nonfinite=0,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
    observed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),
    scalar_names=scalar_names,category_names=cat_names,source_lines=checkpoints,E26_BYTE_columns=34,rows=rows,
    scope='Original per-level checkpoints; unused held scratch is diagnostic only;wholepublic34columns/scalarsBYTE'),indent=2)+'\n')
print(dict(source_lines=checkpoints,rows=rows),flush=True)
