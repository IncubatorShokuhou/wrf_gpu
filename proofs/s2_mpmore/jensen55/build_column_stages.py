"""Print-only complete original ME boundaries, E26 against pristine public caller."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and not artifact.exists()
artifact.mkdir()
spec=importlib.util.spec_from_file_location('original_jensen',tree/'proofs/s2_mpmore/jensen55/build_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
pristine=original.SOURCE.read_text()
vectors='pres_e dzmic theta qv qc qr nr effc1d effi1d dbz1d'.split()
matrices='qi ni ai ci icetype1d vmi1d di1d rhopo1d phii1d'.split()
scratch_vectors='vtrm vtrn rhoair i_rhoair'.split()
scratch_matrices='vtrmi1 vtrni1 vtrzi1'.split()
scratch_categories='rhobar deltastr ani cni rni'.split()
args=['kts','kte','cat','it','dt',*vectors,*matrices,*scratch_vectors,*scratch_matrices,*scratch_categories,'qrpre1d','qipre1d','sedi']
def record(stage):return '\n call s2_me_record('+str(stage)+','+','.join(args)+')\n'
source=pristine
anchors=[('    gammnu   = gamma(NU)',0,True),('    nstep = 1',1,True),
         ('  !.. Check after sedimentation',2,False),('  end subroutine me_ishmael',3,True)]
for anchor,stage,before in anchors:
    assert source.count(anchor)==1
    source=source.replace(anchor,(record(stage)+anchor) if before else (anchor+record(stage)))
# Stage0 does not read uninitialized local scratch: only the defined public
# inputs and diagnostics are emitted. Compiler init remains the standing zero.
observer=f'''
subroutine s2_me_record(stage,{','.join(args)})
integer,intent(in) :: stage,kts,kte,cat,it
real,intent(in) :: dt,qrpre1d,qipre1d(cat)
real,intent(in) :: {','.join(n+'(kts:kte)' for n in vectors+scratch_vectors)}
real,intent(in) :: {','.join(n+'(cat,kts:kte)' for n in matrices+scratch_matrices)}
real,intent(in) :: {','.join(n+'(cat)' for n in scratch_categories)}
logical,intent(in) :: sedi
integer :: unit,sedi_int
open(newunit=unit,file='me_stages.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) stage,kte-kts+1,cat,it
write(unit) dt,qrpre1d,qipre1d
write(unit) {','.join(vectors+matrices)}
if(stage.gt.0) then
write(unit) {','.join(scratch_vectors+scratch_matrices)}
write(unit) {','.join(scratch_categories)}
sedi_int=0
if(sedi) sedi_int=1
write(unit) sedi_int
endif
close(unit)
end subroutine
'''
source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:observer+m[0],source);assert count==1
path=artifact/'observed.F';path.write_text(source);original.SOURCE=path
original.main(tree,artifact/'capture')
out=artifact/'fixtures';out.mkdir();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    for case in range(1,18):
        assert (tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json').read_bytes()==(artifact/f'capture/fixtures/{mode}_{case}.json').read_bytes()
    raw=(artifact/f'capture/build/{mode}/me_stages.bin').read_bytes();offset=0;dtype=np.dtype(R).newbyteorder('>');case=0
    while offset<len(raw):
        stage,nz,cat,it=map(int,np.frombuffer(raw,'>i4',count=4,offset=offset));offset+=16
        assert stage in (0,1,2,3) and cat==3
        if stage==0:case+=1
        scalar=np.frombuffer(raw,dtype,count=2+cat,offset=offset).astype(R);offset+=(2+cat)*dtype.itemsize
        fields=dict(dt=scalar[0],qrpre1d=scalar[1],qipre1d=scalar[2:],it=np.int32(it))
        for name in vectors+matrices+(scratch_vectors+scratch_matrices if stage>0 else []):
            shape=(nz,cat) if name in matrices+scratch_matrices else (nz,)
            count=int(np.prod(shape));fields[name]=np.frombuffer(raw,dtype,count=count,offset=offset).astype(R).reshape(shape);offset+=count*dtype.itemsize
        if stage>0:
            for name in scratch_categories:
                fields['held_'+name]=np.frombuffer(raw,dtype,count=cat,offset=offset).astype(R);offset+=cat*dtype.itemsize
            fields['sedi']=np.frombuffer(raw,'>i4',count=1,offset=offset).astype(np.int32)[0];offset+=4
        assert all(np.isfinite(a).all() for a in fields.values())
        prior=tree/f'proofs/s2_mpmore/jensen55/column_stages/{mode}_{case}_{stage}.npz'
        if prior.is_file():
            with np.load(prior) as old:
                assert all(fields[name].tobytes()==old[name].tobytes() for name in old.files),'All prior original stage fields must remain BYTE'
        path=out/f'{mode}_{case}_{stage}.npz';np.savez_compressed(path,**fields)
        rows.append(dict(mode=mode,case=case,stage=stage,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),nonfinite=0))
    assert offset==len(raw) and case==17
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
    observed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows,E26_BYTE_columns=34,prior_stage_fields_BYTE=True,
    stages={'0':'originalMEentry','1':'beforeadaptive sedimentation','2':'after sedimentation/before final size checks','3':'completeoriginalME'},
    scope='Print-only caller/stage capture; original full public34 columns/scalarsBYTE'),indent=2)+'\n')
print(dict(records=len(rows),E26_BYTE_columns=34,nonfinite=0),flush=True)
