"""Observe original var_check/lookup calls and initialized tables; full E26 gate."""
from pathlib import Path
import hashlib,importlib.util,itertools,json,re,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file()
if artifact.exists():raise SystemExit('Use a fresh artifact directory')
artifact.mkdir()
spec=importlib.util.spec_from_file_location('original_jensen_columns',tree/'proofs/s2_mpmore/jensen55/build_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
source=original.SOURCE.read_text();pristine=source
start=source.index('  subroutine var_check(');end=source.index('  end subroutine var_check',start)
body=source[start:end]
before='nu,ao,fourthirdspi,gammnu,qidum,dsdum,ani,cni,rbdum,nidum,aidum,cidum'
after='nu,ao,fourthirdspi,gammnu,qidum,dsdum,ani,cni,rni,rbdum,nidum,aidum,cidum,alphstr,alphv,betam'
assert body.count('  !.. Deltastr check')==1 and body.count('    return')==1
body=body.replace('  !.. Deltastr check',f'    call s2_shape_record(12,[{before}])\n  !.. Deltastr check')
body=body.replace('    return',f'    call s2_shape_record(16,[{after}])\n    return')
source=source[:start]+body+source[end:]
anchor='    proc=tmp1+(dum5-real(dumjj))*(tmp2-tmp1)'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'\n    call s2_lookup_record(itabdum,dumjj,dumii,dumi,dumk,index,dum1,dum2,dum4,dum5,proc)')
corners=','.join(f'tab(jj+{j},ii+{ii},i+{i},k+{k},component)'
                 for j,ii,i,k in itertools.product(range(2),repeat=4))
observers=f'''
subroutine s2_shape_record(n,values)
 integer,intent(in) :: n
 real,intent(in) :: values(n)
 integer :: unit
 open(newunit=unit,file='shape_calls.bin',access='stream',form='unformatted',status='unknown',position='append')
 write(unit) n
 write(unit) values
 close(unit)
end subroutine
subroutine s2_lookup_record(tab,jj,ii,i,k,component,d1,d2,d4,d5,proc)
 real,intent(in) :: tab(:,:,:,:,:),d1,d2,d4,d5,proc
 integer,intent(in) :: jj,ii,i,k,component
 integer :: unit
 open(newunit=unit,file='lookup_calls.bin',access='stream',form='unformatted',status='unknown',position='append')
 write(unit) size(tab,5),jj,ii,i,k,component
 write(unit) d1,d2,d4,d5,proc,{corners}
 close(unit)
end subroutine
subroutine s2_dump_tables()
 integer :: unit
 open(newunit=unit,file='tables.bin',access='stream',form='unformatted',status='replace')
 write(unit) gamma_tab,coltab,coltabn
 close(unit)
end subroutine
'''
source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:observers+m[0],source)
assert count==1
observed=artifact/'observed.F';observed.write_text(source);original.SOURCE=observed
caller=original.caller()
caller=caller.replace('only: jensen_ishmael_init,mp_jensen_ishmael',
                      'only: jensen_ishmael_init,mp_jensen_ishmael,s2_dump_tables')
caller=caller.replace(' call jensen_ishmael_init()',' call jensen_ishmael_init()\n call s2_dump_tables()')
original.caller=lambda:caller
original.main(tree,artifact/'capture')
out=artifact/'fixtures';out.mkdir();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    b=artifact/'capture/build'/mode
    for case in range(1,18):
        a=tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json'
        observed=artifact/f'capture/fixtures/{mode}_{case}.json'
        assert a.read_bytes()==observed.read_bytes(),(mode,case,'E26 full columns/scalars/input JSON BYTE')
    dtype=np.dtype(R).newbyteorder('>')
    raw=(b/'shape_calls.bin').read_bytes();offset=0;incoming=[];outgoing=[]
    while offset<len(raw):
        count=int(np.frombuffer(raw,'>i4',count=1,offset=offset)[0]);offset+=4
        assert count in (12,16)
        values=np.frombuffer(raw,dtype,count=count,offset=offset).astype(R);offset+=count*dtype.itemsize
        (incoming if count==12 else outgoing).append(values)
    assert offset==len(raw) and len(incoming)==len(outgoing)>0
    incoming,outgoing=np.stack(incoming),np.stack(outgoing)
    raw=(b/'lookup_calls.bin').read_bytes();offset=0;indices=[];values=[]
    while offset<len(raw):
        indices.append(np.frombuffer(raw,'>i4',count=6,offset=offset).astype(np.int32));offset+=24
        values.append(np.frombuffer(raw,dtype,count=21,offset=offset).astype(R));offset+=21*dtype.itemsize
    assert offset==len(raw) and len(indices)>0
    indices,values=np.stack(indices),np.stack(values)
    tables=np.frombuffer((b/'tables.bin').read_bytes(),dtype).astype(R)
    assert len(tables)==505001+2*60*60*35
    gamma=tables[:505001];coltab=tables[505001:505001+126000].reshape((60,60,35),order='F')
    coltabn=tables[505001+126000:].reshape((60,60,35),order='F')
    path=out/(mode+'.npz')
    np.savez_compressed(path,shape_inputs=incoming,shape_outputs=outgoing,lookup_indices=indices,lookup_values=values,
                        gamma_table=gamma,coltab=coltab,coltabn=coltabn)
    finite={name:int(np.count_nonzero(~np.isfinite(a))) for name,a in
            [('shape_inputs',incoming),('shape_outputs',outgoing),('lookup',values),('tables',tables)]}
    rows.append(dict(mode=mode,E26_BYTE_columns=17,shape_calls=len(incoming),lookup_calls=len(indices),
                     lookup_table_components=sorted(set(map(int,indices[:,0]))),nonfinite_values=finite,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
    observed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
    shape_input_names=before.split(','),shape_output_names=after.split(','),
    scope='Print-only original call observations; E26 entire34 full columns/scalars BYTE to pristine reference',rows=rows),indent=2)+'\n')
print(rows,flush=True)
