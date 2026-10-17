"""Original two-step columns: native evolved habits and nonzero accumulators."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file() and not artifact.exists()
artifact.mkdir()
spec=importlib.util.spec_from_file_location('original_jensen',tree/'proofs/s2_mpmore/jensen55/build_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
pristine=original.SOURCE.read_text()
source=pristine.replace('  !.. Inherent growth ratio data','  integer :: s2_step=1,s2_case=0\n\n  !.. Inherent growth ratio data',1)
for anchor,stage in [('    nstep = 1',1),('  !.. Check after sedimentation',2),('  end subroutine me_ishmael',3)]:
    assert source.count(anchor)==1
    source=source.replace(anchor,f'    call s2_ni_record({stage},kte-kts+1,ni(3,kts:kte))\n'+anchor)
anchor='  end subroutine aggregation';assert source.count(anchor)==1
source=source.replace(anchor,"    call s2_agg_record([qagg1,qagg2,qagg3,nagg1,nagg2,nagg3])\n"+anchor)
observer='''
subroutine s2_set_step(value,case_index)
integer,intent(in) :: value,case_index
s2_step=value
s2_case=case_index
end subroutine
subroutine s2_ni_record(stage,n,values)
integer,intent(in) :: stage,n
real,intent(in) :: values(n)
integer :: unit
open(newunit=unit,file='two_step_ni.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) s2_case,s2_step,stage,n
write(unit) values
close(unit)
end subroutine
subroutine s2_agg_record(values)
real,intent(in) :: values(6)
integer :: unit
open(newunit=unit,file='two_step_agg.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) s2_step
write(unit) values
close(unit)
end subroutine
'''
source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:observer+m[0],source);assert count==1
path=artifact/'observed.F';path.write_text(source);original.SOURCE=path
driver=original.caller().replace('only: jensen_ishmael_init,mp_jensen_ishmael','only: jensen_ishmael_init,mp_jensen_ishmael,s2_set_step')
start=driver.index(' call mp_jensen_ishmael(');end=driver.index('type1,type2,type3)',start)+len('type1,type2,type3)')
call=driver[start:end]
first='''
 write(filename,'("first_",I0,".bin")') case_index
 open(newunit=unit,file=filename,access='stream',form='unformatted',status='replace')
 write(unit) '''+','.join(original.NAMES)+''',rain,rainstep,snow,snowstep
 close(unit)
'''
driver=driver[:start]+' call s2_set_step(1,case_index)\n'+call+first+' call s2_set_step(2,case_index)\n'+call.replace('(it,dt,','(it+1,dt,')+driver[end:]
original.caller=lambda:driver
original.main(tree,artifact/'capture')
out=artifact/'fixtures';out.mkdir();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    io=np.dtype(R).newbyteorder('>')
    for case in range(1,18):
        path=tree/f'proofs/s2_mpmore/jensen55/fixtures/{mode}_{case}.json';baseline=json.loads(path.read_text());nz=len(baseline['input']['qv'])
        observed=np.frombuffer((artifact/f'capture/build/{mode}/first_{case}.bin').read_bytes(),io).astype(R)
        expected=np.concatenate([*(np.asarray(baseline['output'][n],R) for n in original.NAMES),
                                  np.asarray([baseline['precipitation'][n] for n in ('rain','rainstep','snow','snowstep')],R)])
        assert observed.tobytes()==expected.tobytes(),'First public step BYTE despite observer'
        second=json.loads((artifact/f'capture/fixtures/{mode}_{case}.json').read_text())
        prior=tree/f'proofs/s2_mpmore/jensen55/two_step/{mode}_{case}.json'
        if prior.is_file():
            old=json.loads(prior.read_text())
            assert second['output']==old['output'] and second['precipitation']==old['precipitation'],'Second public fields unchanged by NI observer'
        second.update(steps=2,initial_it=second['it'],final_it=second['it']+1,
                      first_output={n:observed[i*nz:(i+1)*nz].tolist() for i,n in enumerate(original.NAMES)},
                      first_precipitation=dict(zip(('rain','rainstep','snow','snowstep'),map(float,observed[-4:]))))
        dest=out/f'{mode}_{case}.json';dest.write_text(json.dumps(second,indent=2)+'\n')
        rows.append(dict(mode=mode,case=case,first_step_BYTE=True,finite=second['finite'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    raw=(artifact/f'capture/build/{mode}/two_step_agg.bin').read_bytes();offset=0;steps=[];values=[]
    while offset<len(raw):
        steps.append(int(np.frombuffer(raw,'>i4',count=1,offset=offset)[0]));offset+=4
        values.append(np.frombuffer(raw,io,count=6,offset=offset).astype(R));offset+=6*io.itemsize
    assert offset==len(raw);steps=np.asarray(steps,np.int32);values=np.stack(values);assert np.isfinite(values).all()
    np.savez_compressed(out/(mode+'_aggregation.npz'),steps=steps,values=values)
    raw=(artifact/f'capture/build/{mode}/two_step_ni.bin').read_bytes();offset=0;ids=[];ni=[]
    while offset<len(raw):
        case,step,stage,n=map(int,np.frombuffer(raw,'>i4',count=4,offset=offset));offset+=16
        ids.append((case,step,stage));ni.append(np.frombuffer(raw,io,count=n,offset=offset).astype(R));offset+=n*io.itemsize
    assert offset==len(raw)
    np.savez_compressed(out/(mode+'_ni_stages.npz'),indices=np.asarray(ids,np.int32),Ni3=np.stack(ni))
    print(mode,{str(step):dict(calls=int(np.count_nonzero(steps==step)),mass_transfer_calls=int(np.count_nonzero(np.any(values[steps==step,:3]!=0,axis=1)))) for step in (1,2)},flush=True)
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
    observed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),rows=rows,
    scope='Two unchanged original public calls, second uses first evolved prognostics/accumulators and IT+1; all first-step fields BYTE; aggregation census print-only'),indent=2)+'\n')
