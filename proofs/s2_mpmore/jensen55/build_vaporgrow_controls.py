"""Unchanged original vapor routine, actual-argument ULP and contraction bounds."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import numpy as np

SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'


def main(tree,artifact):
    assert (tree/'.git').is_file() and not artifact.exists()
    build=artifact/'build';out=artifact/'fixtures';build.mkdir(parents=True);out.mkdir()
    folder=tree/'proofs/s2_mpmore/jensen55';manifest=json.loads((folder/'vapor/manifest.json').read_text())
    inputs=manifest['input_names'];outputs=manifest['output_names']
    pristine=SOURCE.read_text()
    signature=re.search(r'(?im)^\s*subroutine vaporgrow\((.*?)\)',pristine,re.S)
    names=re.sub(r'[\s&]+','',signature[1]).lower().split(',')
    # Original gamma table is an actual frozen dependency, not recomputed under FMA.
    setter='''
subroutine s2_set_gamma(a)
real,intent(in) :: a(505001)
allocate(gamma_tab(505001))
gamma_tab=a
end subroutine
'''
    source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:setter+m[0],pristine);assert count==1
    def at(name):return 'v('+str(names.index(name)+1)+')'
    caller=f'''program caller
use module_mp_jensen_ishmael,only: vaporgrow,s2_set_gamma
implicit none
integer :: unit,n,j
real :: v(45),g(505001)
open(newunit=unit,file='gamma.bin',access='stream',form='unformatted',status='old')
read(unit) g
close(unit)
call s2_set_gamma(g)
open(newunit=unit,file='input.bin',access='stream',form='unformatted',status='old')
read(unit) n
open(99,file='output.bin',access='stream',form='unformatted',status='replace')
do j=1,n
v=0.
read(unit) {','.join(at(k) for k in inputs)}
call vaporgrow({','.join(at(k) for k in names)})
write(99) {','.join(at(k) for k in outputs)}
enddo
close(unit);close(99)
end program
'''
    frozen={'nu','ao','gammnu','i_gammnu','fourthirdspi','vtbarb','vtbarbm','vtbarbz','fvdum','fhdum','rdout','dsdumout'}
    arguments=[k for k in inputs if k not in frozen]
    rows=[]
    for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
        with np.load(folder/'vapor'/(mode+'.npz')) as data:baseline=data['inputs'];expected=data['outputs']
        with np.load(folder/'stages'/(mode+'.npz')) as data:gamma=data['gamma_table']
        variants=[baseline];labels=['baseline']
        for name in arguments:
            for direction in (-np.inf,np.inf):
                a=baseline.copy();n=inputs.index(name);a[:,n]=np.nextafter(a[:,n],R(direction))
                variants.append(a);labels.append(name+('_minus1ULP' if direction<0 else '_plus1ULP'))
        values=np.stack(variants);io=np.dtype(R).newbyteorder('>')
        raw=np.asarray([values.shape[0]*values.shape[1]],'>i4').tobytes()+values.astype(io).tobytes()
        for contraction in ('noFMA','FMA'):
            b=build/(mode+'_'+contraction);b.mkdir()
            (b/'module.F').write_text(source);(b/'stubs.f90').write_text('module module_wrf_error\nend module\n')
            (b/'caller.F90').write_text(caller);(b/'input.bin').write_bytes(raw)
            (b/'gamma.bin').write_bytes(gamma.astype(io).tobytes())
            flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0',
                   '-finit-logical=false','-fconvert=big-endian','-mfma','-ffp-contract='+('off' if contraction=='noFMA' else 'fast')]
            if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
            for name in ('stubs.f90','module.F','caller.F90'):
                subprocess.run([FC,*flags,'-c',name],cwd=b,check=True,timeout=180)
            subprocess.run([FC,*flags,'stubs.o','module.o','caller.o','-o','oracle'],cwd=b,check=True,timeout=180)
            subprocess.run([str(b/'oracle')],cwd=b,check=True,timeout=60)
            result=np.frombuffer((b/'output.bin').read_bytes(),io).astype(R).reshape((*values.shape[:2],len(outputs)))
            assert np.isfinite(result).all()
            if contraction=='noFMA':assert result[0].tobytes()==expected.tobytes(),'Actual replay must be BYTE to full-column call observation'
            asm=subprocess.check_output(['objdump','-d','module.o'],cwd=b,text=True)
            fma=len(re.findall(r'\bv(?:f|fn)m(?:add|sub)\w*',asm))
            assert (fma==0)==(contraction=='noFMA')
            path=out/(mode+'_'+contraction+'.npz');np.savez_compressed(path,inputs=values,outputs=result)
            rows.append(dict(mode=mode,contraction=contraction,input_sha256=hashlib.sha256(raw).hexdigest(),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),FMA_opcode_rows=fma,flags=flags,
                             object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
        local_module_sha256=hashlib.sha256(source.encode()).hexdigest(),caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
        input_names=inputs,output_names=outputs,perturbed_arguments=arguments,frozen_fields=sorted(frozen),
        labels=labels,rows=rows,scope='Actual original routine arguments;constants/heldoutput/tablevalues frozen;independent one-ULP input perturbations and same-input FMA'),indent=2)+'\n')
    print(dict(controls=len(rows),arguments=len(arguments),observations_per_arm=result.size,rows=rows),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
