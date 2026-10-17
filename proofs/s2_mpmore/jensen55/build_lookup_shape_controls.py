"""Original, unchanged lookup/shape bodies: same-input FMA and input-ULP controls.

Lookup coordinates are perturbed BEFORE cell normalization. Every normalized
coordinate subtraction is proved byte equal to the original index subtraction.
The shape corpus adds the previously unobserved lower-delta branch; constants
NU/AO/FOURTHIRDSPI/GAMMNU and the original gamma table are never perturbed.
"""
from pathlib import Path
import hashlib,itertools,json,re,subprocess,sys
import numpy as np

FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')


def main(tree,artifact):
    assert (tree/'.git').is_file() and not artifact.exists()
    build=artifact/'build';out=artifact/'fixtures';build.mkdir(parents=True);out.mkdir()
    original=SOURCE.read_text()
    bodies=[]
    for name in ('access_lookup_table','var_check'):
        start=original.index('  subroutine '+name+'(')
        end=original.index('  end subroutine '+name,start)+len('  end subroutine '+name)
        bodies.append(original[start:end])
    module=('module original_controls\nimplicit none\n'
            'real,parameter :: PI=3.14159265,RHOI=920.\n'
            'real :: gamma_tab(505001)\ncontains\n'+'\n'.join(bodies)+'\nend module\n')
    # F order matches the original (jj,ii,i,k) indices exactly.
    corners=','.join('tab('+','.join(map(str,(*a,1)))+')'
                     for a in itertools.product((1,2),repeat=4))
    caller=f'''program caller
use original_controls
implicit none
integer :: unit,n,j
real :: tab(2,2,2,2,1),d1,d2,d4,d5,proc,s(16)
open(newunit=unit,file='gamma.bin',access='stream',form='unformatted',status='old')
read(unit) gamma_tab
close(unit)
open(newunit=unit,file='lookup.bin',access='stream',form='unformatted',status='old')
read(unit) n
open(99,file='lookup_out.bin',access='stream',form='unformatted',status='replace')
do j=1,n
read(unit) d1,d2,d4,d5,{corners}
call access_lookup_table(tab,1,1,1,1,1,d1,d2,d4,d5,proc)
write(99) proc
enddo
close(unit);close(99)
open(newunit=unit,file='shape.bin',access='stream',form='unformatted',status='old')
read(unit) n
open(99,file='shape_out.bin',access='stream',form='unformatted',status='replace')
do j=1,n
read(unit) s(1:12)
call var_check(s(1),s(2),s(3),s(4),s(5),s(6),s(7),s(8),s(13),s(9),s(10),s(11),s(12),s(14),s(15),s(16))
write(99) s
enddo
close(unit);close(99)
end program
'''
    rows=[]
    for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
        with np.load(tree/f'proofs/s2_mpmore/jensen55/stages/{mode}.npz') as data:
            values=data['lookup_values'];idx=data['lookup_indices'];gamma=data['gamma_table']
            observed_shape=data['shape_inputs']
        baseline=values[:,:4];variants=[baseline];labels=['baseline']
        for field in range(4):
            for direction in (-np.inf,np.inf):
                changed=baseline.copy();changed[:,field]=np.nextafter(changed[:,field],R(direction))
                variants.append(changed);labels.append(f'coordinate{field}_'+('minus1ULP' if direction<0 else 'plus1ULP'))
        original_indices=idx[:,[3,4,2,1]].astype(R)
        coords=np.stack(variants);delta=coords-original_indices
        normalized=delta+R(1.)
        assert (normalized-R(1.)).tobytes()==delta.tobytes(),'Cell normalization changed actual delta bits'
        lookup=np.concatenate([normalized,np.broadcast_to(values[None,:,5:],(len(variants),len(values),16))],axis=-1)
        # Use actual callers as the other arguments; explicitly cover both sides
        # of both original shape thresholds, plus a clearly active lower branch.
        targets=np.asarray([R(.4),np.nextafter(R(.55),R(-np.inf)),R(.55),
                            np.nextafter(R(.55),R(np.inf)),R(1.3),np.nextafter(R(1.3),R(np.inf))])
        shape=np.repeat(observed_shape[[0,len(observed_shape)//2,-1]],len(targets),axis=0)
        shape[:,5]=np.tile(targets,3)
        shape_variants=[shape];shape_labels=['baseline']
        for field in range(4,12):
            for direction in (-np.inf,np.inf):
                changed=shape.copy();changed[:,field]=np.nextafter(changed[:,field],R(direction))
                shape_variants.append(changed);shape_labels.append(f'argument{field}_'+('minus1ULP' if direction<0 else 'plus1ULP'))
        shape_inputs=np.stack(shape_variants)
        io_dtype=np.dtype(R).newbyteorder('>')
        raw_lookup=np.asarray([lookup.shape[0]*lookup.shape[1]],'>i4').tobytes()+lookup.astype(io_dtype).tobytes()
        raw_shape=np.asarray([shape_inputs.shape[0]*shape_inputs.shape[1]],'>i4').tobytes()+shape_inputs.astype(io_dtype).tobytes()
        for contraction in ('noFMA','FMA'):
            b=build/(mode+'_'+contraction);b.mkdir()
            (b/'module.F90').write_text(module);(b/'caller.F90').write_text(caller)
            (b/'lookup.bin').write_bytes(raw_lookup);(b/'shape.bin').write_bytes(raw_shape)
            (b/'gamma.bin').write_bytes(gamma.astype(io_dtype).tobytes())
            flags=['-O2','-ffree-line-length-none','-finit-real=zero','-finit-integer=0',
                   '-finit-logical=false','-fconvert=big-endian','-mfma',
                   '-ffp-contract='+('off' if contraction=='noFMA' else 'fast')]
            if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
            subprocess.run([FC,*flags,'-c','module.F90'],cwd=b,check=True,timeout=60)
            subprocess.run([FC,*flags,'module.o','caller.F90','-o','oracle'],cwd=b,check=True,timeout=60)
            subprocess.run([str(b/'oracle')],cwd=b,check=True,timeout=30)
            lookup_outputs=np.frombuffer((b/'lookup_out.bin').read_bytes(),io_dtype).astype(R).reshape(lookup.shape[:2])
            shape_outputs=np.frombuffer((b/'shape_out.bin').read_bytes(),io_dtype).astype(R).reshape((*shape_inputs.shape[:2],16))
            assert np.isfinite(lookup_outputs).all() and np.isfinite(shape_outputs).all()
            if contraction=='noFMA':assert lookup_outputs[0].tobytes()==values[:,4].tobytes(),'Unchanged original noFMA baseline BYTE'
            assembly=subprocess.check_output(['objdump','-d','module.o'],cwd=b,text=True)
            fma=len(re.findall(r'\bv(?:f|fn)m(?:add|sub)\w*',assembly))
            assert (fma==0)==(contraction=='noFMA')
            path=out/(mode+'_'+contraction+'.npz')
            np.savez_compressed(path,lookup_original_coords=coords,lookup_inputs=lookup,lookup_outputs=lookup_outputs,
                                shape_inputs=shape_inputs,shape_outputs=shape_outputs)
            rows.append(dict(mode=mode,contraction=contraction,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             flags=flags,FMA_opcode_rows=fma,lookup_input_sha256=hashlib.sha256(raw_lookup).hexdigest(),
                             shape_input_sha256=hashlib.sha256(raw_shape).hexdigest(),object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        unchanged_body_sha256=[hashlib.sha256(body.encode()).hexdigest() for body in bodies],
        extracted_module_sha256=hashlib.sha256(module.encode()).hexdigest(),caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
        lookup_labels=labels,shape_labels=shape_labels,rows=rows,
        scope='Unchanged original bodies; actual16 table corners; +/-1ULP actual original coordinates before BYTE-proved cell normalization; shape lower-delta boundary'),indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
