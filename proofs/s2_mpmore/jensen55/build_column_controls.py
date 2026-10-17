"""Unchanged public WRF body: frozen actual inputs/tables, FMA and input ULP.

Each REAL argument (19 complete column arrays and DT) is independently changed
by one ULP. IT stays fixed. Original initialized aggregation tables are frozen
to the exact values baked by the port, rather than regenerated under FMA.
"""
from pathlib import Path
import hashlib,importlib.util,json,re,subprocess,sys
import numpy as np

FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')


def main(tree,artifact):
    assert (tree/'.git').is_file() and not artifact.exists()
    build=artifact/'build';out=artifact/'fixtures';build.mkdir(parents=True);out.mkdir()
    folder=tree/'proofs/s2_mpmore/jensen55'
    spec=importlib.util.spec_from_file_location('original_jensen',folder/'build_reference.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    spec=importlib.util.spec_from_file_location('frozen_original_tables',folder/'reference_tables.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    pristine=SOURCE.read_text()
    setter='''
subroutine s2_frozen_tables()
integer :: unit
open(newunit=unit,file='tables.bin',access='stream',form='unformatted',status='old')
read(unit) itab,itabr,gamma_tab,coltab,coltabn,igrdata
close(unit)
end subroutine
'''
    source,count=re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda m:setter+m[0],pristine);assert count==1
    driver=original.caller().replace('only: jensen_ishmael_init,mp_jensen_ishmael',
                                    'only: jensen_ishmael_init,mp_jensen_ishmael,s2_frozen_tables')
    driver=driver.replace('integer :: nz,it,unit,case_index','integer :: nz,it,unit,case_index,variant_index')
    driver=driver.replace(' call jensen_ishmael_init()',' call jensen_ishmael_init()\n call s2_frozen_tables()\n open(98,file="inputs.bin",access="stream",form="unformatted",status="old")\n open(99,file="outputs.bin",access="stream",form="unformatted",status="replace")')
    driver=driver.replace(' do case_index=1,17',' do variant_index=1,41\n do case_index=1,17')
    begin=driver.index(' write(filename,\'("input_",I0,".bin")\') case_index')
    end=driver.index(' read(unit) nz,it',begin)
    driver=driver[:begin]+driver[end:]
    driver=driver.replace('read(unit)','read(98)').replace(' close(unit)','')
    begin=driver.index(' write(filename,\'("output_",I0,".bin")\') case_index')
    end=driver.index(' write(unit)',begin)
    driver=driver[:begin]+driver[end:]
    driver=driver.replace('write(unit)','write(99)')
    driver=driver.replace(' enddo\nend program',' enddo\n enddo\n close(98);close(99)\nend program')
    rows=[]
    for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
        gold=[json.loads((folder/f'fixtures/{mode}_{case}.json').read_text()) for case in range(1,18)]
        sizes={len(g['input']['qv']) for g in gold};assert len(sizes)==1
        nz=sizes.pop();names=original.NAMES[:19]
        baseline=np.stack([np.stack([np.asarray(g['input'][n],R) for n in names]) for g in gold])
        dt=np.asarray([g['dt'] for g in gold],R);inputs=[baseline];times=[dt];labels=['baseline']
        for name in (*names,'dt'):
            for direction in (-np.inf,np.inf):
                x=baseline.copy();t=dt.copy()
                if name=='dt':t=np.nextafter(t,R(direction))
                else:
                    n=names.index(name);x[:,n,:]=np.nextafter(x[:,n,:],R(direction))
                inputs.append(x);times.append(t);labels.append(name+('_minus1ULP' if direction<0 else '_plus1ULP'))
        inputs=np.stack(inputs);times=np.stack(times);assert len(inputs)==41
        io=np.dtype(R).newbyteorder('>');chunks=[]
        for variant in range(41):
            for case,g in enumerate(gold):
                chunks.append(np.asarray([nz,g['it']],'>i4').tobytes()+np.asarray(times[variant,case],io).tobytes()+inputs[variant,case].astype(io).tobytes())
        raw=b''.join(chunks)
        tables=reference.load(tree,mode)
        table_raw=b''.join(tables[n].astype(io).tobytes(order='F') for n in ('itab','itabr','gamma_tab','coltab','coltabn','igrdata'))
        expected=np.stack([np.concatenate([*(np.asarray(g['output'][n],R) for n in original.NAMES),
                                           np.asarray([g['precipitation'][n] for n in ('rain','rainstep','snow','snowstep')],R)]) for g in gold])
        for contraction in ('noFMA','FMA'):
            b=build/(mode+'_'+contraction);b.mkdir()
            (b/'module.F').write_text(source);(b/'stubs.f90').write_text('module module_wrf_error\nend module\n')
            (b/'driver.F90').write_text(driver);(b/'inputs.bin').write_bytes(raw);(b/'tables.bin').write_bytes(table_raw)
            for file in original.TABLE_FILES:(b/file).symlink_to(original.TABLES/file)
            flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false',
                   '-fconvert=big-endian','-frecord-marker=4','-mfma','-ffp-contract='+('off' if contraction=='noFMA' else 'fast')]
            if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
            for file in ('stubs.f90','module.F','driver.F90'):
                subprocess.run([FC,*flags,'-c',file],cwd=b,check=True,timeout=180)
            subprocess.run([FC,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b,check=True,timeout=180)
            subprocess.run([str(b/'oracle')],cwd=b,check=True,timeout=180,capture_output=True)
            values=np.frombuffer((b/'outputs.bin').read_bytes(),io).astype(R).reshape(41,17,-1)
            assert np.isfinite(values).all()
            if contraction=='noFMA':assert values[0].tobytes()==expected.tobytes(),'Full unchanged original noFMA baseline BYTE'
            asm=subprocess.check_output(['objdump','-d','module.o'],cwd=b,text=True)
            fma=len(re.findall(r'\bv(?:f|fn)m(?:add|sub)\w*',asm));assert (fma==0)==(contraction=='noFMA')
            path=out/(mode+'_'+contraction+'.npz')
            np.savez_compressed(path,inputs=inputs,dt=times,outputs=values)
            rows.append(dict(mode=mode,contraction=contraction,observations=values.size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             input_sha256=hashlib.sha256(raw).hexdigest(),frozen_table_sha256=hashlib.sha256(table_raw).hexdigest(),
                             FMA_opcode_rows=fma,flags=flags,object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(pristine.encode()).hexdigest(),
        local_source_sha256=hashlib.sha256(source.encode()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        input_names=names,output_names=original.NAMES,precipitation_names=('rain','rainstep','snow','snowstep'),
        labels=labels,rows=rows,scope='Actual public function arguments frozen; each complete REAL array/DT independently one ULP; actual baked table values frozen identically under FMA/noFMA'),indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
