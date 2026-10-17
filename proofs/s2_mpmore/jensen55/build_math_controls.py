"""Independent original function-input ULP and same-input FMA controls."""
from pathlib import Path
import hashlib,importlib.util,json,re,subprocess,sys
import numpy as np

tree,artifact=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert (tree/'.git').is_file()
if artifact.exists():raise SystemExit('Use a fresh artifact directory')
spec=importlib.util.spec_from_file_location('original_jensen_math',tree/'proofs/s2_mpmore/jensen55/build_math_reference.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
driver=original.caller().replace('nu\nlogical','nu,log_argument\nlogical')
assert 'nu,log_argument\nlogical' in driver
driver=driver.replace('read(unit) temperature,x,ds,a,rime',
                      'read(unit) temperature,x,ds,a,rime,alph,log_argument')
driver=driver.replace('alph=(0.1e-6)**(1.-ds)','! Freeze actual capacitance ALPHSTR input bits.')
driver=driver.replace('v(4)=gammln(abs(x)+.5)','v(4)=gammln(log_argument)')
work=artifact/'build';out=artifact/'fixtures';work.mkdir(parents=True);out.mkdir()
source=original.SOURCE.read_bytes();rows=[]
for mode,R in [('fp32',np.float32),('fp64',np.float64)]:
    golden=json.loads((tree/f'proofs/s2_mpmore/jensen55/math_fixtures/{mode}.json').read_text())
    baseline=np.asarray(golden['inputs'],R);expected=np.asarray(golden['outputs'],R)
    baseline=np.column_stack([baseline,expected[:,5],np.abs(baseline[:,1])+R(.5)])
    variants=[baseline]
    labels=['baseline']
    for index in range(baseline.shape[1]):
        for direction in (-np.inf,np.inf):
            changed=baseline.copy();changed[:,index]=np.nextafter(changed[:,index],R(direction))
            variants.append(changed);labels.append(f'argument{index}_'+('minus1ULP' if direction<0 else 'plus1ULP'))
    inputs=np.concatenate(variants);io_dtype=np.dtype(R).newbyteorder('>')
    raw=np.asarray([len(inputs)],dtype='>i4').tobytes()+inputs.astype(io_dtype).tobytes()
    for contraction in ('noFMA','FMA'):
        b=work/(mode+'_'+contraction);b.mkdir()
        (b/'module.F').write_bytes(source);(b/'stubs.f90').write_text('module module_wrf_error\nend module\n')
        (b/'driver.F90').write_text(driver);(b/'input.bin').write_bytes(raw)
        flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero',
               '-finit-integer=0','-finit-logical=false','-fconvert=big-endian','-frecord-marker=4',
               '-mfma','-ffp-contract='+('off' if contraction=='noFMA' else 'fast')]
        if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
        for name in ('stubs.f90','module.F','driver.F90'):
            subprocess.run([original.FC,*flags,'-c',name],cwd=b,check=True,timeout=180)
        subprocess.run([original.FC,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b,check=True,timeout=180)
        subprocess.run([str(b/'oracle')],cwd=b,check=True,timeout=30)
        data=np.frombuffer((b/'output.bin').read_bytes(),io_dtype).reshape(len(variants),len(baseline),8)
        assert np.isfinite(data).all()
        if contraction=='noFMA':
            assert data[0].astype(R).tobytes()==expected.tobytes(),(mode,'frozen noFMA baseline must be BYTE')
        asm=subprocess.check_output(['objdump','-d','module.o'],cwd=b,text=True)
        opcodes=len(re.findall(r'\bv(?:f|fn)m(?:add|sub)\w*',asm))
        assert (opcodes==0)==(contraction=='noFMA'),(mode,contraction,opcodes)
        result=dict(mode=mode,contraction=contraction,labels=labels,inputs=inputs.tolist(),
                    baseline=baseline.tolist(),outputs=data.tolist(),baseline_count=len(baseline),
                    scope='Actual original function arguments, including frozen ALPHSTR/gammln input; each argument independently +/-1ULP')
        path=out/f'{mode}_{contraction}.json';path.write_text(json.dumps(result,separators=(',',':'))+'\n')
        rows.append(dict(mode=mode,contraction=contraction,input_sha256=hashlib.sha256(raw).hexdigest(),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),FMA_opcode_rows=opcodes,
                         module_object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest(),flags=flags,
                         noFMA_baseline_BYTE=contraction=='noFMA'))
for mode in ('fp32','fp64'):
    assert len({r['input_sha256'] for r in rows if r['mode']==mode})==1
(out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source).hexdigest(),
    caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),rows=rows),indent=2)+'\n')
print(rows,flush=True)
