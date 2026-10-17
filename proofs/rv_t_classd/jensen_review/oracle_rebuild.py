"""Recompile untouched Jensen public WRF routine on the frozen real columns."""
from pathlib import Path
import hashlib, importlib.util, json, subprocess
import numpy as np
import netCDF4

P=Path(__file__).resolve().parent
ROOT=P/'snapshot'
path=ROOT/'proofs/t_classd/jensen_refl_oracle/build.py'
spec=importlib.util.spec_from_file_location('review_jensen_builder',path)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
fixture=path.parent/'fixtures'
m=json.loads((fixture/'manifest.json').read_text())
data=np.load(fixture/'oracle.npz')
assert hashlib.sha256((fixture/'oracle.npz').read_bytes()).hexdigest()==m['npz_sha256']
assert hashlib.sha256(b.SOURCE.read_bytes()).hexdigest()==m['source_sha256']
assert hashlib.sha256(b.caller(m['columns']).encode()).hexdigest()==m['caller_sha256']
for name,h in m['tables'].items():assert hashlib.sha256((b.TABLES/name).read_bytes()).hexdigest()==h
assert np.isfinite(data['inputs']).all() and np.isfinite(data['outputs']).all()

# Re-read the original CPU fields independently; neither port nor GPU output is truth.
columns=[];frames=[];real_ratio_changed_theta=0
for frame in m['frames']:
    case=b.CASE/frame['file']
    assert hashlib.sha256(case.read_bytes()).hexdigest()==frame['sha256']
    with netCDF4.Dataset(case) as ds:
        def g(name):return np.asarray(ds[name][0],np.float32)
        qv=g('QVAPOR');theta=g('T')+np.float32(300)
        z=(g('PH')+g('PHB'))/np.float32(9.81)
        f=dict(p=g('P')+g('PB'),dz=z[1:]-z[:-1],qv=qv,qc=g('QCLOUD'),qr=g('QRAIN'),nr=g('QNRAIN'))
        f['th']=theta/(np.float32(1)+np.float32(461.6/287.)*qv)
        real_th=theta/(np.float32(1)+(np.float32(461.6)/np.float32(287.))*qv)
        for i,suffix in ((1,''),(2,'2'),(3,'3')):
            for n,v in [('qi','QICE'),('ni','QNICE'),('ai','QVOLI'),('ci','QAOLI')]:f[n+str(i)]=g(v+suffix)
        for j,i in frame['columns']:
            columns.append(np.concatenate([f[n][:,j,i] for n in m['input_names']]))
            real_ratio_changed_theta+=int(np.count_nonzero(real_th[:,j,i]!=f['th'][:,j,i]))
    frames.append(dict(file=frame['file'],sha256=frame['sha256'],columns=len(frame['columns'])))
actual=np.stack(columns)
assert actual.tobytes()==data['inputs'].tobytes()

scratch=P/'oracle_build';scratch.mkdir(exist_ok=False)
(scratch/'module.F').write_bytes(b.SOURCE.read_bytes())
(scratch/'stubs.f90').write_text('module module_wrf_error\nend module\n')
(scratch/'driver.F90').write_text(b.caller(m['columns']))
for name in b.TABLE_FILES:(scratch/name).symlink_to(b.TABLES/name)
for name in ('stubs.f90','module.F','driver.F90'):
    subprocess.run([b.FC,*m['flags'],'-c',name],cwd=scratch,check=True,timeout=300)
subprocess.run([b.FC,*m['flags'],'stubs.o','module.o','driver.o','-o','oracle'],cwd=scratch,check=True,timeout=300)
big=np.dtype('>f4')
for c,row in enumerate(actual,1):
    raw=np.asarray([m['nz'],m['it']],'>i4').tobytes()+np.asarray(m['dt'],big).tobytes()+np.asarray(row,big).tobytes()
    (scratch/f'input_{c}.bin').write_bytes(raw)
run=subprocess.run([str(scratch/'oracle')],cwd=scratch,capture_output=True,check=True,timeout=900)
out=np.stack([np.frombuffer((scratch/f'output_{c}.bin').read_bytes(),big).astype(np.float32) for c in range(1,m['columns']+1)])
assert np.isfinite(out).all() and out.tobytes()==data['outputs'].tobytes()
start=m['output_names'].index('dbz')*m['nz'];dbz=out[:,start:start+m['nz']]
result=dict(source_sha256=m['source_sha256'],caller_sha256=m['caller_sha256'],tables=m['tables'],frames=frames,
    real_cpu_inputs_byte=True,all_fresh_public_outputs_byte=True,columns=m['columns'],levels=m['nz'],
    real4_outputs=int(out.size),dbz_values=int(dbz.size),live_dbz_values=int((dbz>-35).sum()),
    finite=True,actual_wrf_real_ratio_changes_frozen_theta_elements=real_ratio_changed_theta,
    input_mapping_note='Frozen fixture uses fp32(461.6/287.); explicit REAL operand division difference is counted separately.',
    scope='Pristine public mp_jensen_ishmael, REAL4, dt18/non-first it2001 on real c87 history columns; not same-frame forecast parity',
    compiled_source_unchanged=True,flags=m['flags'],compiler=b.FC,stdout_sha256=hashlib.sha256(run.stdout).hexdigest())
(P/'RV34_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
