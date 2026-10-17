"""Matched pristine22 case15 controls, varying only QC(k13) by exact ULPs."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import numpy as np

if len(sys.argv)!=3:
 raise SystemExit('usage: build_ulp_reference.py SOURCE_ROOT FRESH_ARTIFACT_ROOT')
tree=Path(sys.argv[1]).resolve();lane=Path(sys.argv[2]).resolve()
work=lane/'build';out=lane/'ulp'
if out.exists(): raise SystemExit('Use a fresh artifact root; preserve previous controls')
lane.mkdir(parents=True,exist_ok=True)
proof=tree/'proofs/v034/oracle/nssl2mom'
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
driver=(proof/'knife_edge/driver_knife.F90').read_text()
driver=driver.replace('nssl_hail_on=.TRUE.','nssl_hail_on=.FALSE.').replace('VHL=VHL, F_VHL=.TRUE.','VHL=VHL, F_VHL=.FALSE.')
start=driver.index('  ! knife-edge probe (lane o1-nssl')
end=driver.index('  END BLOCK',start)+len('  END BLOCK')
driver=driver[:start]+'''  ! Matched original22 caller: exact integer ULP perturbation, not a body edit.
  BLOCK
    CHARACTER(LEN=64) :: ev
    INTEGER :: ln, st, shift
    INTEGER(KIND=8) :: bits
    CALL GET_ENVIRONMENT_VARIABLE('MPM_SHIFT', ev, ln, st)
    shift=0
    IF (st == 0 .and. ln > 0) READ(ev,*) shift
    IF (storage_size(QC(1,13,1)) /= 64) STOP 'ULP caller requires REAL64'
    bits=transfer(QC(1,13,1),bits)
    QC(1,13,1)=transfer(bits+int(shift,8),QC(1,13,1))
  END BLOCK'''+driver[end:]
driver=driver.replace('  ! save inputs','  QHL=0.0; CHL=0.0; VHL=0.0 ! Absent Registry hail fields.\n  ! save inputs')
# Retain the legacy dump for exact baseline reproduction, and separately print
# enough digits for a binary64 round trip. ES23.15 has only 16 significant digits.
for suffix in ['arr(kk)','arr(1,kk,1)']:
    line="      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', "+suffix
    assert driver.count(line)==1
    exact="      WRITE(*,'(A,A,A,I0,A,ES26.17E3)') 'X',name,'[',kk,']=', "+suffix
    driver=driver.replace(line,line+'\n'+exact)
work.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
(work/'module.F').write_bytes(source.read_bytes())
(work/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes())
(work/'driver.F90').write_text(driver);(work/'namelist.input').write_text('')
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off','-fdefault-real-8','-fdefault-double-8']
def run(args,**kw):return subprocess.run(list(map(str,args)),check=True,timeout=120,**kw)
for f in ['stubs.f90','module.F','driver.F90']:
    # Same copied pristine body/stubs and identical explicit flags as previous
    # arm: only the caller's bit perturbation changed.
    if f!='driver.F90' and (work/(Path(f).stem+'.o')).exists():continue
    run([fc,*flags,'-c',f],cwd=work)
run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=work)
base=json.loads((tree/'proofs/s2_mpmore/nssl22/fixtures/fp64_15.json').read_text())
rows=[]
for shift in [0,-1,1,-2,2]:
    env=dict(os.environ,MPM_SHIFT=str(shift))
    raw=work/f'shift_{shift}.txt';target=out/f'shift_{shift}.json'
    with raw.open('w') as f:run([work/'oracle',15],cwd=work,env=env,stdout=f)
    run([sys.executable,proof/'dump_to_json_v2.py',raw,target,'fp64'])
    d=json.loads(target.read_text())
    exact={k[1:]:v for k,v in d['columns'].items() if k.startswith('X')}
    legacy={k:v for k,v in d['columns'].items() if not k.startswith('X')}
    if shift==0:
        assert legacy==base['columns'] and d['scalars']==base['scalars'],'Must exactly reproduce staged original22 baseline'
        exact_base=exact
    d['columns']=exact
    d['legacy_columns']=legacy
    target.write_text(json.dumps(d)+'\n')
    q0=np.float64(exact_base['QC_IN'][12]);q=np.float64(exact['QC_IN'][12])
    steps=int(q.view(np.uint64))-int(q0.view(np.uint64))
    assert steps==shift,(shift,steps)
    row=dict(shift=shift,actual_input_ULPs=steps,QC_IN=float(q),CN_OUT=exact['CN_OUT'][12],
             QNN_OUT_sha256=hashlib.sha256(np.asarray(exact['CN_OUT'],np.float64).tobytes()).hexdigest(),
             source_output_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    rows.append(row);print(row,flush=True)
seal=dict(scope='Original22 same-case15 input-ULP controls; no tolerance or product changes',
          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),
          flags=flags,zero_shift_BYTE_baseline=True,rows=rows)
(out/'manifest.json').write_text(json.dumps(seal,indent=2)+'\n')
