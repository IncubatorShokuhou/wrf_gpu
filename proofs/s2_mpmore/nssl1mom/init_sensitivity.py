"""Original legacy19/21 compiler-init INF control, pristine bodies unchanged."""
from pathlib import Path
import importlib.util,hashlib,json,math,subprocess,sys
tree=Path(sys.argv[1]);lane=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore')
b=lane/'nssl1mom_init_build';b.mkdir(exist_ok=False)
proof=tree/'proofs/v034/oracle/nssl2mom'
gen=tree/'proofs/s2_mpmore/nssl1mom/build_reference.py'
spec=importlib.util.spec_from_file_location('caller',gen);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
driver=mod.make_driver(proof)
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F')
(b/'module.F').write_bytes(source.read_bytes());(b/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes());(b/'driver.F90').write_text(driver);(b/'namelist.input').write_text('')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=inf','-finit-integer=0','-finit-logical=false','-ffp-contract=off']
run=lambda args,**kw:subprocess.run(list(map(str,args)),check=True,timeout=120,cwd=b,**kw)
for file in ('stubs.f90','module.F','driver.F90'):run([fc,*flags,'-c',file])
run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'])
rows=[]
for v in (19,21):
 for c in range(1,18):
  text=b/f'v{v}_{c}.txt';target=b/f'v{v}_{c}.json'
  with text.open('w') as f:run([b/'oracle',c,'none',v],stdout=f)
  subprocess.run([sys.executable,proof/'dump_to_json_v2.py',text,target,'fp32'],check=True,timeout=30)
  alt=json.loads(target.read_text());ref=json.loads((tree/f'proofs/s2_mpmore/nssl1mom/fixtures/v{v}_fp32_{c}.json').read_text())
  changes=[section+'.'+key for section in ('columns','scalars') for key,value in alt[section].items() if value!=ref[section][key]]
  rows.append(dict(variant=v,case=c,changed_fields=changes))
result=dict(scope='Original19/21 REAL4 compiler-init INF vs ZERO; allreturnedcolumns/scalars; no body edits',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),flags=flags,rows=rows,changed_cases=sum(bool(x['changed_fields']) for x in rows))
(lane/'MPM1R-init-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print('INITCONTROL changedcases',result['changed_cases'],flush=True)
