"""Same-input original REAL4 case16, compiler contraction on/off control."""
from pathlib import Path
import importlib.util,hashlib,json,math,subprocess,sys
tree=Path(sys.argv[1]);lane=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore')
work=lane/'nssl1mom_fma_build';out=lane/'nssl1mom_fma_controls';work.mkdir();out.mkdir()
proof=tree/'proofs/v034/oracle/nssl2mom';gen=tree/'proofs/s2_mpmore/nssl1mom/build_reference.py'
spec=importlib.util.spec_from_file_location('caller',gen);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);driver=m.make_driver(proof)
source=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F');fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
run=lambda args,**kw:subprocess.run(list(map(str,args)),check=True,timeout=120,**kw)
rows=[]
for mode in ('noFMA','FMA'):
 b=work/mode;b.mkdir();(b/'module.F').write_bytes(source.read_bytes());(b/'driver.F90').write_text(driver);(b/'stubs.f90').write_bytes((proof/'nssl2mom_stub_modules.f90').read_bytes());(b/'namelist.input').write_text('')
 flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false','-mfma','-ffp-contract='+('off' if mode=='noFMA' else 'fast')]
 for file in ('stubs.f90','module.F','driver.F90'):run([fc,*flags,'-c',file],cwd=b)
 run([fc,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b)
 for v in (19,21):
  text=b/f'v{v}.txt';target=out/f'v{v}_{mode}.json'
  with text.open('w') as f:run([b/'oracle',16,'none',v],cwd=b,stdout=f)
  run([sys.executable,proof/'dump_to_json_v2.py',text,target,'fp32'])
  d=json.loads(target.read_text());base=json.loads((tree/f'proofs/s2_mpmore/nssl1mom/fixtures/v{v}_fp32_16.json').read_text())
  if mode=='noFMA':
   assert d['columns']==base['columns'] and d['scalars']==base['scalars'],(v,'matched noFMA differs')
  assert all(math.isfinite(x) for values in d['columns'].values() for x in values)
  rows.append(dict(variant=v,mode=mode,flags=flags,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),QC0=d['columns']['QC_OUT'][0],same_input_columns=all(d['columns'][k]==base['columns'][k] for k in base['columns'] if k.endswith('_IN'))))
  print('CONTROL',rows[-1],flush=True)
manifest=dict(scope='Original WRf4.7.1 one-moment case16 REAL4 compilerFMA vsnoFMA; source/inputs/caller same, no reassociation flags',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),rows=rows)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
