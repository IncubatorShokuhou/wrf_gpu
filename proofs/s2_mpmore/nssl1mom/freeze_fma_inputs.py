"""Replay identical raw input bits into original FMA/noFMA modules."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
tree=Path(sys.argv[1]);lane=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmore');work=lane/'nssl1mom_fma_build';out=lane/'nssl1mom_fma_controls_frozen';out.mkdir()
proof=tree/'proofs/v034/oracle/nssl2mom';fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
names=['TH','QV','QC','QR','QI','QS','QH','QHL','CCW','CRW','CCI','CSW','CHW','CHL','CN','VHW','VHL','PII','P','DZ','W','DN']
cols=[name+'_IN' if i<17 else name for i,name in enumerate(names)]
rows=[]
for mode in ('noFMA','FMA'):
 b=work/mode;driver=(b/'driver.F90').read_text();assert driver.count('  INTEGER :: variant')==1 and driver.count('  ! save inputs')==1
 driver=driver.replace('  INTEGER :: variant','  INTEGER :: variant, frozen_unit')
 freeze="  OPEN(newunit=frozen_unit,file='fixed_input.bin',access='stream',form='unformatted',status='old')\n  READ(frozen_unit) "+','.join(names)+"\n  CLOSE(frozen_unit)\n"
 driver=driver.replace('  ! save inputs',freeze+'  ! save inputs');(b/'driver_frozen.F90').write_text(driver)
 flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero','-finit-integer=0','-finit-logical=false','-mfma','-ffp-contract='+('off' if mode=='noFMA' else 'fast')]
 for cmd in ([fc,*flags,'-c','driver_frozen.F90'],[fc,*flags,'stubs.o','module.o','driver_frozen.o','-o','oracle_frozen']):subprocess.run(cmd,cwd=b,check=True,timeout=90)
 for v in (19,21):
  base=json.loads((tree/f'proofs/s2_mpmore/nssl1mom/fixtures/v{v}_fp32_16.json').read_text())
  raw=b''.join(np.asarray(base['columns'][key],np.float32).tobytes() for key in cols)
  (b/'fixed_input.bin').write_bytes(raw)
  text=b/f'v{v}_frozen.txt';target=out/f'v{v}_{mode}.json'
  with text.open('w') as f:subprocess.run([b/'oracle_frozen','16','none',str(v)],cwd=b,stdout=f,check=True,timeout=60)
  subprocess.run([sys.executable,proof/'dump_to_json_v2.py',text,target,'fp32'],check=True,timeout=30)
  d=json.loads(target.read_text())
  assert all(d['columns'][key]==base['columns'][key] for key in cols),(mode,v,'frozeninputs')
  if mode=='noFMA':assert d['columns']==base['columns'] and d['scalars']==base['scalars'],(v,'noFMAbaseline')
  rows.append(dict(variant=v,mode=mode,input_bytes_sha256=hashlib.sha256(raw).hexdigest(),same_input_bits=True,caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),flags=flags,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),QC0=d['columns']['QC_OUT'][0]))
  print('FROZEN',rows[-1],flush=True)
(out/'manifest.json').write_text(json.dumps(dict(scope='Original REAL4 WRF compilerFMA vsnoFMA with identical rawbaseline inputs; bodyobjects reused, onlycaller frozenreadsadded',rows=rows),indent=2)+'\n')
