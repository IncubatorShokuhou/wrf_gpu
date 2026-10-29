"""Remove one completed own consumer layer after SHA/size receipts; never follow links."""
from pathlib import Path
import hashlib,json,os,re,shutil,stat,sys
root,layer,arm,scope=map(Path,sys.argv[1:5]);failed="--failed" in sys.argv[5:];owned=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc')
assert root.resolve().is_relative_to(owned) or root.resolve().is_relative_to(Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-rc'))
assert layer.parent==root/'layers' and layer.is_dir() and not layer.is_symlink()
timer=json.loads((arm/'timer.json').read_text());assert timer['teardown']['all_reaped'] and (timer['rc']==0 or failed)
lines=[l for l in scope.read_text().splitlines()if l.startswith('BUNDLE_CONSUMER rc=')];assert lines
assert re.search(r'^BUNDLE_CONSUMER rc='+str(timer['rc'])+' ',lines[-1]) and re.search(r' compiled_new=0 ',lines[-1]),'retain new-compile layer for diagnosis'
receipt=arm/'LAYER_CLEANUP.json';rows=[];saved=arm/'LAYER_RECEIPTS';saved.mkdir(exist_ok=True)
# Overlayfs creates a mode000 work/work directory. Only these own disposable layer dirs gain owner traversal.
for d,dirs,files in os.walk(layer,topdown=True,followlinks=False):
 d=Path(d)
 for name in dirs:
  p=d/name
  if not p.is_symlink():p.chmod(stat.S_IMODE(p.stat().st_mode)|0o700)
 for name in files:
  p=d/name
  if p.is_symlink():rows.append(dict(path=str(p),link=os.readlink(p)));continue
  if not p.is_file():raise RuntimeError('retain special consumer file '+str(p))
  with p.open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest))
  if p.is_relative_to(layer/'runs'):
   q=saved/p.relative_to(layer/'runs');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
receipt.write_text(json.dumps(dict(layer=str(layer),consumer=lines[-1],all_children_reaped=True,files=rows),indent=2)+'\n')
shutil.rmtree(layer)
print('LAYER_CLEANED',layer,'files',len(rows))
