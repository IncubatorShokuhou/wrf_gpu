"""Receipt-first operations on the literal owned RC run list; never follow links."""
import hashlib,json,os,re,shutil,stat,time
from pathlib import Path
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-rc');R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');GIB=2**30

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2)+'\n')
def under(path,roots):return any(path==r or path.is_relative_to(r)for r in roots)
def validate_roots(roots):
 assert len(roots)==6 and len(set(roots))==6
 for p in roots:
  assert p.is_absolute()and p.is_dir()and not p.is_symlink()and p.resolve().is_relative_to(D)
  assert p.name.startswith(('session_254c48237452_gpu_v6','prime_swiss_','prime_prod_','prime_wn3_','rw_warm_','rw3_'))
 return roots

def readers(roots):
 found=[]
 for proc in Path('/proc').glob('[0-9]*'):
  for p in [proc/'cwd',*list((proc/'fd').glob('*'))]:
   try:t=Path(os.readlink(p).removesuffix(' (deleted)'))
   except(OSError,ValueError):continue
   if t.is_absolute()and under(t,roots):found.append(dict(pid=int(proc.name),path=str(p),target=str(t)))
 return found

def dependencies(roots,out,scan_roots=None):
 hits=[];links=[];count=0;scan=scan_roots if scan_roots is not None else [Path('<USER_HOME>/wrf_gpu2_lanes'),Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel'),Path('<DATA_ROOT>/wrf_gpu2/worktrees')]
 needles=[str(p).encode()for p in roots];patterns=[re.compile(re.escape(n)+rb'''(?=[/"'\s]|$)''')for n in needles];prune={'.git','__pycache__','jax','cuda','triton','autotune','wrfout','data','cases','dump','node_modules','.venv'}
 for base in scan:
  for d,dirs,files in os.walk(base,followlinks=False):
   here=Path(d)
   if under(here,roots):dirs[:]=[];continue
   for name in dirs+files:
    p=here/name
    if p.is_symlink():
     try:t=p.resolve()
     except(OSError,RuntimeError):continue
     if under(t,roots):links.append(dict(path=str(p),target=str(t)))
   dirs[:]=[n for n in dirs if n not in prune and not (here/n).is_symlink()]
   for name in files:
    p=here/name
    if p.suffix not in {'.sh','.py','.json','.md','.txt','.tsv'} or p.is_symlink():continue
    if p.name in {'MAILBOX.md','EXPERIMENTS.tsv','INBOX.md','FINDINGS.md'}:continue  # historical ledgers, not queued consumers
    try:recent=p.stat().st_mtime>=time.time()-24*3600
    except OSError:continue
    if not recent:continue  # current queued material, beyond E246's eight-hour minimum
    try:
     with p.open('rb')as f:
      prev=b'';matched=False
      while b:=f.read(1<<20):
       data=prev+b
       if any(n in data for n in needles) and any(pat.search(data)for pat in patterns):matched=True;break
       prev=data[-4096:]
     count+=1
     if count%10000==0:save(out.with_suffix('.progress.json'),dict(files_scanned=count))
    except(OSError,ValueError):continue
    if matched:
     from dependency_policy import expected
     current=[out.parent/'CLOSED_CURRENT_RECEIPTS.json']
     ok,reason=expected(p,Path(__file__).with_name('DEPENDENCY_POLICY.json'),current)
     hits.append(dict(path=str(p),expected=ok,reason=reason))

 save(out,dict(files_scanned=count,matches=hits,links_to_deleted_roots=links,scope='all lane/manager scripts/manifests/plans; source text only; no binary/NSYS string reads'))
 assert not links,'external symlink reader would become dangling'
 assert all(h['expected']for h in hits),'queued/external dependency requires manager resolution'
 return count

def preserve(roots,out):
 rows=[];payload=[];dest=out/'preserved';dest.mkdir(parents=True,exist_ok=True)
 for root in roots:
  for d,dirs,files in os.walk(root,topdown=True,followlinks=False):
   here=Path(d)
   for name in dirs:
    p=here/name
    if not p.is_symlink():p.chmod(stat.S_IMODE(p.stat().st_mode)|0o700)  # own disposable overlay work dirs only
   for name in files:
    p=here/name;rel=p.relative_to(root)
    if p.is_symlink():rows.append(dict(path=str(p),link=os.readlink(p)));continue
    assert p.is_file(),'special path cannot be deleted'
    digest=sha(p);record=dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
    if any(k in rel.parts for k in ['wrfout','tmp','scratch','layers']):payload.append(record)
    else:
     q=dest/root.name/rel;q.parent.mkdir(parents=True,exist_ok=True)
     assert os.statvfs(out).f_bavail*os.statvfs(out).f_frsize-p.stat().st_size>=20*GIB,'ROOT preservation floor'
     shutil.copy2(p,q);q.chmod(0o600);assert sha(q)==digest
     record['preserved']=str(q);rows.append(record)
 save(out/'PRESERVATION.json',dict(keepers=rows,payload=payload,raw_retained_until_required_score_complete=True));return rows,payload

def remove_literal(roots,out,proof,apply=False):
 assert proof.get('required_score_complete')is True,'no raw deletion before required D6/RW3 evidence'
 validate_roots(roots);assert not readers(roots),'owned/foreign reader remains'
 dependencies(roots,out/'DEPENDENCIES.json');keep,raw=preserve(roots,out)
 save(out/'DELETE_SHA_SIZE.json',dict(roots=[str(p)for p in roots],keepers=keep,payload=raw,apply=apply))
 with (out/'DELETE_SHA_SIZE.json').open('rb')as f:os.fsync(f.fileno())
 if not apply:return dict(state='DRY_RUN',paths=[str(p)for p in roots])
 for root in roots:shutil.rmtree(root)
 residual=0
 for p in D.rglob('*'):
  if p.is_file()and not p.is_symlink():residual+=p.stat().st_size
 free=os.statvfs('<DATA_ROOT>');free=free.f_bavail*free.f_frsize
 result=dict(state='OWN_DATA_REMOVED',own_DATA_bytes=residual,DATA_free_bytes=free,meets_404_5GiB=free>=404.5*GIB)
 save(out/'DATA_RECOVERY.json',result);assert residual<=GIB
 return result

def registry_deleted(out,manifest):
 """Authorized terminal retention updates only; no new start/end and no backfill."""
 import importlib.util,sys
 m=json.loads(Path(manifest).read_text());script=Path(m['snapshot'])/'scripts/alisios_registry.py'
 spec=importlib.util.spec_from_file_location('post_registry',script);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 reg=mod.Registry();record=json.loads((out/'PRESERVATION.json').read_text());updates=[]
 for item in record['keepers']:
  if Path(item['path']).name!='REGISTRY_START.json':continue
  j=json.loads(Path(item['preserved']).read_text());rid=j['run_id']
  if rid in updates:continue
  reg.update(dict(run_id=rid,by='v45-rc',raw_deleted=True,extract=[str(out/'DELETE_SHA_SIZE.json')+'::'+sha(out/'DELETE_SHA_SIZE.json')],notes='Required CPU scores/extraction complete; literal own run roots removed, all receipts preserved'),authorization='manager2026-10-28T22:07+22:15+22:19 post-extract cleanup')
  updates.append(rid)
 save(out/'REGISTRY_UPDATES.json',updates)
