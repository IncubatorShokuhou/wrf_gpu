#!/usr/bin/env python3
"""Unsealed CPU-only reference view and guarded scoring. No staging copy and no original writes.
Prepare/score payload reads require --execute; default emits the exact metadata plan only.
"""
from pathlib import Path
import argparse,contextlib,datetime,hashlib,importlib.util,json,os,runpy,subprocess,sys,time
sys.dont_write_bytecode=True  # isolated child processes must not write into the pinned source
D=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');HERE=Path(__file__).resolve().parent
MAN=D/'tools_v6/PIN_MANIFEST.json'
W3P=Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate/arms0227/W3P/d6_72h/summary.json')
F20=Path('<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/WN3-0227/CPU_OUTPUTS.json')
GO=HERE/'REF_AUTHORITY.json'
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+chr(10))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_guard(snapshot):
 p=Path(snapshot)/'src/gpuwrf/assimilation/letkf/alisios_guard.py';s=importlib.util.spec_from_file_location('rc_cpu_ref_guard',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def targets():
 summary=json.loads(W3P.read_text());rows=[]
 for dom in ['d01','d02','d03']:
  pairs=summary['domains'][dom]['pairing']['pairs'];assert len(pairs)==73 and [p['lead_h'] for p in pairs]==list(range(73))
  rows.extend(dict(domain=dom,lead_h=p['lead_h'],source=p['cpu_file']) for p in pairs)
 assert len({Path(r['source']).name for r in rows})==219
 return rows

def allowed(ag,rows):
 return {r['source']:[Path(r['source']).name.split('_')[2].replace('-','')] for r in rows if ag.is_alisios_data(r['source'])}

def prepare(rows,sealed,stage,receipt,ag,fixture=False):
 stage=Path(stage);receipt=Path(receipt);assert stage.resolve().is_relative_to(D)
 assert not any(stage.resolve().is_relative_to(Path(r['source']).parent.resolve()) for r in rows)
 stage.mkdir(parents=True,exist_ok=True);receipt.mkdir(parents=True,exist_ok=True);index={};checked=0;bad=[];records=[]
 with ag.guarded_session(allowed(ag,rows),['20260227'],receipt/'guard_hash',purpose='RC approved original CPU-WRF reference SHA/read-only symlink view; no copy'):
  for row in rows: # one original file open at a time, bounded streaming hash
   src=Path(row['source']);assert src.is_file();h=hashlib.sha256();size=0
   with src.open('rb') as f:
    before=os.fstat(f.fileno())
    for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk);size+=len(chunk)
    after=os.fstat(f.fileno())
   assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'reference changed during read'
   value=h.hexdigest();name=src.name
   if name in sealed:
    checked+=1
    if sealed[name]!=value:bad.append(name)
   link=stage/name
   if link.is_symlink():assert link.readlink()==src,'existing view target differs'
   else:
    assert not link.exists(),'refuse overwrite of stage file';link.symlink_to(src)
   index.setdefault(row['domain'],{})[name]=dict(staged=str(link),sha256=value,lead_h=row['lead_h'],source=str(src),reference_bytes=size,in_place_read_only=True)
   records.append(dict(path=str(src),sha256=value,bytes=size))
 assert not bad and checked==(len(sealed) if fixture else 75),(checked,bad)
 result=dict(index=index,sealed24_checked=checked,sha_mismatch_vs_f20=bad,in_place=True,reference_copy_bytes=0,original_write_touch_chmod=False,sequential_reference_reads=True,files=records,fixture_transport=fixture)
 dump(receipt/'STAGED_INDEX.json',result);return result

def score_index(index_path,snapshot,gpu,out,execute_script=None,script_args=None,manifest_path=MAN):
 data=json.loads(Path(index_path).read_text());assert data['in_place'] and not data['sha_mismatch_vs_f20']
 fixture_ok=Path(manifest_path).resolve().is_relative_to(HERE/'validation') and Path(snapshot).resolve().is_relative_to(HERE/'validation')
 assert not data.get('fixture_transport') or fixture_ok,'fixture index cannot qualify a production score'
 rows=[dict(source=r['source']) for files in data['index'].values() for r in files.values()]
 stage=Path(next(iter(data['index']['d01'].values()))['staged']).parent;ag=load_guard(snapshot);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 target=Path(execute_script) if execute_script else Path(snapshot)/'scripts/wn3_score.py'
 assert target.resolve() in [(Path(snapshot)/'scripts/wn3_score.py').resolve(),(Path(snapshot)/'scripts/wn3_fast_compare.py').resolve()],'unreviewed scoring script'
 argv=list(script_args) if execute_script else ['--cpu-dir',str(stage),'--gpu-dir',str(gpu),'--out',str(out),'--hours','72','--sequential-domains','--rw3-floor']
 child='child' if execute_script else 'parent';Popen=subprocess.Popen
 def guarded_popen(cmd,*args,**kwargs):
  if isinstance(cmd,(list,tuple)) and len(cmd)>1 and Path(cmd[1]).resolve()==(Path(snapshot)/'scripts/wn3_fast_compare.py').resolve():
   domain=cmd[cmd.index('--domain')+1];guardout=out/('guard_child_'+domain)
   wrapped=[cmd[0],str(Path(__file__).resolve()),'child','--execute','--manifest',str(manifest_path),'--index',str(index_path),'--gpu-dir',str(gpu),'--out',str(guardout),'--',*cmd[2:]]
   return Popen(wrapped,*args,**kwargs)
  return Popen(cmd,*args,**kwargs)
 subprocess.Popen=guarded_popen
 try:
  with ag.guarded_session(allowed(ag,rows),['20260227'],out/('guard_'+child),purpose='Approved read-only CPU-WRF D6 original reference; parent/each comparator separately admitted'):
   sys.argv=[str(target),*argv]
   try:runpy.run_path(str(target),run_name='__main__')
   except SystemExit as e:
    if e.code not in (None,0):return int(e.code)
 finally:subprocess.Popen=Popen
 return 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','score','child']);p.add_argument('--execute',action='store_true');p.add_argument('--manifest',type=Path,default=MAN);p.add_argument('--index',type=Path);p.add_argument('--gpu-dir',type=Path);p.add_argument('--out',type=Path);a,extra=p.parse_known_args()
 assert os.environ.get('JAX_PLATFORMS')=='cpu';assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
 m=json.loads(a.manifest.read_text());snapshot=Path(m['snapshot']);pin=m['pin'];stage=D/('cpu72_'+pin[:12]);receipt=D/'receipts'/('cpu72_'+pin[:12]);rows=targets()
 if not a.execute:
  print(json.dumps(dict(mode='PLAN_ONLY',originals=219,stage=str(stage),index=str(receipt/'STAGED_INDEX.json'),copy_bytes=0,no_original_payload_reads=True,GUARD_parent_and_children=True)));return 0
 assert GO.is_file(),'approved carried authority missing'
 authority=json.loads(GO.read_text());assert authority['manager_utc']=='2026-10-28T12:47:13Z' and authority['in_place_read_only'] is True
 for key in ['path','metrics']:
  assert sha(m['rw3_floor'][key])==m['rw3_floor'][key+'_sha256' if key=='metrics' else 'sha256'],'frozen floor moved'
 if a.action=='prepare':
  # No copy. Current global GPU window forbids this heavy sequential hash pass; run it outside the window.
  assert not Path('/tmp/wrf_gpu2_gpu_window').exists(),'wait for GPU window to close before bulk reference hashing'
  ag=load_guard(snapshot);sealed={r['file']:r['sha256'] for r in json.loads(F20.read_text())};result=prepare(rows,sealed,stage,receipt,ag);result['GO_sha256']=sha(GO);dump(receipt/'STAGED_INDEX.json',result);print(json.dumps(dict(staged_links=219,checked24=result['sealed24_checked'],reference_copy_bytes=0)));return 0
 assert a.index and a.gpu_dir and a.out
 if a.action=='score':
  assert not Path('/tmp/wrf_gpu2_gpu_window').exists(),'CPU D6 scoring waits until GPU window ends'
  return score_index(a.index,snapshot,a.gpu_dir,a.out,manifest_path=a.manifest)
 extra=extra[1:] if extra[:1]==['--'] else extra
 return score_index(a.index,snapshot,a.gpu_dir,a.out,execute_script=snapshot/'scripts/wn3_fast_compare.py',script_args=extra,manifest_path=a.manifest)
if __name__=='__main__':raise SystemExit(main())
