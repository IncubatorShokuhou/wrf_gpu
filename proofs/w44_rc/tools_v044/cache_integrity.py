"""Byte inventory for the explicitly unsealed prime -> E41 -> warm transition."""
from pathlib import Path
import hashlib,json,sys

def capture(root):
 root=Path(root).resolve();assert root.is_dir();files={}
 for p in sorted(root.rglob('*')):
  name=p.relative_to(root).as_posix()
  if p.is_symlink():
   assert p.resolve().is_relative_to(root), 'cache link escapes private root'
   files[name]=dict(kind='link',target=str(p.readlink()))
  elif p.is_file():
   with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
   files[name]=dict(kind='file',bytes=p.stat().st_size,sha256=d)
  elif p.is_dir():files[name]=dict(kind='directory')
  else:raise AssertionError('unsupported cache entry')
 return dict(schema='w44-rc.unsealed-cache-integrity.v1',cache=str(root),files=files)

def main():
 mode,cache,before,*rest=sys.argv[1:];actual=capture(cache);before=Path(before)
 if mode=='snapshot':
  assert not before.resolve().is_relative_to(Path(cache).resolve())
  before.write_text(json.dumps(actual,sort_keys=True,indent=1)+'\n');print('UNSEALED_CACHE_SNAPSHOT',len(actual['files']));return
 assert mode=='check' and len(rest)==1
 expected=json.loads(before.read_text());old=expected['files'];new=actual['files']
 diff=dict(missing=sorted(set(old)-set(new)),new=sorted(set(new)-set(old)),changed=sorted(k for k in old.keys()&new.keys() if old[k]!=new[k]))
 ok=expected['schema']==actual['schema'] and expected['cache']==actual['cache'] and not any(diff.values())
 report=dict(status='UNCHANGED' if ok else 'CHANGED',diff=diff,cache=actual['cache'],entries=len(new))
 Path(rest[0]).write_text(json.dumps(report,indent=1)+'\n')
 print(('UNSEALED_CACHE_OK' if ok else 'UNSEALED_CACHE_CHANGED')+' missing='+str(len(diff['missing']))+' changed='+str(len(diff['changed']))+' new='+str(len(diff['new'])))
 if not ok:raise SystemExit(6)
if __name__=='__main__':main()
