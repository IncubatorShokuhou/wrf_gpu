from pathlib import Path
import hashlib,json,os
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');a=base/'TP32/50a';b=base/'TP32/50b'
ra=json.loads((a/'receipt.json').read_text());rb=json.loads((b/'receipt.json').read_text())
assert ra['status']==rb['status']=='COMPLETE' and ra['source_executable_sha256']==rb['source_executable_sha256'] and ra['input_sha256']==rb['input_sha256']
rows=[]
for root,r in [(a,ra),(b,rb)]:
 raw=(root/'outputs.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==r['output_manifest_sha256'];rows.append(json.loads(raw))
assert len(rows[0])==len(rows[1])==25
fields=0;bad=[];files=0
for x,y in zip(*rows):
 assert x['file']==y['file'] and x['times']==y['times'] and x['fields'].keys()==y['fields'].keys()
 files+=x['sha256']==y['sha256']
 for name in x['fields']:
  fields+=1
  if x['fields'][name]!=y['fields'][name]:bad.append({'file':x['file'],'field':name,'a':x['fields'][name],'b':y['fields'][name]})
r={'scope':'full24h original compilerZERO repeatability; no port or GPU execution','source_executable_sha256':ra['source_executable_sha256'],'input_sha256':ra['input_sha256'],'arms':[ra,rb],'frames':25,'paired_fields':fields,'whole_files_BYTE':files,'field_mismatches':bad,'all_fields_BYTE':not bad,'cache':'NONE','duplicate_cleanup':'PENDING'}
p=base/'TP44_repeat.json';p.write_text(json.dumps(r,indent=2)+'\n');print('REPEAT',fields,'fields; BYTE files',files,'bad',len(bad),flush=True)
assert not bad
if Path('/tmp/wrf_gpu2_quiet').exists():raise RuntimeError('QUIET before duplicate cleanup')
for row in rows[1]:
 q=b/row['file'];assert q.parent==b and q.is_file() and not q.is_symlink();q.unlink()
q=b/'wrf.exe'
if q.is_file() and not q.is_symlink():q.unlink()
r['duplicate_cleanup']='own50b duplicate wrfouts/privateexe deleted after sealed BYTE proof; first reference retained';p.write_text(json.dumps(r,indent=2)+'\n')
