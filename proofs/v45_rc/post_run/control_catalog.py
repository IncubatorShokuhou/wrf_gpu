from pathlib import Path
import importlib.util,json,hashlib
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'POST_CATALOG_CONTROL';O.mkdir(exist_ok=True);s=importlib.util.spec_from_file_location('pcatalog',R/'post_run/controller.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);D=O/'private_data';D.mkdir(exist_ok=True);m.D=D
pin='254c4823745275e3e9d397ae4d1db0fd35ec361d';T=O/'tools';T.mkdir(exist_ok=True);(T/'SHA256SUMS').write_text('INERT_TOOLSET');MAN=T/'PIN_MANIFEST.json';B=O/'bundle';(B/'meta/primes').mkdir(parents=True,exist_ok=True);S=D/('session_'+pin[:12]+'_gpu_v6');A=S/'runs/A1_swiss';A.mkdir(parents=True,exist_ok=True)
proof=A/'proof.json';proof.write_text('{"scope":"INERT CATALOG CONTROL ONLY"}');sums=A/'wrfout_sha.txt';sums.write_text('INERT_METADATA_ONLY')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(MAN).write_text(json.dumps(dict(pin=pin,tools_dir=str(T),bundles=dict(prod=dict(dir=str(B))))))
(B/'meta/primes/A1_swiss.json').write_text(json.dumps(dict(pin=pin,mode='gpu',backend=dict(platform='gpu'),arm=str(A),proof=str(proof),proof_sha256=sha(proof),sha_list=str(sums),sha_list_sha256=sha(sums),scope='INERT structural fixture; no actual model/GPU/scientific result')))
for name in ['prime_swiss','prime_prod','prime_wn3','rw_warm','rw3']:
 run=D/'runs'/(name+'_INERT');run.mkdir(parents=True,exist_ok=True);j=dict(entry=name,mode='gpu',manifest=str(MAN),window=['OLD'if name=='prime_swiss'else'NEW'],tools_sums_sha256='OLD_CHECKSUM'if name=='prime_swiss'else sha(T/'SHA256SUMS'))
 (run/'ENTRY.json').write_text(json.dumps(j));(run/'LIFECYCLE.json').write_text(json.dumps(dict(cleanup=dict(all_children_reaped=True,remaining_pids=[]))))
_,entries,short,rw3=m.discover(MAN,'NEW');assert len(entries)==5 and short==S and entries['prime_swiss']['entry']['window']==['OLD']
rows=[dict(case='paid_swiss_prior_window_and_header',accepted=True)]
p=D/'runs/prime_prod_INERT/ENTRY.json';old=p.read_bytes();j=json.loads(old);j['tools_sums_sha256']='WRONG';p.write_text(json.dumps(j));bad=False
try:m.discover(MAN,'NEW')
except AssertionError:bad=True
finally:p.write_bytes(old)
assert bad;rows.append(dict(case='noncarried_tool_mismatch',refused=True))
p=D/'runs/rw3_INERT/LIFECYCLE.json';old=p.read_bytes();j=json.loads(old);j['cleanup']['all_children_reaped']=False;p.write_text(json.dumps(j));bad=False
try:m.discover(MAN,'NEW')
except AssertionError:bad=True
finally:p.write_bytes(old)
assert bad;rows.append(dict(case='live_reader_lifecycle',refused=True))
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,rows=rows,scope='INERT catalog only; no model/GPU/reference reads/deletes'),indent=2)+'\n');print('POST catalog controls PASS3: exact paidSwiss carry; mismatched live tuple/lifecycle refused')
