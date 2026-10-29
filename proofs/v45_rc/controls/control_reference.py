"""Inert transport test only: full-floor argv and independent parent/child guard seams; no science claim."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from netCDF4 import Dataset
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');T=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O=R/'v6_reference_controls';V=O/'validation';SN=V/'snapshot';V.mkdir(parents=True,exist_ok=True);(SN/'scripts').mkdir(parents=True,exist_ok=True);(SN/'src/gpuwrf/assimilation/letkf').mkdir(parents=True,exist_ok=True)
for name in ['reference_io.py','REF_AUTHORITY.json']:shutil.copy2(T/name,O/name)
old=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/CPU_REF_ADAPTER/validation/snapshot')
for name in ['wn3_score.py','wn3_fast_compare.py']:shutil.copy2(old/'scripts'/name,SN/'scripts'/name)
guard=Path('src/gpuwrf/assimilation/letkf/alisios_guard.py');shutil.copy2(R/'archive'/guard,SN/guard)
p=SN/'scripts/wn3_score.py';s=p.read_text().replace("p.add_argument('--sequential-domains',action='store_true');","p.add_argument('--sequential-domains',action='store_true');p.add_argument('--rw3-floor',action='store_true',required=True);").replace('assert a.sequential_domains','assert a.sequential_domains and a.rw3_floor').replace("opened=count,scope=", "opened=count,full_floor_argv=a.rw3_floor,scope=");p.write_text(s)
ST=V/'stage';ST.mkdir(exist_ok=True);f=V/'fixture.nc'
with Dataset(f,'w')as d:d.createDimension('n',1);d.createVariable('fixture_value','i4',('n',))[:]=[17]
h=hashlib.sha256(f.read_bytes()).hexdigest();index={}
for domain in ['d01','d02','d03']:
 index[domain]={}
 for lead in range(73):
  when=datetime.datetime(2026,2,28)+datetime.timedelta(hours=lead);name='wrfout_'+domain+'_'+when.strftime('%Y-%m-%d_%H:%M:%S');path=ST/name
  if not path.exists():path.symlink_to(f)
  index[domain][name]=dict(staged=str(path),source=str(path),sha256=h,lead_h=lead)
IDX=V/'STAGED_INDEX.json';IDX.write_text(json.dumps(dict(index=index,in_place=True,fixture_transport=True,sha_mismatch_vs_f20=[],sealed24_checked=75)))
m=json.loads((R/'tools_v6/PIN_MANIFEST.json').read_text());m['snapshot']=str(SN);m['pin']='INERT_TRANSPORT_V045';MAN=V/'PIN_MANIFEST.json';MAN.write_text(json.dumps(m))
cmd=[sys.executable,'-B',str(O/'reference_io.py'),'score','--execute','--manifest',str(MAN),'--index',str(IDX),'--gpu-dir',str(V/'not_a_GPU_run'),'--out',str(V/'score')]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(O/'TRANSPORT.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(r.returncode,r.stderr[-1500:])
parent=json.loads((V/'score/PARENT_FIXTURE.json').read_text());assert parent['opened']==219 and parent['full_floor_argv'] is True
for domain in ['d01','d02','d03']:assert json.loads((V/'score'/(domain+'.json')).read_text())['opened']==73
sessions=list((V/'score').rglob('guarded_session.json'));assert len(sessions)==4,(len(sessions),sessions)
# Mutation: removing --rw3-floor must prevent even this inert parent scoring path.
p=O/'reference_io.py';s=p.read_text();p.write_text(s.replace(",'--rw3-floor'",''));r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);assert r.returncode!=0;(O/'MUTANT.log').write_text(r.stdout+r.stderr);shutil.copy2(T/'reference_io.py',p)
result=dict(scope='INERT transport ONLY; no CPU-WRF/GPU science or fidelity result',real_guard_source_exact=True,code_sha256=hashlib.sha256((T/'reference_io.py').read_bytes()).hexdigest(),parent_private_netcdf_opens=219,child_private_netcdf_opens=219,separate_guard_sessions=4,full_floor_argv_required=True,omitted_floor_mutant_killed=True,original_CPU_payload_reads=0,original_CPU_writes=0)
(O/'TRANSPORT_CONTROLS.json').write_text(json.dumps(result,indent=2)+'\n');print('reference transport PASS parent219+children219 /4guards; missingfloor mutant killed')
# Exact inert fixture cleanup; keep receipts, scripts, index and SHA/size.
rows=[]
for path in ST.iterdir():assert path.is_symlink() and path.resolve()==f.resolve();path.unlink()
ST.rmdir();rows.append(dict(path=str(f),bytes=f.stat().st_size,sha256=h));f.unlink();(O/'FIXTURE_CLEANUP.json').write_text(json.dumps(rows,indent=2)+'\n')
