from pathlib import Path
import json,os,subprocess,sys,datetime
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');T=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O=R/'v6_admissions';O.mkdir(exist_ok=True);rows=[]
def shell(text,*args):return subprocess.run(['bash','-c',text,'control',*map(str,args)],capture_output=True,text=True)
lib=T/'lib.sh'
for label,rootfree,used,data,forecast,expected in [('root_below',20*2**30-1,0,450,1,False),('root_at',20*2**30,0,450,1,True),('root_cap_at',25*2**30,8*2**30,450,1,True),('root_cap_over',25*2**30,8*2**30+1,450,1,False),('data_below',25*2**30,0,395,31,False),('data_at',25*2**30,0,396,31,True)]:
 text='source "$1"; df(){ printf "Avail\\n%s\\n" "$2"; };'
 # Fixed function arguments avoid confusing df's own argv with the fixture value.
 text='source "$1"; df(){ printf "Avail\\n'+str(rootfree)+'\\n"; }; du(){ printf "'+str(used)+'\\tROOT\\n"; }; w45_free_gib(){ echo '+str(data)+'; }; w45_disk_ok '+str(forecast)
 result=shell(text,lib);assert (result.returncode==0)==expected,(label,result.stdout,result.stderr);rows.append(dict(case=label,refused=result.returncode!=0))
# Semantic mutation: disable the ROOT byte cap in a private copy; the cap-over observation changes.
mut=O/'lib_without_root_cap.sh';mut.write_text(lib.read_text().replace('root_bytes <= W45R_ROOT_CAP_GIB * 1073741824','1 == 1'))
result=shell('source "$1"; df(){ printf "Avail\\n26843545600\\n"; }; du(){ printf "8589934593\\tROOT\\n"; }; w45_free_gib(){ echo 450; }; w45_disk_ok 1',mut);assert result.returncode==0;rows.append(dict(case='root_cap_removed_mutant',killed=True))
s=(T/'session_rw3.sh').read_text();start=s.index('run_slot() {');end=s.index('\n}\n',start)+3;fn=s[start:end]
for free,want in [(410,False),(411,True)]:
 text='source "$1"; SMOKE=0; P=unused; P0=unused; DEADLINE=$(( $(date +%s) + 9999 )); log(){ :; }; w45_free_gib(){ echo '+str(free)+'; }; w45_disk_ok(){ echo DISK_CALLED; return 1; };\n'+fn+'\nrun_slot O72 O 1 72 1200 23 0'
 r=shell(text,lib);assert ("DISK_CALLED" in r.stdout)==want,(free,r.returncode,r.stdout,r.stderr);rows.append(dict(case='O72_free_'+str(free),rc=r.returncode,unstarted=True))
mutfn=fn.replace('< 411','< 410');r=shell('source "$1"; SMOKE=0; P=x; P0=x; DEADLINE=$(( $(date +%s) + 9999 )); log(){ :; }; w45_free_gib(){ echo 410; }; w45_disk_ok(){ echo DISK_CALLED; return 1; };\n'+mutfn+'\nrun_slot O72 O 1 72 1200 23 0',lib);assert "DISK_CALLED" in r.stdout;rows.append(dict(case='O72_lowered_floor_mutant',killed=True))
for free_bytes,want in [(411*2**30-1,410),(411*2**30,411)]:
 r=shell('source "$1"; df(){ printf "Avail\\n'+str(free_bytes)+'\\n"; }; w45_free_gib <DATA_ROOT>',lib)
 assert int(r.stdout)==want,(free_bytes,r.stdout);rows.append(dict(case='floor_bytes_'+str(free_bytes),free_GiB=int(r.stdout)))
now=datetime.datetime.now(datetime.timezone.utc);iso=lambda t:t.strftime('%Y-%m-%dT%H:%M:%SZ');lo=iso(now-datetime.timedelta(seconds=1));hi=iso(now+datetime.timedelta(minutes=5));receipt=O/'NIGHTLY_DONE.json';good=dict(schema='v045-rc.nightly-done.v1',Nightly_DONE=True,by='manager',completed_utc=iso(now-datetime.timedelta(minutes=1)))
for label,delta,want in [('done',{},0),('notdone',dict(Nightly_DONE=False),1),('notmanager',dict(by='worker'),1),('doneafterwindow',dict(completed_utc=hi),1)]:
 receipt.write_text(json.dumps(good|delta));r=subprocess.run([sys.executable,'-I',str(T/'night_admission.py'),str(receipt),lo,hi],capture_output=True,text=True);assert (r.returncode==0)==(want==0),label;rows.append(dict(case=label,refused=r.returncode!=0))
# Fail-closed absent lease; no manager production state touched.
e=dict(os.environ,GA_GATE_ID='MISSING_CONTROL',CPU_LEASE_DIR=str(O),GA_END_UTC=hi)
r=subprocess.run([sys.executable,'-I',str(T/'cpu_lease_check.py')],env=e,capture_output=True,text=True);assert r.returncode!=0;rows.append(dict(case='missing_cpu_lease',refused=True))
(O/'ADMISSION_CONTROLS.json').write_text(json.dumps(dict(rows=rows,checks=len(rows),passed=True,scope='stdlib CPU private fixtures; actual guard functions, no GPU or production registry'),indent=2)+'\n');print('ADMISSION controls PASS',len(rows))
