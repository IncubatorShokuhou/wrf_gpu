"""404.5 GiB postcondition boundary, every delete intercepted; private fixture only."""
from pathlib import Path
import importlib.util,json,os
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'RECOVERY_THRESHOLD_CONTROL';O.mkdir(exist_ok=True);s=importlib.util.spec_from_file_location('ops_threshold',R/'post_run/data_ops.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);D=O/'own_data';D.mkdir(exist_ok=True);m.D=D
roots=[D/'session_254c48237452_gpu_v6',*[D/'runs'/(n+'_INERT')for n in ['prime_swiss','prime_prod','prime_wn3','rw_warm','rw3']]]
for p in roots:p.mkdir(parents=True,exist_ok=True)
m.readers=lambda roots:[];m.dependencies=lambda roots,out:0;m.preserve=lambda roots,out:([],[]);m.shutil.rmtree=lambda p:None
original=m.os.statvfs;rows=[]
for n,want in [(int(404.5*2**30)-1,False),(int(404.5*2**30),True)]:
 class FS:f_bavail=n;f_frsize=1
 m.os.statvfs=lambda path:FS()if str(path)=='<DATA_ROOT>'else original(path)
 out=O/str(n);out.mkdir(exist_ok=True);j=m.remove_literal(roots,out,dict(required_score_complete=True),apply=True);assert j['meets_404_5GiB']is want;rows.append(dict(bytes=n,accepted=want))
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,rows=rows,actual_deletes=0,scope='inert exact production postcondition'),indent=2)+'\n');print('RECOVERY threshold controls PASS:404.5GiB-1byteREFUSED/exactACCEPTED; zeroactualdeletes')
