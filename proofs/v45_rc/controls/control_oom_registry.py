import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');T=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6');O=R/'v6_oom_control';O.mkdir(exist_ok=True);(O/'runs/N72').mkdir(parents=True,exist_ok=True)
# Inert actual START then deliberate SIGKILL of child; supervisor is the real changed tool.
script=O/'killed_child.py';script.write_text('''import os,sys,signal\nfrom pathlib import Path\nsys.dont_write_bytecode=True\nsys.path.insert(0,sys.argv[1])\nimport registry_hook\nregistry_hook.begin(['bash','/inert/arm.sh','unused','N72',sys.argv[2],sys.argv[3],'--','--hours','72'])\nos.kill(os.getpid(),signal.SIGKILL)\n''')
m=json.load((R/'tools_v6/PIN_MANIFEST.json').open());e=dict(os.environ,JAX_PLATFORMS='cpu',WRF_GPU2_SEQ_SMOKE='1',W45R_MODE='smoke',W45R_MANIFEST_PATH=str(R/'tools_v6/PIN_MANIFEST.json'),W45R_PIN_MANIFEST=str(R/'tools_v6/PIN_MANIFEST.json'),W45R_PINNED_SOURCE=str(R/'archive'),W45R_PRIVATE_ATLAS=str(R/'smoke_v6/private_atlas'),W45R_ENTRY_NAME='rw3',W45R_RUN=str(O))
cmd=[sys.executable,'-B','-I',str(T/'arm_supervisor.py'),'--seconds','10','--receipt',str(O/'LIFECYCLE.json'),'--group-file',str(O/'GROUP.json'),'--lock-file',str(O/'private.lock'),'--',sys.executable,'-B',str(script),str(T),m['inputs']['WN3_0227'],str(O/'runs/N72')]
r=subprocess.run(cmd,env=e,capture_output=True,text=True,timeout=20);(O/'control.log').write_text(r.stdout+r.stderr);assert r.returncode==137,(r.returncode,r.stdout,r.stderr)
j=json.loads((O/'LIFECYCLE.json').read_text());assert j['cleanup']['all_children_reaped'] and len(j['registry_terminal_after_teardown'])==1
end=json.loads((O/'runs/N72/REGISTRY_END.json').read_text());assert end['status']=='ABORTED' and end['rc']==137
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,actual_child_SIGKILL=True,external_supervisor_alive=True,all_reaped=True,one_terminal_end=True,backfill=False),indent=2)+'\n');print('SIGKILL/owned-OOM terminal fallback PASS, one ABORTED after reaping')
