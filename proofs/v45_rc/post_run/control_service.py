"""Actual synchronous systemd CPU scope, inert controller only; no model/reference/raw deletion."""
from pathlib import Path
import json,os,shutil,subprocess,time
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');O=R/'POST_SERVICE_CONTROL'/str(time.time_ns());O.mkdir(parents=True,exist_ok=True);shutil.copy2(R/'post_run/launch.sh',O/'launch.sh')
(O/'controller.py').write_text('''import json,os,sys\nfrom pathlib import Path\nout=Path(__file__).with_name('SERVICE_RESULT.json')\ncg=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().split(':',2)[2].strip())\nout.write_text(json.dumps(dict(scope='INERT SYSTEMD HANDOFF ONLY',JAX_cpu=os.environ.get('JAX_PLATFORMS')=='cpu',cpus=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),memory_max=(cg/'memory.max').read_text().strip(),swap_max=(cg/'memory.swap.max').read_text().strip(),args=sys.argv[1:],original_payload_reads=0,raw_deletes=0)))\n''')
p=subprocess.run(['bash',str(O/'launch.sh'),str(O/'INERT_MANIFEST.json'),'INERT_WINDOW','2026-10-29T05:30:00Z'],capture_output=True,text=True,timeout=20);(O/'launch.log').write_text(p.stdout+p.stderr);assert p.returncode==0,(p.returncode,p.stderr)
for _ in range(100):
 if(O/'SERVICE_RESULT.json').exists():break
 time.sleep(.05)
j=json.loads((O/'SERVICE_RESULT.json').read_text());assert j['JAX_cpu']and j['cpus']==[12,28]and j['nice']==19 and int(j['memory_max'])==8*2**30 and j['swap_max']=='0';(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,result=j,actual_synchronous_CPU_scope=True,real_GPU_or_reference_or_raw_operations=0),indent=2)+'\n');print('POST real synchronous CPU scope PASS:8G/swap0/12,28/nice19/JAXcpu; inert controller only')
