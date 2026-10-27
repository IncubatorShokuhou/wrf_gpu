from pathlib import Path
import os,json,collections,subprocess as sp,importlib.util,time
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc');T=D/'registry_snapshot';O=D/'registry_affected';O.mkdir(exist_ok=True);tmp=O/'tmp';tmp.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('runner',T/'proofs/w42_bundle042/tools/rc_suite.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class Pause:
 def exists(self):return any(Path(x).exists() for x in ('/tmp/wrf_gpu2_quiet','/tmp/wrf_gpu2_gpu_window'))
 def read_text(self):return 'QUIET/GPU_WINDOW'
r.QUIET=Pause()
while r.QUIET.exists():time.sleep(5)
env=r.base_env(T,tmp)
for k in list(env):
 if k.startswith('GPUWRF_'):env.pop(k)
env.update(JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS='0',JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE='false')
selectors=['tests/v025/restart/test_runtime_data_identity.py','tests/test_runtime_assets.py']
cmd=r.pytest_cmd(T,Path('<USER_HOME>/wrf_gpu2_lanes/w44-rc/pytest'),O/'pytest.xml',selectors,extra=('--tb=short',))
cmd=['bash','-c','if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi; exec "$@"','reg-affected',*cmd]
res=r.run_paused(cmd,T,env,O/'pytest.log',600,print);nodes=r.parse_xml(O/'pytest.xml');res.update(selectors=selectors,head=sp.check_output(['git','rev-parse','HEAD'],cwd=T,text=True).strip(),nodes=nodes,counts=dict(collections.Counter(x[0] for x in nodes.values())))
(O/'result.json').write_text(json.dumps(res,indent=2)+'\n');print(res['rc'],res['counts'])
while r.QUIET.exists():time.sleep(5)
import shutil
shutil.rmtree(tmp);shutil.rmtree(D/'pytest_home/pytest');(D/'pytest_home/pytest').mkdir()
raise SystemExit(res['rc'])
