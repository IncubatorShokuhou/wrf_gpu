from pathlib import Path
import hashlib,json,subprocess
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');P=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6/arm.sh');s=P.read_text();old=subprocess.check_output(['git','show','5e2bb4ce1e:proofs/v45_rc/tools_v6/arm.sh'],text=True)
def extract(v):a=v.index('for _secret in $(compgen');b=v.index('unset _secret JAX_PLATFORM_NAME',a);return v[a:b]
checks='''
[[ ${GPUWRF_GPU_LOCK_TOKEN:-} == INERT_CAPABILITY ]] || exit 41
[[ ${GPUWRF_GPU_LOCK_HELD:-} == 1 ]] || exit 42
[[ ${GPUWRF_GPU_LOCK_FD:-} == 9 ]] || exit 43
for name in EARTHDATA_PASSWORD RELEASE_APIKEY SECRET_VALUE AUTH_TOKEN HTTP_COOKIE GPUWRF_OTHER_TOKEN; do [[ -z ${!name+x} ]] || exit 44; done
[[ $SAFE_CONTROL == OK ]] || exit 45
'''
e=dict(PATH='/usr/bin:/bin',GPUWRF_GPU_LOCK_TOKEN='INERT_CAPABILITY',GPUWRF_GPU_LOCK_HELD='1',GPUWRF_GPU_LOCK_FD='9',SAFE_CONTROL='OK',**{k:'INERT_CREDENTIAL'for k in ['EARTHDATA_PASSWORD','RELEASE_APIKEY','SECRET_VALUE','AUTH_TOKEN','HTTP_COOKIE','GPUWRF_OTHER_TOKEN']})
rows=[]
for name,v,want in [('actual_old_filter',old,41),('actual_fixed_filter',s,0),('preservation_removed_mutant',s.replace('[[ $_secret == GPUWRF_GPU_LOCK_TOKEN ]] || ',''),41)]:
 r=subprocess.run(['bash','-c',extract(v)+checks],env=e,capture_output=True,text=True);assert r.returncode==want,(name,r.returncode);rows.append(dict(case=name,rc=r.returncode,expected=want))
O=R/'TOKEN_FIX';O.mkdir(exist_ok=True);(O/'CONTROLS.json').write_text(json.dumps(dict(PASS=True,rows=rows,exact_source_loop=True,credentials_still_stripped=True,official_nonsecret_capability_preserved=True,no_environment_values_printed=True,no_JAX_or_GPU_import=True,arm_sha256=hashlib.sha256(P.read_bytes()).hexdigest()),indent=2)+'\n');print('TOKEN filter controls PASS: reproduced rc41, fixed rc0, mutant rc41; credential removal unchanged')
