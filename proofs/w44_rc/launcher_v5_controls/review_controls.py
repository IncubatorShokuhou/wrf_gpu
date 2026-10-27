"""Inert transport controls of actual launcher shell code; no models/GPU."""
import hashlib,json,os,re,subprocess,tempfile,types
from pathlib import Path

OUT=Path(__file__).resolve().parent
TOOLS=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5')
results=[]
def shell(script,env):
    p=subprocess.run(['bash','-c',script],env={**os.environ,**env,'JAX_PLATFORMS':'cpu','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=10)
    return p.returncode
def fn(source,name):
    m=re.search(r'^'+re.escape(name)+r'\(\) \{.*?^\}',source,re.M|re.S);assert m,name;return m.group(0)

with tempfile.TemporaryDirectory(prefix='rc_controls_',dir=OUT) as tmp:
    root=Path(tmp);a=root/'rw3';(a/'runs').mkdir(parents=True)
    for n in ('N1','O1','N72','O72'):(a/'runs'/n).mkdir()
    shim=root/'python_transport.py'
    shim.write_text('''#!/usr/bin/env python3
import os,sys
args=[x for x in sys.argv[1:] if not x.startswith('-')]
program=next((os.path.basename(x) for x in args if x.endswith('.py')), '')
codes={'accept.py':17,'attrib_detail.py':19,'rw3t_verdict.py':23,'disk_watch.py':29,'arm_receipt.py':37,'arm_timer.py':31}
raise SystemExit(codes.get(program,0) if os.environ.get('FAIL_CHILD')==program else 0)
''');shim.chmod(0o700)
    stub=root/'lib_transport.sh';stub.write_text(f'W44R_PY={shim}\nW44R_TOOLS={TOOLS}\nw44_log() {{ :; }}\n')
    original=(TOOLS/'rw3_summary.sh').read_text()
    localized=original.replace('source '+str(TOOLS/'lib.sh'),'source '+str(stub),1)
    assert localized!=original
    for child in ('','accept.py','attrib_detail.py','rw3t_verdict.py','disk_watch.py'):
        env=dict(W44R_RUN=str(a),W44R_RECEIPTS=str(root/'receipts'),W44R_MANIFEST_PATH=str(root/'PIN.json'),FAIL_CHILD=child)
        rc=shell(localized,env)
        assert rc=={'':0,'accept.py':17,'attrib_detail.py':19,'rw3t_verdict.py':23,'disk_watch.py':29}[child],('post rc propagation',child,rc)
        results.append(dict(control='actual RW3 post with inert Python transport',injected_child=child or 'none',child_rc={'accept.py':17,'attrib_detail.py':19,'rw3t_verdict.py':23,'disk_watch.py':29}.get(child,0),observed_post_rc=rc,defect=False))

    short=(TOOLS/'session_short.sh').read_text()
    run=root/'short_runs';lay=root/'layers';run.mkdir();lay.mkdir()
    prefix=f'''set -u
R={run}; LAY={lay}; RB={root}/bundle; W44R_RUN={root}; W44R_RECEIPTS={root}/receipts; W44R_LOG={root}/session.log; W44R_PY={shim}; W44R_TOOLS={TOOLS}
w44_log() {{ :; }}
input_of() {{ echo /HOST_FIXTURE_CASE; }}
cap_for() {{ echo 600; }}
w44_disk_ok() {{ return 0; }}
check() {{ echo BUNDLE_OK; return 0; }}
samplers_on() {{ :; }}
samplers_off() {{ :; }}
host_load() {{ :; }}
run_arm() {{ return 0; }}
'''
    # Exercise the unchanged receipt and replay functions, not a reimplementation.
    body=prefix+fn(short,'receipt')+'\n'+fn(short,'replay_arm')+'\nreplay_arm B1 KEY 100 600 2 1 model --hours 3\n'
    rc=shell(body,dict(FAIL_CHILD='arm_receipt.py'));assert rc==37
    results.append(dict(control='actual receipt+replay_arm functions',injected_child='arm_receipt.py',child_rc=37,observed_replay_rc=rc,defect=False))
    tail=short[short.index('if [[ $SEALED == 1 ]]; then\n  final='):]
    rc=shell('PRIME_OK=1; SEALED=1; SMOKE=0; STAGE=warm; P=/HOST_FIXTURE; W44R_RUN=/HOST_FIXTURE; T0=$(date +%s)\ncheck(){ return 41; };w44_log(){ :; }\n'+tail,{})
    assert rc==41
    results.append(dict(control='actual session_short final-check tail',injected_child='final_check_P',child_rc=41,observed_session_rc=rc,defect=False))

    rw3=(TOOLS/'session_rw3.sh').read_text()
    # An actual replay model nonzero IS propagated by the current RW3 function.
    slotroot=root/'slot';slotroot.mkdir();(slotroot/'core_watch_summary.json').write_text('{}')
    prefix2=f'''set -u
R={root}/slots; A={root}; P={root}/bundle; P0={root}/old; LAY={root}/slotlayers; W44R_PY={shim}; W44R_TOOLS={TOOLS}; W44R_LOG={root}/slot.log; W44R_RECEIPTS={root}/receipts; CASE=/HOST_FIXTURE_CASE; BASE_ARGS='--hours 3'; SMOKE=0; DEADLINE=$(( $(date +%s)+10000 ))
mkdir -p "$R" "$LAY"
log() {{ :; }}
chk() {{ return 0; }}
pre_gate() {{ return 0; }}
hold_stop() {{ return 1; }}
w44_disk_ok() {{ return 0; }}
samplers_on() {{ :; }}
samplers_off() {{ :; }}
quiet_off() {{ :; }}
host_load() {{ :; }}
QUIET_FILE={root}/quiet
'''
    rc=shell(prefix2+fn(rw3,'run_slot')+'\nrun_slot N1 N 1 24 540 8 0\n',dict(FAIL_CHILD='arm_timer.py'))
    assert rc==31,rc
    results.append(dict(control='actual RW3 run_slot model child rc',injected_child='arm_timer.py',child_rc=31,observed_slot_rc=rc,defect=False))

policy=types.ModuleType('policy');exec(compile((TOOLS/'e41_policy.py').read_text(),str(TOOLS/'e41_policy.py'),'exec'),policy.__dict__)
base=dict(e41_gate='INCOMPLETE',hard_stack_gate='PASS',n_entries=1,coverage=dict(status='PASS',n_discovered=1,n_scanned=1),entries=[dict(path='SYNTHETIC',elfs=[dict(max_stack_B=32,stack_over256=[],nonleaf_over1k=[],callgraph_unknown=[])],kinput_transpose_status='UNKNOWN_AOT_METADATA')])
import copy
checks=[]
for label,mutate,expected in [
    ('compactAOTunknown',lambda g:None,'PASS'),
    ('nativeStack257',lambda g:g['entries'][0]['elfs'][0].update(max_stack_B=257),'FAIL'),
    ('nativeNonleaf1001',lambda g:g['entries'][0]['elfs'][0].update(nonleaf_over1k=[['callee',1001]]),'FAIL'),
    ('unknownCallgraph',lambda g:g['entries'][0]['elfs'][0].update(callgraph_unknown=['x']),'BLOCKED'),
    ('rawFAILunexplained',lambda g:g.update(e41_gate='FAIL'),'BLOCKED'),
    ('rawFAILkInputDiagnostic',lambda g:(g.update(e41_gate='FAIL'),g['entries'][0].update(kinput_transpose_over500=[['site',664]])),'PASS')]:
    g=copy.deepcopy(base);mutate(g);r=policy.assess(g);assert r['hard_gate']==expected,(label,r);checks.append(dict(case=label,hard_gate=r['hard_gate'],raw_gate=r['raw_e41_gate']))
report=dict(scope='Actual sealed shell functions/scripts with only process transport localized; no model/JAX/GPU/nsys/raw-data execution',shell_controls=results,E41_controls=checks,source_sha256={n:hashlib.sha256((TOOLS/n).read_bytes()).hexdigest() for n in ('session_short.sh','session_rw3.sh','rw3_summary.sh','e41_policy.py')},temporary_fixtures='deleted')
(OUT/'CONTROLS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(shell_controls=len(results),masked_failures=sum(r['defect'] for r in results),RW3_model_rc_propagates=True,E41_controls=len(checks))))
