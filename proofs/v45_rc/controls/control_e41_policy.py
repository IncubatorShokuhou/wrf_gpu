from pathlib import Path
import copy,importlib.util,json
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');p=Path('<DATA_ROOT>/wrf_gpu2/worktrees/w44-rc/proofs/v45_rc/tools_v6/e41_policy.py');spec=importlib.util.spec_from_file_location('e41policy',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=R/'smoke_v6/session_254c48237452_smoke_resume1_v6/e41/all/wave_gate.json';g=json.loads(f.read_text());base=m.assess(g);assert base['hard_gate']=='PASS' and base['status']=='HARD_PASS_HLO_UNKNOWN'
large=copy.deepcopy(g);large['entries'][0]['kinput_transpose_over500']=[['CONTROL',501]];assert m.assess(large)['hard_gate']=='FAIL'
stack=copy.deepcopy(g);stack['entries'][0]['elfs'][0]['max_stack_B']=257;assert m.assess(stack)['hard_gate']=='FAIL'
unknown=copy.deepcopy(g);unknown['entries'][0]['elfs'][0]['callgraph_unknown']=['CONTROL'];assert m.assess(unknown)['hard_gate']=='BLOCKED'
# Removing the newly explicit existing HLO-limit branch must falsely pass the over500 input.
ns={'__name__':'policy_mutant'};s=p.read_text();line=next(l for l in s.splitlines(True)if "if result['diagnostics']:hard.append"in l);exec(compile(s.replace(line,''),'<private-mutant>','exec'),ns);assert ns['assess'](large)['hard_gate']=='PASS'
(R/'v6_admissions/E41_POLICY_CONTROLS.json').write_text(json.dumps(dict(checks=5,PASS=True,raw_actual_baseline=str(f),baseline_compact_HLO='UNKNOWN; never a measured HLO PASS',over500_FAIL=True,stack257_FAIL=True,unknown_native_BLOCKED=True,removed_limit_mutant_killed=True),indent=2)+'\n');print('E41 policy controls PASS5 / no gate relaxation')
