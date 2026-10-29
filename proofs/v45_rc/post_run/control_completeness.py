"""Actual F2 validator: complete measured FAIL eligible; missing data never eligible. Inert metadata only."""
from pathlib import Path
import copy,importlib.util,json,sys
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');sys.path.insert(0,str(R/'post_run'));from evidence_completeness import PRIME,REPLAY,complete

from fixtures_complete import fixture

s,r=fixture();assert complete(s,r)['required_score_complete'];rows=[dict(case='complete_measured_numeric_FAIL',eligible=True,scientific_PASS=False)]
for label,mut in [('six_missing_gates',lambda a,b:a.update({k:'MISSING'for k in ['RW1','RW2','RW4','E41','swiss_identity','compile_s_cold_minus_warm']})),('missing_speed',lambda a,b:b.pop('RW3_speed')),('missing_memory',lambda a,b:b.pop('RW3_memory')),('missing_timing_pair',lambda a,b:b['RW3_speed']['pairs_24h'].pop('N3/O3')),('missing_rss',lambda a,b:b['RW3_memory'].pop('rss_gib')),('missing_vram',lambda a,b:b['RW3_memory']['vram_pid_peak_mib'].update(N=[None]*3)),('missing_E41_coverage',lambda a,b:a['E41']['post']['gates']['all'].pop('coverage')),('missing_compile_cost',lambda a,b:a['compile_s_cold_minus_warm'].update(v045=None)),('missing_RW4_keys',lambda a,b:a['RW4'].update(cpu_aot_key_pairs={})),('absent_actual_arm',lambda a,b:a['arms']['B1_wn3_0227_3h'].update(present=False))]:
 a,b=fixture();mut(a,b);v=complete(a,b);assert not v['required_score_complete']and v['missing_evidence'],label;rows.append(dict(case=label,eligible=False,missing=v['missing_evidence']))
# Actual controller composition with valid D6/floor coverage must still refuse these missing gate summaries.
spec=importlib.util.spec_from_file_location('post_controller_f2',R/'post_run/controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);O=R/'F2_COMPLETENESS_CONTROL';O.mkdir(exist_ok=True);F=O/'short';Q=O/'rw3';(F/'d6_swiss').mkdir(parents=True,exist_ok=True);(Q/'d6_N72').mkdir(parents=True,exist_ok=True)
def wr(p,j):p.write_text(json.dumps(j))
wr(Q/'d6_N72/summary.json',dict(domains={n:dict(pairing=dict(paired_file_count=73))for n in ['d01','d02','d03']},all_common_variables_scored=True,all_cpu_variables_required=True,pass_=False,**{'pass':False},output_integrity_pass=False));wr(Q/'d6_N72/FLOOR72.json',dict(coverage_complete=True,binding72_A3='FAIL'));wr(F/'d6_swiss/summary.json',dict(domains=dict(d01=dict(pairing=dict(paired_file_count=25))),all_common_variables_scored=True));wr(F/'SHORT_VERDICT.json',s);wr(Q/'RW_VERDICT_v045.json',r);assert m.check_scores_paths(F,Q)['required_score_complete']
wr(F/'SHORT_VERDICT.json',{k:'MISSING'for k in s if k!='arms'});proof=m.check_scores_paths(F,Q);assert proof['required_score_complete']is False
(O/'VERDICT.json').write_text(json.dumps(dict(PASS=True,rows=rows,actual_controller_missing_gate_refused=True,metadata_only=True,raw_reads=0,deletes=0),indent=2)+'\n');print('F2 completeness controls PASS',len(rows)+1,'measuredFAIL retained; missing gates/timing/RSS/VRAM refuse deletion')
