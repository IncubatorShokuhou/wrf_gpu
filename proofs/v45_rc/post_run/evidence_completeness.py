"""Presence/completeness only. A measured numeric FAIL is not missing evidence."""
import math,re
PRIME=('A1_swiss','A2_prod','A3_wn3_0227_3h','A4_wn3_0227_icloud0_3h')
REPLAY=('B1_wn3_0227_3h','B2_wn3_0502_3h','B4_wn3_0227in48h_3h','B5_swiss_3h','B6_prod_3h','B9_swiss_24h','B8_wn3_0502_icloud0_3h','B3_wn3_0614_3h')
EXPECTED_FILES={n:(25 if n=='B9_swiss_24h' else 4 if 'swiss'in n else 6 if 'prod'in n else 12)for n in PRIME+REPLAY}
PAIRS=('N1/O1','N2/O2','N3/O3');COUNTED={'N1','N2','N3','O1','O2','O3'}
def number(x,positive=False):return type(x)in(int,float) and math.isfinite(x) and (not positive or x>0)
def flag_map(x,keys=None):return isinstance(x,dict)and bool(x)and(keys is None or set(keys)<=set(x))and all(type(v)is bool for v in x.values())
def status(x):return isinstance(x,str)and bool(x)and not any(s in x.upper()for s in ['MISSING','UNAVAILABLE','INVALID','NOT_RUN','NOT RUN'])
def complete(short,rw):
 missing=[]
 def need(ok,name):
  if not ok:missing.append(name)
 def row(g):return short.get(g)if isinstance(short.get(g),dict)else{}
 for name in ['RW1','RW2','RW4','E41','swiss_identity','compile_s_cold_minus_warm']:
  need(isinstance(short.get(name),dict)and bool(short[name]),'short.'+name)
 arms=short.get('arms')or{}
 for name in PRIME+REPLAY:
  a=arms.get(name)or{}
  need(a.get('present')is True and a.get('rc')=='0','arm.'+name+'.execution')
  need(type(a.get('all_domains_finite'))is bool and number(a.get('proof_wall_total_s'),True)and type(a.get('n_wrfout'))is int and a['n_wrfout']==EXPECTED_FILES[name],'arm.'+name+'.proof')
  if name in REPLAY:
   c=a.get('consumer_counts')or{}
   need(all(type(c.get(k))is int and c[k]>=0 for k in ['compiled_new','aliases_new','runtime_new','other_new','layer_compiled']),'arm.'+name+'.consumer_counts')
   need(type(a.get('all_loaded'))is bool,'arm.'+name+'.aot')
   for k in ['check_before','check_after']:need(isinstance(a.get(k),str)and a[k].startswith(('BUNDLE_OK ','BUNDLE_BROKEN ')),'arm.'+name+'.'+k)
 r=row('RW1');need(status(r.get('verdict')),'RW1.verdict');need((r.get('new_prefix_checks')or{}).get('PREFIX B9_swiss_24h vs A1')in('PASS','FAIL'),'RW1.B9_prefix');bind=r.get('binding')or{}
 for name in ['B1_wn3_0227_3h','B5_swiss_3h','B6_prod_3h']:
  b=bind.get(name)or{};need(type(b.get('replay_ok'))is bool and b.get('bytes')in('PASS','FAIL'),'RW1.'+name)
 r=row('RW2');need(status(r.get('verdict')),'RW2.verdict');need(flag_map(r.get('arms'),['B2_wn3_0502_3h','B4_wn3_0227in48h_3h','B8_wn3_0502_icloud0_3h']),'RW2.arms')
 for k in ['one_key_per_program','keys_sealed']:need(flag_map(r.get(k)),'RW2.'+k)
 keys=r.get('gpu_keys')or{};need(bool(keys),'RW2.gpu_keys')
 for program,rows in keys.items():
  need(isinstance(rows,dict)and all(name in rows and isinstance(rows[name],list)and rows[name]and all(isinstance(k,str)and re.fullmatch('[0-9a-f]{64}',k)for k in rows[name])for name in ['A3_wn3_0227_3h','B1_wn3_0227_3h','B2_wn3_0502_3h','B4_wn3_0227in48h_3h','B3_wn3_0614_3h']),'RW2.keys.'+program)
 r=row('RW4');need(status(r.get('verdict')),'RW4.verdict');need(flag_map(r.get('dates'),['B2_wn3_0502_3h','B4_wn3_0227in48h_3h','B3_wn3_0614_3h']),'RW4.dates');pairs=r.get('cpu_aot_key_pairs')or{}
 need(set(['k0227_v_0502_h72','k0227_v_0614_h48','k0227_v_0227in48_h3','ki0227_v_ki0502_h3'])<=set(pairs)and all(flag_map(v)for v in pairs.values()),'RW4.CPU_pairs')
 e=row('E41');post=e.get('post')or{};g=(post.get('gates')or{}).get('all')or{};policy=e.get('RC_policy')or{}
 need(status(e.get('verdict')),'E41.verdict');need(type(g.get('n_entries'))is int and g['n_entries']>0 and g.get('coverage')in('PASS','COMPLETE'),'E41.native_coverage')
 need(g.get('hard_stack_gate')in('PASS','FAIL')and g.get('e41_gate')in('PASS','FAIL','INCOMPLETE')and type(g.get('conservative_flags'))is bool,'E41.measured_gates')
 need(policy.get('hard_gate')in('PASS','FAIL')and status(policy.get('status')or policy.get('reason')),'E41.policy')
 sw=row('swiss_identity');need(status(sw.get('verdict'))and sw.get('paired')==25 and type(sw.get('d6_pass'))is bool and type(sw.get('integrity'))is bool,'Swiss.measured24h')
 c=row('compile_s_cold_minus_warm');need(number(c.get('v045'))and number(c.get('v044'),True)and isinstance(c.get('basis'),str),'compile.measured_cost')
 speed=rw.get('RW3_speed')if isinstance(rw.get('RW3_speed'),dict)else{}
 need(status(speed.get('verdict'))and speed.get('primary_metric')=='d03_step_median','RW3.speed.verdict')
 need(COUNTED<=set(speed.get('counted')or[]),'RW3.counted_3_pairs')
 pairs=speed.get('pairs_24h')or{}
 for name in PAIRS:
  a=pairs.get(name)or{};need(all(number(a.get(k),True)for k in ['d03_step_median','whole_s_per_fch','forecast_only_s','d03_step_mean','d01_step_median']),'RW3.timing.'+name)
 mem=rw.get('RW3_memory')if isinstance(rw.get('RW3_memory'),dict)else{}
 for name in ['rss_gib','vram_pid_peak_mib']:
  a=mem.get(name)or{};need(all(isinstance(a.get(k),list)and len(a[k])>=3 and all(number(v,True)for v in a[k])for k in ['N','O'])and number(a.get('ratio'),True)and type(a.get('over_10pct'))is bool,'RW3.memory.'+name)
 return dict(required_score_complete=not missing,missing_evidence=missing)
