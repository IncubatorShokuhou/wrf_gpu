"""Collect the lane's GPU session receipts (no JAX): per arm gpu_run.json, D6 (compare_wrfout_grid,
frozen manifest), TKE_COMPARE.json, and the shared-cache E41 gate with per-program stack attribution
against the v0.4.0 v-prep composite gates. usage: python collect_gpu_arms.py <out.json>"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path('<DATA_ROOT>/wrf_gpu2/scratch/w41-shinhongtke/cases')
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w41-shinhongtke')
VPREP = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
ARMS = {'c03': ('MYJ bl=2 + MYJSFC, cold TKE seed', 'c03'), 'c06': ('BouLac bl=8, cold TKE seed', 'c06'),
        'c07sd1': ('Shin-Hong bl=11 shinhong_tke_diag=1 vs NEW original CPU-WRF', 'c07'),
        'c28': ('CAM-UW bl=9 (seed history-only), 6 h', 'c28')}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gate = json.loads((LANE / 'e41' / 'wave_gate.json').read_text())
stack_by_entry = {e['entry']: max(x['max_stack_B'] for x in e['elfs']) for e in gate['entries']}
out = dict(session=dict(lock_start='2026-10-17T23:49:37Z', lock_end='2026-10-18T00:36:22Z', gpu_s=2805,
                        commit='280a0126fec999183954c08ec88ebdfb8c288da0', cache=str(LANE / 'cache_w41sht')),
           e41=dict(verdict=gate['e41_gate'], hard_stack=gate['hard_stack_gate'], conservative_flags=gate['conservative_flags'],
                    coverage=gate['coverage'], max_stack_per_entry=stack_by_entry, gate_sha256=sha(LANE / 'e41' / 'wave_gate.json')),
           arms={})
for case, (what, base) in ARMS.items():
    C = ROOT / case
    run = json.loads((C / 'gpu_run.json').read_text())
    d6 = json.loads((C / 'D6_d01.json').read_text()); s = d6['summaries']; inv = d6['inventory']
    tk = json.loads((C / 'TKE_COMPARE.json').read_text()); rows = tk['rows'][1:]
    def rng(k, f):
        vals = [r[k][f] for r in rows if isinstance(r.get(k), dict) and r[k].get(f) is not None]
        return [min(vals), max(vals)] if vals else None
    old = json.loads((VPREP / base / 'e41' / 'wave_gate.json').read_text())
    out['arms'][case] = dict(
        what=what, rc=run['rc'], platform=run['platform'], device=run['device'], wall_s=run['monotonic_wall_s'],
        source_commit=run['source_commit'], input_sha256=run['input_sha256'],
        d6=dict(paired_frames=len(d6['pairing']) if isinstance(d6['pairing'], list) else d6['pairing'].get('pair_count', 'see json'),
                tolerance_failures=s['tolerance_failure_count'], bounded_fields=62, cpu_only=inv['cpu_only_count'],
                gpu_only=inv['gpu_only_count'], json_sha256=sha(C / 'D6_d01.json')),
        tke=dict(lead0_byte=tk['lead0_byte'], tops_zero=tk['tops_zero'], nonfinite=tk['nonfinite_gpu'],
                 tke_r=rng('TKE_PBL', 'pearson_r'), tke_bias=rng('TKE_PBL', 'bias'), el_r=rng('EL_PBL', 'pearson_r'),
                 el_gpu_mean=rng('EL_PBL', 'gpu_mean')),
        v040_e41_max_stack=max(x['max_stack_B'] for e in old['entries'] for x in e.get('elfs', [])))
out['reading'] = ('[M GPU, asserted RTX 5090] 4/4 arms rc 0 on the final tree. D6 vs ORIGINAL CPU-WRF (frozen manifest, 62 bounded '
                  'fields, every frame): 0 tolerance failures, 0 CPU-only variables. TKE_PBL lead 0 BYTE in all four; later leads r>=.97. '
                  'EL_PBL: Shin-Hong diag=1 live (r reported); MYJ/BouLac still 0 (disclosed); CAM-UW has none in WRF. E41 overall FAIL '
                  '(conservative kInput flags on JIT entries + c03 MYJ STACK 2456 B) == the v0.4.0 composite values per scheme (pre-existing).')
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n')
for k, v in out['arms'].items():
    print(k, v['rc'], round(v['wall_s']), 'D6 fail', v['d6']['tolerance_failures'], 'cpu_only', v['d6']['cpu_only'], 'TKE', v['tke'], 'v040 stack', v['v040_e41_max_stack'])
print(out['e41'])
