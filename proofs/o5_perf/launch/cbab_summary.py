"""Summarize a cbab session: per arm rc, median host ms of d01/d02 advance calls (segments 2+), seg2-3 call ms/root,
pre-first-segment s, wrfout SHA identity vs W0 (sealed OFF); per variant mean/spread of the measured arms (A/A rule 5 %)."""
import json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1]); out = {}
for d in sorted(p for p in R.iterdir() if p.is_dir()):
    rec = {'rc': (d / 'rc.txt').read_text().strip() if (d / 'rc.txt').exists() else None}
    f = d / 'wrfout_sha.txt'
    rec['sha'] = sorted(l.split()[0] + ' ' + l.split()[1] for l in f.read_text().splitlines()) if f.exists() else None
    r = d / 'run' / 'receipt.json'
    if r.exists():
        j = json.loads(r.read_text()); segs = j['segments']
        if len(segs) >= 3:
            t1 = segs[1]['t_start']; adv = [x for x in j['advance'] if x['t'] >= t1]
            rec['d01_host_ms'] = 1e3 * st.median([x['host_s'] for x in adv if x['d'] == 'd01'])
            rec['d02_host_ms'] = 1e3 * st.median([x['host_s'] for x in adv if x['d'] == 'd02'])
            rec['seg23_call_ms_per_root'] = 1e3 * sum(s['call_s'] for s in segs[1:3]) / sum(s['root_steps'] for s in segs[1:3])
            rec['pre_first_segment_s'] = segs[0]['t_start']
    log = d / 'run.log'
    if log.exists():
        text = log.read_text(errors='replace')
        rec['vlog_min_graph_size_rejections'] = text.count('is not lowered to command buffer because command size is less than the min graph size')
        rec['vlog_cmdbuf_created'] = text.count('Creating command buffer thunk')
    out[d.name] = rec
ref = out.get('W0', {}).get('sha')
for v in out.values():
    v['wrfout_identical_to_W0'] = (v.get('sha') == ref) if ref else None
    v.pop('sha', None)
groups = {'OFF': ['OFF_1', 'OFF_2', 'OFF_3'], 'V1': ['V1_1', 'V1_2', 'V1_3'], 'V2': ['V2_1', 'V2_2', 'V2_3']}
agg = {}
for g, arms in groups.items():
    vals = [out[a]['seg23_call_ms_per_root'] for a in arms if a in out and 'seg23_call_ms_per_root' in out[a]]
    if vals:
        m = st.mean(vals)
        agg[g] = {'seg23_call_ms_per_root_mean': m, 'aa_spread_pct': 100 * (max(vals) - min(vals)) / m, 'n_arms': len(vals),
                  'd02_host_ms_mean': st.mean(out[a]['d02_host_ms'] for a in arms if 'd02_host_ms' in out.get(a, {}))}
res = {'arms': out, 'variants': agg}
(R / 'CBAB_SUMMARY.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(agg))
