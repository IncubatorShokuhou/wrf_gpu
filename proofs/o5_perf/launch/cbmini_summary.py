"""cbmini summary: per arm rc, wrfout SHA == sealed bundle B6 PROD 3 h receipt, d02/d01 host ms per call (segments 2+),
seg2-3 call ms/root, VLOG counts (min-graph-size rejections, command buffers created)."""
import json, statistics as st, sys
from pathlib import Path
R, ref_file = Path(sys.argv[1]), Path(sys.argv[2])
ref = sorted(l.split()[0] for l in ref_file.read_text().splitlines())
out = {}
for a in ('M_OFF', 'M_V1P'):
    d = R / a; rec = {'rc': (d / 'rc.txt').read_text().strip() if (d / 'rc.txt').exists() else None}
    f = d / 'wrfout_sha.txt'
    sha = sorted(l.split()[0] for l in f.read_text().splitlines()) if f.exists() else None
    rec['n_wrfout'] = len(sha or []); rec['wrfout_identical_to_sealed_B6'] = sha == ref if sha else None
    r = d / 'run' / 'receipt.json'
    if r.exists():
        j = json.loads(r.read_text()); segs = j['segments']
        if len(segs) >= 3:
            t1 = segs[1]['t_start']; adv = [x for x in j['advance'] if x['t'] >= t1]
            for dom in ('d01', 'd02'):
                hs = [x['host_s'] for x in adv if x['d'] == dom]
                rec[f'{dom}_host_ms_median'] = 1e3 * st.median(hs); rec[f'{dom}_calls'] = len(hs)
            rec['seg23_call_ms_per_root'] = 1e3 * sum(s['call_s'] for s in segs[1:3]) / sum(s['root_steps'] for s in segs[1:3])
            rec['pre_first_segment_s'] = segs[0]['t_start']
        rec['device'] = j.get('device')
    log = d / 'run.log'
    if log.exists():
        t = log.read_text(errors='replace')
        rec['vlog_min_graph_size_rejections'] = t.count('because command size is less than the min graph size')
        rec['vlog_command_buffer_thunks_created'] = t.count('Creating command buffer thunk')
    out[a] = rec
(R / 'CBMINI_SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))
