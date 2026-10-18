"""Summarize the w41-stoch GPU qualification arms (CPU only, no JAX).

Per arm: D6 verdict/failures/nonfinite/missing-vs-CPU, stochastic pattern + seed verdict, E41 gate, census
nonfinite/repairs, process wall, steady per-forecast-hour segment time from wrfout mtimes (median of hours
2-24, cold process on a shared box), sampled VRAM peak, persistent device state bytes.
usage: python summarize.py <cases_root> <out.json> [case ...]
"""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def arm(case):
    out = dict(case=case.name, prepare=(load(case/'PREPARE.json') or {}).get('stoch'))
    run = load(case/'gpu_run.json')
    if run:
        mt = run.get('wrfout_mtime') or {f.name: f.stat().st_mtime for f in (case/'gpu').glob('wrfout_d01_*')}
        times = [t for _, t in sorted(mt.items())]
        seg = [b - a for a, b in zip(times[:-1], times[1:])]
        out.update(rc=run['rc'], platform=run['platform'], source_commit=run['source_commit'],
                   process_wall_s=round(run['monotonic_wall_s'], 1), vram_mib_peak=run.get('vram_mib_peak', run.get('sampled_process_vram_mib_peak')),
                   frames=len(times), first_frame_s=round(times[0] - run['t0_epoch'], 1) if times and 't0_epoch' in run else None,
                   steady_s_per_fch=round(statistics.median(seg[1:]), 3) if len(seg) > 2 else None,
                   segment_s=[round(s, 2) for s in seg])
    proof = load(case/'gpu_proof/nested_pipeline_run.json')
    if proof:
        out['persistent_state_bytes'] = proof.get('hierarchy', {}).get('persistent_state_bytes')
        out['all_domains_finite'] = proof.get('all_domains_finite')
    d6 = load(case/'D6_d01.json')
    if d6:
        missing = sorted({m for f in d6['by_frame'] for m in f.get('missing_vs_cpu', [])})
        nonfinite = sorted({m for f in d6['by_frame'] for m in f.get('nonfinite', [])})
        out['d6'] = dict(verdict=d6['verdict'], paired_frames=d6['paired_frames'], coverage_complete=d6['coverage_complete'],
                         failures=d6.get('failures'), missing_vs_cpu=missing, nonfinite=nonfinite,
                         manifest_sha256=d6['manifest_sha256'],
                         cpu_fields=d6['by_frame'][-1]['cpu_field_count'], gpu_fields=d6['by_frame'][-1]['gpu_field_count'],
                         worst=_worst(d6))
    pat = load(case/'PATTERN_d01.json')
    if pat:
        out['pattern'] = dict(verdict=pat['verdict'], paired_frames=pat['paired_frames'], missing_on_gpu=pat['missing_on_gpu'],
                              worst_rel=max((v['worst_rel'] for v in pat['fields'].values()), default=None),
                              fields=sorted(pat['fields']), seeds={k: f"{v['exact_frames']}/{v['frames']}" for k, v in pat['seeds'].items()})
    gate = load(case/'e41/wave_gate.json')
    if gate:
        e = gate['entries']
        out['e41'] = dict(gate=gate['e41_gate'], hard_stack=gate.get('hard_stack_gate'),
                          kernels=[x['elfs'][0]['kernels'] for x in e if x.get('elfs')],
                          max_stack_B=max((x['elfs'][0]['max_stack_B'] for x in e if x.get('elfs')), default=None),
                          kinput_over500=sorted({tuple(k) for x in e for k in x.get('kinput_transpose_over500', [])}),
                          nonleaf_over1k=sorted({tuple(k) for x in e for el in x.get('elfs', []) for k in el.get('nonleaf_over1k', [])}))
    census = load(case/'gpu/census.json')
    if census:
        guards = [g for d in census.get('domains', {}).values() for g in d.get('guards', {}).values()]
        out['census'] = dict(guards=len(guards), **{k: sum(int(g.get(k, 0)) for g in guards)
                                                   for k in ('nonfinite', 'out_of_range', 'repaired')})
    return out


def _worst(d6):
    fields = d6.get('fields') or {}
    rows = {}
    for name in ('T2', 'U10', 'V10', 'PSFC', 'RAINNC', 'T', 'U', 'V', 'W', 'QVAPOR'):
        f = fields.get(name)
        if isinstance(f, dict):
            rows[name] = {k: f.get(k) for k in ('worst_rmse', 'max_rmse', 'rmse_max', 'limit', 'pass') if k in f}
    return rows


LABEL = {'st0': 'OFF control', 'st1': 'SKEBS', 'st2': 'SPPT 15 km', 'st3': 'SPP (MYNN) 15 km',
         'st4': 'all + RAND, vert 1, nens 7 (serial RNG)', 'st4p': 'all + RAND, vert 1, nens 7',
         'c00': 'OFF reference (v-prep c00, 2026-10-16, tree 04cb09f8)'}


def markdown(rows):
    lines = ['| arm | &stoch | D6 vs CPU-WRF (fields x frames) | stochastic fields / seeds | E41 kernels / STACK | census nonfinite / repaired | cold run s | steady s per forecast h | VRAM MiB |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        d6, pat, e41 = r.get('d6') or {}, r.get('pattern') or {}, r.get('e41') or {}
        d6s = (f"{d6.get('verdict')} ({d6.get('cpu_fields')} x {d6.get('paired_frames')}; {len(d6.get('failures') or [])} fail, "
               f"{len(d6.get('nonfinite') or [])} nonfinite" + (f"; CPU-only {','.join(d6['missing_vs_cpu'])}" if d6.get('missing_vs_cpu') else '') + ')')
        pats = (f"{pat.get('verdict')} {len(pat.get('fields', []))} fields, worst {pat.get('worst_rel'):.1e}; seeds "
                + ','.join(sorted(set(pat.get('seeds', {}).values())))
                + (f"; missing on GPU {','.join(pat['missing_on_gpu'])}" if pat.get('missing_on_gpu') else '')
                if pat.get('fields') else 'none written (OFF)')
        e = f"{e41.get('gate')} {'/'.join(map(str, sorted(set(e41.get('kernels', [])))))} / {e41.get('max_stack_B')} B" if e41 else 'n/a'
        cen = r.get('census') or {}
        cens = f"{cen.get('nonfinite')} / {cen.get('repaired')}" if cen else 'n/a'
        lines.append(f"| {r['case']} | {LABEL.get(r['case'], '')} | {d6s} | {pats} | {e} | {cens} | {r.get('process_wall_s')} | "
                     f"{r.get('steady_s_per_fch')} | {r.get('vram_mib_peak')} |")
    return '\n'.join(lines) + '\n'


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or sorted(p.name for p in root.iterdir() if (p/'PREPARE.json').exists())
    rows = [arm(root/n) for n in names]
    out.write_text(json.dumps(dict(schema='w41-stoch-summary-v1', arms=rows), indent=1) + '\n')
    out.with_suffix('.md').write_text(markdown(rows))
    for r in rows:
        print(r['case'], r.get('rc'), r.get('process_wall_s'), r.get('steady_s_per_fch'), r.get('vram_mib_peak'),
              (r.get('d6') or {}).get('verdict'), (r.get('pattern') or {}).get('verdict'), (r.get('e41') or {}).get('gate'))


if __name__ == '__main__':
    main()
