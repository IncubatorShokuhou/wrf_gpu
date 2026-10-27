"""Per-kernel static HBM bytes x measured node-trace duration -> per-family bandwidth roofline (S4 bw view).

Inputs are existing artifacts only (no JAX, no GPU):
  * actual rich GPU optimized HLO of the v0.4.3 bundle PROD programs (w44-perf ACTUAL_HLO, bound to native records),
  * runtime thunk -> HLO instruction -> native symbol bindings (w44-perf ORDINARY_THUNKS),
  * node-trace kernel rows (w44-perf TRACE_SUMMARY N43 = 3 ordinary roots, R43 = rad/KF window),
  * family labels per (domain, kernel) from w44-perf FAMILY_TIMES_FINAL joins.
Bytes are a streaming lower bound: every distinct operand byte read once, every output byte written once.
Slice-aware: a fusion parameter consumed only by (dynamic-)slice/gather counts the slice bytes; a root
dynamic-update-slice into a parameter counts only the update. Pallas custom calls count full operands+outputs.
Duration is node-trace time (node tracing inflates small kernels, E185): achieved bandwidth is a lower bound.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
DT = {'pred': 1, 's8': 1, 'u8': 1, 's16': 2, 'u16': 2, 'f16': 2, 'bf16': 2, 's32': 4, 'u32': 4, 'f32': 4,
      's64': 8, 'u64': 8, 'f64': 8, 'c64': 8, 'c128': 16, 'token': 0, 's4': 1, 'u4': 1, 'f8e4m3fn': 1, 'f8e5m2': 1}
ARR = re.compile(r'([a-z0-9]+)\[([0-9,]*)\]')
SLICERS = {'slice', 'dynamic-slice', 'gather'}
VIEWS = {'bitcast', 'get-tuple-element'}


def shape_bytes(shape):
    total = 0
    for dt, dims in ARR.findall(shape):
        n = 1
        for d in dims.split(','):
            if d:
                n *= int(d)
        total += n * DT.get(dt, 4)
    return total


def split_top(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur and ''.join(cur).strip():
        out.append(''.join(cur).strip())
    return out


def parse_inst(line):
    body = line.strip()
    root = body.startswith('ROOT ')
    if root:
        body = body[5:]
    if not body.startswith('%') or ' = ' not in body:
        return None
    name, rest = body[1:].split(' = ', 1)
    if rest.startswith('('):
        depth = 0
        for i, ch in enumerate(rest):
            depth += ch == '('
            depth -= ch == ')'
            if depth == 0:
                break
        shape, rest = rest[:i + 1], rest[i + 1:].lstrip()
    else:
        shape, rest = rest.split(' ', 1)
    m = re.match(r'([\w\-]+)\(', rest)
    if not m:
        return None
    op = m.group(1)
    depth, start = 0, m.end() - 1
    for i in range(start, len(rest)):
        depth += rest[i] == '('
        depth -= rest[i] == ')'
        if depth == 0:
            break
    args = split_top(re.sub(r'/\*.*?\*/', '', rest[start + 1:i]))
    operands = [a.split()[-1].lstrip('%') for a in args if '%' in a]
    attrs = rest[i + 1:]
    calls = re.search(r'\bcalls=%?([\w.\-]+)', attrs)
    cname = re.search(r', name = "([^"]+)"', attrs)
    return {'name': name, 'shape': shape, 'op': op, 'operands': operands, 'root': root,
            'calls': calls.group(1) if calls else None, 'pallas_name': cname.group(1) if cname else None,
            'kind': (re.search(r'kind=(k\w+)', attrs) or [None, None])[1]}


def parse_hlo(path):
    comps, current = {}, None
    with path.open() as f:
        for line in f:
            if line.startswith(('FileNames', 'FunctionNames', 'FileLocations', 'StackFrames')):
                current = None; continue
            m = re.match(r'^(ENTRY )?%?([\w.\-]+) .*\{\s*$', line)
            if m and not line.startswith(' '):
                current = m.group(2); comps[current] = {}
                continue
            if line.startswith('}'):
                current = None; continue
            if current is not None:
                inst = parse_inst(line)
                if inst:
                    comps[current][inst['name']] = inst
    return comps


def fusion_bytes(inst, caller, comps):
    """(read, write) bytes of one top-level fusion launch."""
    body = comps[inst['calls']]
    params = {}
    for x in body.values():
        if x['op'] == 'parameter':
            params[x['name']] = x
    uses = defaultdict(list)
    for x in body.values():
        for o in x['operands']:
            uses[o].append(x)
    root = next(x for x in body.values() if x['root'])
    roots = [body[o] for o in root['operands']] if root['op'] == 'tuple' else [root]

    def alias_of(name, seen=()):
        x = body.get(name)
        while x is not None and x['op'] in ('bitcast',) and x['operands']:
            x = body.get(x['operands'][0])
        return x['name'] if x is not None else name

    inplace_params, write = set(), 0
    for r in roots:
        if r['op'] == 'dynamic-update-slice' and alias_of(r['operands'][0]) in params:
            inplace_params.add(alias_of(r['operands'][0]))
            write += shape_bytes(body[r['operands'][1]]['shape']) if r['operands'][1] in body else shape_bytes(r['shape'])
        else:
            write += shape_bytes(r['shape'])
    read = 0
    for pname, p in params.items():
        full = shape_bytes(p['shape'])
        frontier, sliced, other = [pname], 0, False
        seen = set()
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen.add(n)
            for u in uses.get(n, []):
                if u['op'] == 'bitcast':
                    frontier.append(u['name'])
                elif u['op'] in SLICERS and u['operands'] and u['operands'][0] == n:
                    sliced += shape_bytes(u['shape'])
                elif u['op'] == 'dynamic-update-slice' and u['operands'][0] == n and pname in inplace_params:
                    pass
                else:
                    other = True
        read += full if other else min(full, sliced)
    return read, write


def instruction_bytes(inst, comp, comps):
    if inst['op'] == 'fusion' and inst['calls'] in comps:
        return fusion_bytes(inst, comp, comps)
    read = sum(shape_bytes(comps[comp][o]['shape']) for o in inst['operands'] if o in comps[comp])
    return read, shape_bytes(inst['shape'])


def kclass(symbol, op, kind, pallas):
    if pallas or op == 'custom-call':
        return 'pallas/custom' if pallas else 'custom-call'
    if 'transpose' in symbol:
        return 'xla-transpose-' + (kind or op)
    if op == 'fusion':
        return 'xla-' + (kind or 'fusion')
    return 'xla-' + op


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bw', type=float, default=1.792e12, help='peak HBM bytes/s (RTX 5090 spec 1792 GB/s)')
    ap.add_argument('--eff-bw', type=float, default=1.5e12, help='achievable streaming bytes/s for the XLA-fusion excess model')
    ap.add_argument('--tmin', type=float, default=2.0e-6, help='per-launch latency floor s (graph node)')
    a = ap.parse_args()
    index = json.loads((L / 'ACTUAL_HLO/INDEX.json').read_text())
    models, bindings = {}, []
    for rec in index['records']:
        if rec['version'] != 'v043':
            continue
        path = Path(rec['hlo_file'])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == rec['hlo_sha256'], 'HLO drift'
        comps = parse_hlo(path)
        inst_comp = {n: c for c, insts in comps.items() for n in insts}
        thunks = json.loads((L / f"ORDINARY_THUNKS/v043_{rec['program']}.json").read_text())
        assert thunks['native_record2_sha256'] == rec['native_record2_sha256']
        sym = defaultdict(set)
        for t in thunks['thunks']:
            if t['kind'] == 'kernel':
                sym[t['native_symbol']].add(t['hlo_instruction'])
        per_sym = {}
        for s, insts in sym.items():
            vals = set()
            for n in insts:
                c = inst_comp.get(n)
                if c is None:
                    continue
                vals.add(instruction_bytes(comps[c][n], c, comps))
                kinds = (comps[c][n]['op'], comps[c][n]['kind'], bool(comps[c][n]['pallas_name']))
            if vals:
                rd = sorted(vals)
                per_sym[s] = {'read': max(v[0] for v in rd), 'write': max(v[1] for v in rd),
                              'variants': len(rd), 'instructions': sorted(insts)[:4], 'cls': kclass(s, *kinds)}
        models[rec['program']] = per_sym
        bindings.append({'program': rec['program'], 'hlo_sha256': sha, 'native_record2_sha256': rec['native_record2_sha256'],
                         'computations': len(comps), 'symbols_with_bytes': len(per_sym)})
        print('parsed', rec['program'], len(comps), 'computations', len(per_sym), 'symbols', flush=True)

    results = {}
    for arm in ('N43', 'R43'):
        kernels = json.loads((L / f'TRACE_SUMMARY/{arm}_kernels.json').read_text())
        joins = json.loads((L / f'FAMILY_TIMES_FINAL/{arm}_joins.json').read_text())
        fam = defaultdict(lambda: Counter())
        cls = defaultdict(lambda: Counter())
        famcls = defaultdict(lambda: Counter())
        rows = []
        for k in kernels:
            d = k['domain']; name = k['name']
            dur = (k['end'] - k['start']) * 1e-9
            info = models.get(d, {}).get(name) or models.get(d, {}).get(re.sub(r'__\d+$', '', name))
            family = joins.get(f'{d}:{name}', {}).get('family', 'force/unbound' if d == 'force' else 'unmatched')
            f = fam[family]
            f['count'] += 1; f['ns'] += dur * 1e9
            if info:
                b = info['read'] + info['write']
                f['bytes'] += b; f['bound_count'] += 1; f['bound_ns'] += dur * 1e9
                floor = b / a.bw
                f['floor_ns'] += floor * 1e9
                f['excess_ns'] += max(0.0, dur - floor) * 1e9
                if dur < 4e-6:
                    f['tiny_count'] += 1; f['tiny_ns'] += dur * 1e9
                rows.append((dur, b, d, name, family))
                fc = famcls[(family.split(':')[0] if not family.startswith('mixed') else 'mixed', info['cls'])]
                fc['count'] += 1; fc['ns'] += dur * 1e9; fc['bytes'] += b
                c = cls[info['cls']]
                c['count'] += 1; c['ns'] += dur * 1e9; c['bytes'] += b
                c['eff_floor_ns'] += b / a.eff_bw * 1e9
                c['excess_eff_ns'] += max(0.0, dur - max(b / a.eff_bw, a.tmin)) * 1e9
                if dur < 4e-6:
                    c['tiny_count'] += 1; c['tiny_ns'] += dur * 1e9
        roots = 3
        out = {}
        for family, f in sorted(fam.items(), key=lambda kv: -kv[1]['ns']):
            bw = f['bytes'] / (f['bound_ns'] * 1e-9) if f['bound_ns'] else float('nan')
            out[family] = {'kernels_per_root': f['count'] / roots, 'ms_per_root': f['ns'] / 1e6 / roots,
                           'bound_fraction_of_time': f['bound_ns'] / f['ns'] if f['ns'] else 0,
                           'GB_per_root': f['bytes'] / 1e9 / roots, 'bw_floor_ms_per_root': f['floor_ns'] / 1e6 / roots,
                           'achieved_TBps': bw / 1e12 if bw == bw else None,
                           'pct_of_peak_bw': 100 * bw / a.bw if bw == bw else None,
                           'excess_over_bw_floor_ms_per_root': f['excess_ns'] / 1e6 / roots,
                           'tiny_lt4us_kernels_per_root': f['tiny_count'] / roots, 'tiny_ms_per_root': f['tiny_ns'] / 1e6 / roots}
        tot = Counter()
        for f in fam.values():
            tot.update(f)
        rows.sort(reverse=True)
        top = [{'ms_per_root': r[0] * 1e3 / roots, 'MB': r[1] / 1e6, 'domain': r[2], 'kernel': r[3], 'family': r[4],
                'TBps': r[1] / r[0] / 1e12} for r in rows[:60]]
        # aggregate same-kernel launches for a ranked list of the heaviest symbols
        agg = defaultdict(lambda: [0, 0.0, 0.0, None])
        for dur, b, d, name, family in rows:
            g = agg[(d, name)]; g[0] += 1; g[1] += dur; g[2] += b; g[3] = family
        heavy = sorted(agg.items(), key=lambda kv: -kv[1][1])[:80]
        classes = {k: {'kernels_per_root': v['count'] / roots, 'ms_per_root': v['ns'] / 1e6 / roots,
                       'GB_per_root': v['bytes'] / 1e9 / roots, 'achieved_TBps': v['bytes'] / v['ns'] / 1e3 if v['ns'] else None,
                       'excess_over_eff_floor_ms_per_root': v['excess_eff_ns'] / 1e6 / roots,
                       'tiny_lt4us_per_root': v['tiny_count'] / roots, 'tiny_ms_per_root': v['tiny_ns'] / 1e6 / roots}
                   for k, v in sorted(cls.items(), key=lambda kv: -kv[1]['ns'])}
        for k, v in classes.items():
            print('   class', f'{k:28s}', ' '.join(f'{x}={y:.3f}' for x, y in v.items() if y is not None), flush=True)
        family_class = {f'{fa}|{cl}': {'kernels_per_root': v['count'] / roots, 'ms_per_root': v['ns'] / 1e6 / roots,
                                       'GB_per_root': v['bytes'] / 1e9 / roots,
                                       'TBps': v['bytes'] / v['ns'] / 1e3 if v['ns'] else None}
                        for (fa, cl), v in sorted(famcls.items(), key=lambda kv: -kv[1]['ns'])}
        results[arm] = {'roots': roots, 'kernels': len(kernels), 'classes': classes, 'family_class': family_class,
                        'kernel_ms_per_root': tot['ns'] / 1e6 / roots,
                        'bound_ms_per_root': tot['bound_ns'] / 1e6 / roots,
                        'GB_per_root': tot['bytes'] / 1e9 / roots,
                        'bw_floor_ms_per_root': tot['floor_ns'] / 1e6 / roots,
                        'tiny_lt4us_kernels_per_root': tot['tiny_count'] / roots, 'tiny_ms_per_root': tot['tiny_ns'] / 1e6 / roots,
                        'families': out, 'top_launches': top,
                        'heaviest_symbols': [{'domain': d, 'kernel': n, 'launches_per_root': g[0] / roots,
                                              'ms_per_root': g[1] * 1e3 / roots, 'GB_per_root': g[2] / 1e9 / roots,
                                              'TBps': g[2] / g[1] / 1e12 if g[1] else None, 'family': g[3]}
                                             for (d, n), g in heavy]}
        print(arm, json.dumps({k: v for k, v in results[arm].items() if k not in ('families', 'top_launches', 'heaviest_symbols')}), flush=True)
        for family, v in list(out.items())[:16]:
            print('  ', f'{family:32s}', ' '.join(f'{k}={v[k]:.3f}' if isinstance(v[k], float) else f'{k}={v[k]}' for k in
                                              ('kernels_per_root', 'ms_per_root', 'GB_per_root', 'bw_floor_ms_per_root', 'pct_of_peak_bw', 'tiny_ms_per_root')), flush=True)
    a.out.write_text(json.dumps({'schema': 'o5-perf.roofline_join.v1', 'scope': __doc__, 'peak_bw_Bps': a.bw,
                                 'hlo_bindings': bindings, 'arms': results}, indent=1) + '\n')


if __name__ == '__main__':
    main()
