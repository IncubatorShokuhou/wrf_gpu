"""Bounded read-only cross-fusion SSA/materialization screen of exact GPU HLO.

No model imports, lowering or device access. Structural equality preserves
operand order, dtype, dimensions, layout and non-debug attributes. Control and
native calls are opaque SSA boundaries; no equality across RK/while boundaries.
Equal broadcasts/views alone do not establish saved memory traffic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

LINE = re.compile(r'^\s*(ROOT )?%([\w.\-]+) = (.*?) ([a-z][a-z0-9-]*)\(')
COMP = re.compile(r'^(?:ENTRY )?%([\w.\-]+) .*\{\s*$')
PURE = set('abs add subtract multiply divide maximum minimum compare select and or not negate convert bitcast reshape transpose slice dynamic-slice dynamic-update-slice broadcast concatenate pad exp log sqrt rsqrt power reduce-window'.split())
VIEWS = {'bitcast', 'reshape', 'transpose', 'broadcast', 'slice', 'constant'}
L = Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')


def parse(path):
    comps = {}; comp = None
    for line in path.read_text().splitlines():
        if m := COMP.match(line):
            comp = m[1]; comps[comp] = []
        elif line == '}':
            comp = None
        elif comp and (m := LINE.match(line)):
            tail = line[m.end():]
            # Operands contain no nested calls in printed HLO (constants may
            # contain literal vectors). Find the balanced closing delimiter.
            depth = 1; end = None
            for i, c in enumerate(tail):
                if c == '(': depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0: end = i; break
            assert end is not None
            args, attrs = tail[:end], tail[end+1:]
            attrs = attrs.split(', metadata=', 1)[0]
            attrs = re.sub(r'/\*.*?\*/', '', attrs).strip(' ,')
            shape = re.sub(r'/\*.*?\*/', '', m[3]).replace(' ', '')
            frame = re.search(r'stack_frame_id=(\d+)', tail)
            comps[comp].append(dict(name=m[2], shape=shape, op=m[4], frame=int(frame[1]) if frame else None,
                                    args=args, inputs=re.findall(r'%([\w.\-]+)', args),
                                    attrs=attrs, root=bool(m[1])))
    return comps


def nbytes(shape):
    m = re.fullmatch(r'(pred|[sufbc]\d+)\[([\d,]*)\](?:\{.*\})?', shape)
    if not m: return None
    size = 1 if m[1] == 'pred' else int(re.search(r'\d+', m[1])[0]) // 8
    for d in filter(None, m[2].split(',')): size *= int(d)
    return size


def audit(comps, parent, targets):
    intern = {}; descriptors = {}; next_id = 0; values = {}; instances = defaultdict(list)
    materialized = defaultdict(list); output_values = {}; body_counts = Counter()
    source_operands = {}
    rows = comps[parent]; by_name = {x['name']: x for x in rows}
    consumers = defaultdict(list)
    for x in rows:
        for arg in x['inputs']: consumers[arg].append(x['name'])

    def node(key):
        nonlocal next_id
        if key not in intern:
            next_id += 1; intern[key] = next_id; descriptors[next_id] = key
        return intern[key]

    def get(value, index):
        key = descriptors[value]
        if key[0] == 'tuple': return key[1][index]
        return node(('opaque-gte', value, index))

    def evaluate(x, env, owner=None):
        op = x['op']; args = [env[a] for a in x['inputs']]
        if op == 'parameter': return node(('opaque-parameter', parent, x['args']))
        if op == 'tuple': return node(('tuple', tuple(args)))
        if op == 'get-tuple-element':
            return get(args[0], int(re.search(r'index=(\d+)', x['attrs'])[1]))
        if op == 'constant': return node(('constant', x['shape'], x['args'], x['attrs']))
        if op == 'fusion':
            callee = re.search(r'calls=%([\w.\-]+)', x['attrs'])[1]
            sub = {}
            for v in comps[callee]:
                if v['op'] == 'parameter':
                    sub[v['name']] = args[int(v['args'])]
                    source_operands[(x['name'], v['name'])] = x['inputs'][int(v['args'])]
                else: sub[v['name']] = evaluate(v, sub, x['name'])
                if v['root']: root = sub[v['name']]
            key = descriptors[root]
            outputs = list(key[1]) if key[0] == 'tuple' else [root]
            output_values[x['name']] = outputs
            for i, value in enumerate(outputs):
                materialized[value].append({'producer': x['name'], 'index': i})
            body_counts[callee] = len(comps[callee])
            return root
        if op in PURE:
            value = node((op, x['shape'], tuple(args), x['attrs']))
            if owner:
                instances[value].append({'owner': owner, 'instruction': x['name'], 'frame':x['frame'],
                                         'opcode': op, 'shape': x['shape'],
                                         'body_inputs':x['inputs'],
                                         'direct_outer_parameters':{a:source_operands[(owner,a)] for a in x['inputs'] if (owner,a) in source_operands},
                                         'bytes_if_materialized': nbytes(x['shape'])})
            return value
        return node(('opaque', parent, x['name'], x['shape']))

    for x in rows: values[x['name']] = evaluate(x, values)

    duplicate = []; repeated_materialized = []
    for value, sites in instances.items():
        owners = sorted({s['owner'] for s in sites})
        if len(owners) < 2 or not (set(owners) & targets): continue
        outputs = materialized.get(value, [])
        item = {'canonical_id': value, 'owners': owners, 'opcode': sites[0]['opcode'],
                'shape': sites[0]['shape'], 'bytes_if_materialized': sites[0]['bytes_if_materialized'],
                'materialized_outputs': outputs, 'instances': sites}
        duplicate.append(item)
        if len({o['producer'] for o in outputs}) >= 2: repeated_materialized.append(item)

    target_outputs = []
    for name in sorted(targets & output_values.keys()):
        for i, value in enumerate(output_values[name]):
            direct = consumers[name]
            if len(output_values[name]) > 1:
                gtes = [n for n in direct if by_name[n]['op'] == 'get-tuple-element'
                        and int(re.search(r'index=(\d+)', by_name[n]['attrs'])[1]) == i]
                uses = sorted({u for g in gtes for u in consumers[g]})
                shape = by_name[gtes[0]]['shape'] if gtes else None
            else:
                gtes = []; uses = direct; shape = by_name[name]['shape']
            target_outputs.append(dict(producer=name, index=i, canonical_id=value,
                                       shape=shape, bytes=nbytes(shape) if shape else None,
                                       gtes=gtes, consumers=uses, live=bool(uses)))
    return {'parent': parent, 'parent_instruction_count': len(rows),
            'targets': sorted(targets), 'fusion_bodies_examined': len(body_counts),
            'exact_cross_fusion_repeated_expressions': duplicate,
            'repeated_materialized_outputs': repeated_materialized,
            'target_outputs': target_outputs,
            'limits': ['HLO SSA liveness, not a buffer-assignment or dynamic traffic measurement.',
                       'Structural equality only; no reassociation or equivalence across control/native boundaries.',
                       'Broadcast/view repetition alone is not evidence of materialized redundant traffic.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    started = time.monotonic(); result = {'schema': 'w44-perf.q1-cross-fusion.v1', 'programs': []}
    index = json.loads((L/'ACTUAL_HLO/INDEX.json').read_text())
    for rec in index['records']:
        if rec['version'] != 'v043': continue
        witnesses = json.loads((L/'SOURCE_CENSUS_FINAL/N43_witnesses.json').read_text())
        groups = defaultdict(set); times = defaultdict(lambda: [0, 0])
        for w in witnesses:
            if w['domain'] != rec['program'] or w['label'] != 'dycore/advection': continue
            for s in w['possible_source_sites']:
                src = s['source']
                if src['operation_class'] == 'native Pallas': continue
                if not src['hlo_instruction'].startswith(('loop_', 'input_', 'wrapped_')): continue
                groups[src['computation']].add(src['hlo_instruction'])
                if len(w['possible_source_sites']) == 1:
                    times[src['hlo_instruction']][0] += w['activities']
                    times[src['hlo_instruction']][1] += w['kernel_ns']
        p = Path(rec['hlo_file']); assert hashlib.sha256(p.read_bytes()).hexdigest() == rec['hlo_sha256']
        comps = parse(p)
        for parent, targets in groups.items():
            data = audit(comps, parent, targets)
            data.update(domain=rec['program'], hlo_sha256=rec['hlo_sha256'],
                        source_file=str(p), unambiguous_target_count_ns={k:times[k] for k in targets})
            result['programs'].append(data)
            print(rec['program'], parent, 'targets', len(targets), 'duplicate expressions',
                  len(data['exact_cross_fusion_repeated_expressions']), 'repeated materialized',
                  len(data['repeated_materialized_outputs']), flush=True)
    result['wall_seconds'] = time.monotonic()-started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['cache'] = 'NONE; existing hash-bound HLO and trace witnesses read-only'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
