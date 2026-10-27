"""Original-protobuf caller provenance for the fourteen E84 quotient families.

Do not use HloModule.to_string StackFrames: XLA remaps their indices (TP75B).
Full-constant proto hashes bind values that ordinary HLO text elides (E224).
"""
import ast
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re

import numpy as np
from gpuwrf.runtime.aot_executable import _proto_fields

C = 'src/gpuwrf/coupling/physics_couplers.py'
O = 'src/gpuwrf/physics/wrf_cam_ozone.py'
PATTERNS = {
    'solar_lead60': (C, '_real4_div_rn(F(lead_seconds), F(60))'),
    'solar_day1440': (C, '_real4_div_rn(absolute, F(1440))'),
    'solar_degrad': (C, 'F(np.float32(np.float32(np.pi) / np.float32(180)))'),
    'solar_dpd': (C, 'F(np.float32(np.float32(360) / np.float32(365)))'),
    'solar_eot365': (C, '_real4_div_rn(mul(F(6.2831853071795862), sub(call_julian, F(1))), F(365))'),
    'solar_midpoint60': (C, '_real4_div_rn(F(solar_lead_seconds), F(60))'),
    'solar_local60': (C, '_real4_div_rn(xt24, F(60))'),
    'solar_lon15': (C, '_real4_div_rn(jnp.asarray(lon, F), F(15))'),
    'solar_solcon365': (C, 'div(mul(call_julian, F(360)), F(365))'),
    'ozone_lead60': (O, '_real4_div_rn(lead_seconds, 60.0)'),
    'ozone_cyclic1early': (O, '_real4_div_rn(cday_plus - day, delta_cyclic)'),
    'ozone_cyclic2early': (O, '_real4_div_rn(day + jnp.asarray(365.0, dtype=dtype) - cday_minus, delta_cyclic)'),
    'ozone_linear1': (O, '_real4_div_rn(cday_plus - day, delta)'),
    'ozone_linear2': (O, '_real4_div_rn(day - cday_minus, delta)'),
}
HOST = {'solar_degrad', 'solar_dpd'}


def spans(root):
    result = {}
    for family, (file, expression) in PATTERNS.items():
        source = (Path(root) / file).read_text()
        expected = ast.dump(ast.parse(expression, mode='eval').body)
        found = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.expr) and ast.dump(node) == expected]
        assert found, ('missing protected source expression', family)
        result[family] = [(file, n.lineno, n.col_offset, n.end_lineno, n.end_col_offset) for n in found]
    return result


def debug_from_proto(proto):
    table = next(v for f, w, v in _proto_fields(proto) if f == 17)
    parts = list(_proto_fields(table))
    debug = dict(files=[''] + [v.decode() for f, w, v in parts if f == 1],
        functions=[''] + [v.decode() for f, w, v in parts if f == 2],
        locations=[{}] + [{f: v for f, w, v in _proto_fields(x)} for ff, w, x in parts if ff == 3],
        frames=[{}] + [{f: v for f, w, v in _proto_fields(x)} for ff, w, x in parts if ff == 4], metadata={})
    for f, w, comp in _proto_fields(proto):
        if f != 3 or w != 2:
            continue
        name = next(v.decode() for ff, ww, v in _proto_fields(comp) if ff == 1)
        for ff, ww, ins in _proto_fields(comp):
            if ff != 2 or ww != 2:
                continue
            values = dict((k, v) for k, _, v in _proto_fields(ins))
            if 7 not in values:
                continue
            meta = dict((k, v) for k, _, v in _proto_fields(values[7]))
            debug['metadata'][(name, values[1].decode())] = meta
    return debug


def paths(debug, meta):
    fid = meta.get(15, 0)
    seen = set()
    result = []
    while fid and fid not in seen:
        seen.add(fid)
        frame = debug['frames'][fid]
        loc = debug['locations'][frame.get(1, 0)]
        filename = debug['files'][loc.get(1, 0)]
        if '/gpuwrf/' in filename:
            filename = 'src/gpuwrf/' + filename.split('/gpuwrf/', 1)[1]
            result.append(dict(file=filename, line=loc.get(3), column=loc.get(4),
                end_line=loc.get(5, loc.get(3)), end_column=loc.get(6),
                function=debug['functions'][loc.get(2, 0)]))
        fid = frame.get(2, 0)
    return result


def overlaps(frame, span):
    file, start, column, end, end_column = span
    if frame['file'] != file or frame['line'] is None:
        return False
    a = (frame['line'], (frame['column'] or 1) - 1)
    b = (frame['end_line'] or frame['line'], (frame['end_column'] or frame['column'] or 1) - 1)
    return a < (end, end_column) and b >= (start, column)


def audit(proto, text, root, require=None):
    registry = spans(root)
    debug = debug_from_proto(proto)
    records = []
    family_records = defaultdict(list)
    instructions = {}
    computation = ''
    for line in text.splitlines():
        match = re.match(r'(?:ENTRY )?%?([^ (]+)(?: \(| \{)', line)
        if match:
            computation = match[1]
        node = re.match(r'\s+(?:ROOT )?%?([^ =]+) = ', line)
        if node:
            instructions[(computation, node[1])] = line
        match = re.match(r'\s+(?:ROOT )?%?([^ =]+) = (f32|f64)\[([^]]*)\](?:\{[^}]*\})? (divide|convert)\(', line)
        if not match:
            continue
        name, dtype, shape, opcode = match.groups()
        frames = paths(debug, debug['metadata'].get((computation, name), {}))
        matched = [family for family, ss in registry.items()
                   if any(overlaps(frame, span) for frame in frames for span in ss)]
        if opcode == 'divide':
            assert not (HOST & set(matched)), ('host RN constant regressed to device division', matched, line)
        if not any(f['function'] == '_real4_div_rn' for f in frames):
            continue
        record = dict(computation=computation, instruction=name, dtype=dtype,
                      shape=shape, opcode=opcode, families=matched, frames=frames,
                      elements=math.prod(int(d) for d in shape.split(',') if d))
        records.append(record)
        if dtype == 'f64' and opcode == 'divide':
            assert matched, ('unattributed RN division', record)
            for family in matched:
                family_records[family].append(record)
    # CSE shares solar and ozone lead/60 in the full root. Require a direct
    # ozone-clock consumer of that SAME narrowed solar quotient, rather than
    # claiming native absence is proof that the source division is unreachable.
    aliases = []
    if 'ozone_lead60' not in family_records and 'solar_lead60' in family_records:
        for record in records:
            if not (record['dtype'] == 'f32' and record['opcode'] == 'convert'
                    and 'solar_lead60' in record['families']):
                continue
            token = '%' + record['instruction']
            for (comp, name), line in instructions.items():
                if comp != record['computation'] or not re.search(r'\badd\(', line):
                    continue
                if token not in re.findall(r'%[^ ,)]+', line):
                    continue
                consumers = paths(debug, debug['metadata'].get((comp, name), {}))
                if any(f['function'] == 'wrf_ozn_time_indices_factors' and 91 <= f['line'] <= 95 for f in consumers):
                    aliases.append(dict(family='ozone_lead60', shared_with='solar_lead60',
                        computation=comp, narrowed_quotient=record['instruction'],
                        direct_ozone_consumer=name, consumer_instruction=line.strip()))
                    family_records['ozone_lead60'] = family_records['solar_lead60']
    if 'ozone_lead60' not in family_records and 'solar_lead60' in family_records:
        # Swiss puts the narrowing in a separate fusion. Resolve scalar
        # parameter -> fusion argument -> called root edges explicitly.
        nodes = {}
        roots = {}
        callers = defaultdict(list)
        for key, line in instructions.items():
            rhs = line.split('=', 1)[1]
            op = re.search(r'\b([\w-]+)\(([^)]*)\)', rhs)
            if not op:
                continue
            args = re.findall(r'%([^ ,)]+)', op[2])
            called = re.search(r'calls=%([^ ,]+)', line)
            nodes[key] = (op[1], args, op[2], called[1] if called else None)
            if line.lstrip().startswith('ROOT '):
                roots[key[0]] = key[1]
            if called:
                callers[called[1]].append((key, args))
        targets = {(r['computation'], r['instruction']) for r in family_records['solar_lead60']}
        def upstream(key, seen=frozenset()):
            if key in seen or len(seen) > 24:
                return None
            if key in targets:
                return [key]
            if key not in nodes:
                return None
            op, args, literal, called = nodes[key]
            candidates = []
            if op in ('convert', 'copy', 'bitcast', 'reshape'):
                candidates = [(key[0], arg) for arg in args]
            elif op == 'parameter':
                index = int(literal)
                candidates = [(parent[0], operands[index]) for parent, operands in callers[key[0]]
                              if index < len(operands)]
            elif op == 'fusion' and called in roots:
                candidates = [(called, roots[called])]
            for candidate in candidates:
                path = upstream(candidate, seen | {key})
                if path:
                    return [key] + path
            return None
        for key, line in instructions.items():
            if key not in nodes or nodes[key][0] != 'add':
                continue
            frames = paths(debug, debug['metadata'].get(key, {}))
            if not any(f['function'] == 'wrf_ozn_time_indices_factors' and 91 <= f['line'] <= 95 for f in frames):
                continue
            for arg in nodes[key][1]:
                dependency = upstream((key[0], arg))
                if dependency:
                    aliases.append(dict(family='ozone_lead60', shared_with='solar_lead60',
                        computation=key[0], direct_ozone_consumer=key[1],
                        consumer_instruction=line.strip(),
                        exact_dependency_path=[dict(computation=k[0], instruction=k[1],
                            hlo=instructions[k].strip()) for k in dependency]))
                    family_records['ozone_lead60'] = family_records['solar_lead60']
    needed = set(require) if require is not None else set(PATTERNS) - HOST
    missing = sorted(needed - set(family_records))
    assert not missing, ('missing protected f64 divide provenance', missing, sorted(family_records))
    return dict(full_constant_proto_sha256=hashlib.sha256(proto).hexdigest(), records=records, CSE_aliases=aliases,
                family_counts={f: len(v) for f, v in sorted(family_records.items())},
                f64_divides=sum(r['dtype']=='f64' and r['opcode']=='divide' for r in records),
                f64_operand_converts=sum(r['dtype']=='f64' and r['opcode']=='convert' for r in records),
                f32_result_converts=sum(r['dtype']=='f32' and r['opcode']=='convert' for r in records),
                f64_divide_elements=sum(r['elements'] for r in records if r['dtype']=='f64' and r['opcode']=='divide'))
