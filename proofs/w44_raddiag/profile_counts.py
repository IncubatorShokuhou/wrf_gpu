"""Summarize actual nsys graph-node kernels/memcpy/memset rows, normalized by captured work."""
import argparse
import json
from pathlib import Path
import sqlite3


def counts(database, receipt):
    run = json.loads(Path(receipt).read_text())
    assert run["device"]["platform"] == "gpu" and run["rc"] == 0
    segment = run["derived"]["segments"][1]
    assert segment["k"] == 2 and segment["compile_s_in_call"] == 0
    assert not any("profiler" in error for error in run["errors"]), run["errors"]
    scale = 3600 / (segment["root_steps"] * 54)
    with sqlite3.connect(database) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("missing actual CUDA kernel activities")
        out = dict(kernels=db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0])
        if out['kernels'] <= 0:
            raise ValueError('empty kernel capture')
        if 'NVTX_EVENTS' not in tables:
            raise ValueError('missing captured NVTX work scope')
        columns = {row[1] for row in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
        scope = db.execute("SELECT count(*) FROM NVTX_EVENTS WHERE text LIKE '%S0:segment:2%'").fetchone()[0] if 'text' in columns else 0
        if not scope and 'textId' in columns and 'StringIds' in tables:
            scope = db.execute("SELECT count(*) FROM NVTX_EVENTS n JOIN StringIds s ON n.textId=s.id WHERE s.value LIKE '%S0:segment:2%'").fetchone()[0]
        if not scope:
            raise ValueError('capture does not attest the second forecast segment')
        out['segment2_nvtx_ranges'] = scope
        for key, table in (("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"), ("memset", "CUPTI_ACTIVITY_KIND_MEMSET")):
            out[key] = db.execute('SELECT count(*) FROM '+table).fetchone()[0] if table in tables else 0
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            out["copy_kind"] = [dict(kind=int(k), rows=int(n), bytes=int(b)) for k, n, b in db.execute(
                'SELECT copyKind,count(*),sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')]
        out["memcpy_enums"] = {name: list(db.execute('SELECT * FROM "'+name+'"')) for name in tables
                               if name.startswith('ENUM_') and 'MEMCPY' in name}
    out.update(root_steps=segment['root_steps'], duration_fc_s=segment['root_steps']*54,
               per_fc_hour={name: out[name]*scale for name in ('kernels','memcpy','memset')},
               total_backend_compile_s=sum(e['s'] for e in run['compile_events'] if 'backend_compile' in e['event']))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    result = {name: counts(str(args.root/f'profile_{name}.sqlite'),
                           str(args.root/f'profile_{name}/run/receipt.json')) for name in ('off','on')}
    result['on_minus_off_per_fc_hour'] = {key: result['on']['per_fc_hour'][key]-result['off']['per_fc_hour'][key]
                                        for key in ('kernels','memcpy','memset')}
    carry = {name: json.loads((args.root/f'profile_{name}/run/CARRY_HASHES.json').read_text())
             for name in ('off','on')}
    expected = {f'd0{n}:{lead}' for n in (1,2,3) for lead in (0,10800)}
    if any(set(value['records']) != expected for value in carry.values()):
        raise ValueError('missing full-carry domain/initial/final coverage')
    differences = {}
    for frame in sorted(expected):
        left, right = (carry[name]['records'][frame] for name in ('off','on'))
        if left['kind'] != 'OperationalCarry' or right['kind'] != 'OperationalCarry':
            raise ValueError('full-carry kind differs')
        if left['state_fields'] != right['state_fields'] or set(left['leaves']) != set(right['leaves']):
            raise ValueError('full-carry schema differs')
        differences[frame] = [key for key in left['leaves'] if left['leaves'][key] != right['leaves'][key]]
    result['full_carry_byte_identical'] = all(not value for value in differences.values())
    result['full_carry_differences'] = differences
    args.output.write_text(json.dumps(result, indent=2)+'\n')
