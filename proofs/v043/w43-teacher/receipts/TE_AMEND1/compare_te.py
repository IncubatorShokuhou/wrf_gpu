#!/usr/bin/env python3
"""Compare a reproduced teacher extract against an ALISIOS reference extract (array by array)."""
import json, os, sys, hashlib
import numpy as np
import pyarrow.parquet as pq

ref, ours, outp = sys.argv[1], sys.argv[2], sys.argv[3]
NPZ = ['d01_crop_sfc.npz', 'd02_crop_sfc.npz', 'd03_sfc.npz']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def cmp_arr(a, b):
    r = {'ref_dtype': str(a.dtype), 'our_dtype': str(b.dtype), 'ref_shape': list(a.shape), 'our_shape': list(b.shape)}
    if a.shape != b.shape or a.dtype != b.dtype:
        r.update(exact_equal=False, note='shape/dtype mismatch')
        return r
    if a.dtype.kind == 'f':
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        diff = (a != b) & ~(nan_a & nan_b)
        both = ~nan_a & ~nan_b
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))[both]
        r['max_abs_diff'] = float(d.max()) if d.size else 0.0
        r['nan_ref'] = int(nan_a.sum()); r['nan_ours'] = int(nan_b.sum())
        r['nan_pattern_equal'] = bool((nan_a == nan_b).all())
        r['bitwise_equal_non_nan'] = bool(np.array_equal(a[both].view(np.uint32 if a.itemsize == 4 else np.uint64),
                                                          b[both].view(np.uint32 if a.itemsize == 4 else np.uint64)))
    else:
        diff = a != b
        r['max_abs_diff'] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0
    r['n_differing'] = int(diff.sum())
    r['exact_equal'] = r['n_differing'] == 0 and r.get('bitwise_equal_non_nan', True)
    return r


out = {'reference': ref, 'ours': ours, 'npz': {}, 'profile': {}, 'files': {}}
for f in NPZ + ['POINTS.parquet', 'points_profile.parquet']:
    out['files'][f] = {'ref_sha256': sha(os.path.join(ref, f)), 'our_sha256': sha(os.path.join(ours, f))}
    out['files'][f]['file_bytes_identical'] = out['files'][f]['ref_sha256'] == out['files'][f]['our_sha256']
for f in NPZ:
    zr, zo = np.load(os.path.join(ref, f)), np.load(os.path.join(ours, f))
    d = {'key_order_equal': list(zr.files) == list(zo.files), 'arrays': {}}
    for k in zr.files:
        d['arrays'][k] = cmp_arr(zr[k], zo[k]) if k in zo.files else {'exact_equal': False, 'note': 'missing'}
    out['npz'][f] = d
pr, po = pq.ParquetFile(os.path.join(ref, 'points_profile.parquet')), pq.ParquetFile(os.path.join(ours, 'points_profile.parquet'))
P = out['profile']
P['schema_equal'] = pr.schema_arrow.equals(po.schema_arrow)
P['rows'] = [pr.metadata.num_rows, po.metadata.num_rows]
P['row_groups'] = [pr.num_row_groups, po.num_row_groups]
cols = {}
key_cols = ['phys_id', 'role', 'issue', 'attempt', 'valid_utc', 'actual_utc', 'lead_h', 'tau_h', 'level', 'grid_y', 'grid_x']
for i in range(min(pr.num_row_groups, po.num_row_groups)):
    tr, to = pr.read_row_group(i), po.read_row_group(i)
    for c in tr.column_names:
        a = tr.column(c).to_numpy(zero_copy_only=False)
        b = to.column(c).to_numpy(zero_copy_only=False) if c in to.column_names else None
        acc = cols.setdefault(c, {'n_differing': 0, 'max_abs_diff': 0.0, 'exact_equal': True})
        if b is None or len(a) != len(b):
            acc['exact_equal'] = False; acc['note'] = 'missing/length'; continue
        if a.dtype == object:
            nd = int((a != b).sum())
        else:
            r = cmp_arr(a, b); nd = r['n_differing']
            acc['max_abs_diff'] = max(acc['max_abs_diff'], r['max_abs_diff'])
            if not r['exact_equal']:
                acc['exact_equal'] = False
        acc['n_differing'] += nd
        if nd:
            acc['exact_equal'] = False
P['columns'] = cols
P['row_order_and_keys_equal'] = all(cols[c]['exact_equal'] for c in key_cols if c in cols)
summ = {'npz_exact': [], 'npz_not_exact': [], 'profile_exact': [], 'profile_not_exact': []}
for f, d in out['npz'].items():
    for k, r in d['arrays'].items():
        (summ['npz_exact'] if r['exact_equal'] else summ['npz_not_exact']).append(f'{f}:{k}')
for c, r in cols.items():
    (summ['profile_exact'] if r['exact_equal'] else summ['profile_not_exact']).append(c)
summ['counts'] = {k: len(v) for k, v in summ.items() if isinstance(v, list)}
out['summary'] = summ
json.dump(out, open(outp, 'w'), indent=1)
print(json.dumps({'counts': summ['counts'], 'npz_not_exact': {x: {kk: out['npz'][x.split(':')[0]]['arrays'][x.split(':')[1]].get(kk) for kk in ('n_differing', 'max_abs_diff', 'nan_ref', 'nan_ours')} for x in summ['npz_not_exact']},
                  'profile_not_exact': {c: cols[c] for c in summ['profile_not_exact']}, 'files': out['files'],
                  'key_order': {f: d['key_order_equal'] for f, d in out['npz'].items()}, 'profile_schema_equal': P['schema_equal'], 'rows': P['rows']}, indent=1))
