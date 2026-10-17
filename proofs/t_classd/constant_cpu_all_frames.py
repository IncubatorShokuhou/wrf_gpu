"""Original CPU-WRF bit census of WRF-assigned constant history fields, every cell of every frame.

Usage: python proofs/t_classd/constant_cpu_all_frames.py <out.json> <case_dir>:<FIELD,FIELD> [...]
Records per frame: file SHA-256, dims, dtype, metadata, count of +0 bits and of any other bit pattern.
"""
import glob, hashlib, json, sys
import numpy as np, netCDF4

out, cases = sys.argv[1], {}
for spec in sys.argv[2:]:
    case_dir, names = spec.split(':')
    files = sorted(glob.glob(f'{case_dir}/cpu/wrfout_d01_*'))
    frames, fields = [], {}
    for path in files:
        rec = dict(file=path.split('/')[-1], sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest(), fields={})
        with netCDF4.Dataset(path) as ds:
            attrs = {k: str(ds.getncattr(k)) for k in ('MP_PHYSICS', 'CU_PHYSICS', 'RA_SW_PHYSICS', 'SF_URBAN_PHYSICS')
                     if k in ds.ncattrs()}
            for name in names.split(','):
                v = ds[name]
                bits = np.asarray(v[:], np.float32).view(np.uint32)
                rec['fields'][name] = dict(bits=0 if np.all(bits == 0) else None, all_equal_bits=bool(np.all(bits == bits.flat[0])),
                                           nonzero_bits=int(np.count_nonzero(bits)))
                fields.setdefault(name, dict(dimensions=list(v.dimensions), dtype=str(v.dtype),
                                             **{k: str(v.getncattr(k)) for k in v.ncattrs()}))
        frames.append(rec)
    ok = all(f['fields'][n]['bits'] == 0 for f in frames for n in names.split(','))
    cases[case_dir.rstrip('/').split('/')[-1]] = dict(global_attributes=attrs, fields=fields, frames=frames,
                                                      all_frames_zero=ok, frame_count=len(frames))
json.dump(dict(cases=cases), open(out, 'w'), indent=1)
print({k: (v['frame_count'], v['all_frames_zero'], v['global_attributes']) for k, v in cases.items()})
