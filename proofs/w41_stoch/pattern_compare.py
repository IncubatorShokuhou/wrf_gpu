"""Compare WRF stochastic history fields GPU vs original CPU-WRF, every paired frame (lane w41-stoch).

NumPy/netCDF4 only (no JAX). Fields = every WRF Registry.stoch history variable present in the CPU file
(native pattern fields, field_* SPP fields, SPP spectral coefficients SPFORCC/S2-4, ISEEDARR_* seeds).
Real fields: max|gpu-cpu| / max|cpu| per frame against the u2-stoch pre-registered bound 8e-6 + 64 eps(float32)
(ST18/ST40, original-WRF pattern gates); seeds must be EXACT. Missing-on-GPU fields are reported, never skipped.
usage: python pattern_compare.py <cpu_dir> <gpu_dir> <out.json> [domain]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import netCDF4

STOCH = ('RU_TENDF_STOCH', 'RV_TENDF_STOCH', 'RT_TENDF_STOCH', 'RSTOCH', 'RAND_PERT',
         'PATTERN_SPP_CONV', 'PATTERN_SPP_PBL', 'PATTERN_SPP_LSM', 'FIELD_CONV', 'FIELD_PBL', 'FIELD_SF',
         'SPFORCC2', 'SPFORCS2', 'SPFORCC3', 'SPFORCS3', 'SPFORCC4', 'SPFORCS4')
SEEDS = ('ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'ISEEDARR_RAND_PERTURB', 'ISEEDARRAY_SPP_CONV',
         'ISEEDARRAY_SPP_PBL', 'ISEEDARRAY_SPP_LSM')
BOUND = 8e-6 + 64 * float(np.finfo(np.float32).eps)


def main():
    cpu_dir, gpu_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dom = sys.argv[4] if len(sys.argv) > 4 else 'd01'
    cpu_files = sorted(cpu_dir.glob(f'wrfout_{dom}_*'))
    assert cpu_files, cpu_dir
    rows, fields, seeds, missing, incomplete = [], {}, {}, set(), []
    for cf in cpu_files:
        gf = gpu_dir / cf.name
        if not gf.exists():
            incomplete.append(cf.name)
            continue
        with netCDF4.Dataset(cf) as c, netCDF4.Dataset(gf) as g:
            c.set_auto_mask(False); g.set_auto_mask(False)
            for name in STOCH:
                if name not in c.variables:
                    continue
                if name not in g.variables:
                    missing.add(name); continue
                a = np.asarray(c[name][:], np.float64); b = np.asarray(g[name][:], np.float64)
                r = fields.setdefault(name, dict(frames=0, shape=list(a.shape), worst_rel=0.0, worst_abs=0.0,
                                                 max_ref=0.0, nonfinite_cpu=0, nonfinite_gpu=0, shape_mismatch=0,
                                                 zero_ref_frames=0, fail_frames=[]))
                r['frames'] += 1
                if a.shape != b.shape:
                    r['shape_mismatch'] += 1; r['fail_frames'].append(cf.name); continue
                nc, ng = int((~np.isfinite(a)).sum()), int((~np.isfinite(b)).sum())
                r['nonfinite_cpu'] += nc; r['nonfinite_gpu'] += ng
                if nc or ng:
                    r['fail_frames'].append(cf.name); continue
                ref = float(np.abs(a).max()); d = float(np.abs(a - b).max())
                rel = d / ref if ref > 0 else (0.0 if d == 0 else float('inf'))
                r['zero_ref_frames'] += int(ref == 0)
                r['worst_abs'] = max(r['worst_abs'], d); r['max_ref'] = max(r['max_ref'], ref)
                r['worst_rel'] = max(r['worst_rel'], rel)
                if rel > BOUND:
                    r['fail_frames'].append(cf.name)
            for name in SEEDS:
                if name not in c.variables:
                    continue
                if name not in g.variables:
                    missing.add(name); continue
                a = np.asarray(c[name][:]); b = np.asarray(g[name][:])
                s = seeds.setdefault(name, dict(frames=0, exact_frames=0, words=int(a.size), fail_frames=[]))
                s['frames'] += 1
                if a.shape == b.shape and np.array_equal(a, b):
                    s['exact_frames'] += 1
                else:
                    s['fail_frames'].append(cf.name)
        rows.append(cf.name)
    fail = bool(missing or incomplete or any(r['fail_frames'] for r in fields.values())
                or any(s['fail_frames'] for s in seeds.values()))
    result = dict(schema='w41-stoch-pattern-compare-v1', cpu_dir=str(cpu_dir), gpu_dir=str(gpu_dir), domain=dom,
                  bound_rel=BOUND, bound_source='u2-stoch ST18/ST40 pre-registered 8e-6+64eps maxref',
                  paired_frames=len(rows), incomplete=incomplete, missing_on_gpu=sorted(missing),
                  fields=fields, seeds=seeds, verdict='FAIL' if fail else 'PASS',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference='original CPU-WRF; GPU-vs-GPU is never fidelity evidence')
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(verdict=result['verdict'], paired=len(rows), missing=sorted(missing),
                          worst={k: (v['worst_rel'], len(v['fail_frames']), v['zero_ref_frames']) for k, v in fields.items()},
                          seeds={k: (v['exact_frames'], v['frames']) for k, v in seeds.items()}), indent=1))


if __name__ == '__main__':
    main()
