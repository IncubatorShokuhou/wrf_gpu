"""All common CPU-WRF filtered-file variables, finite first; primary frozen DFI gates.

usage: python compare_filtered.py port_wrfinput cpu_initialized raw_input out.json
Primary residual-fraction cap .10 (includes inherited P .051), diagnosed surface Q2
cap .25; roundoff floors are 8 float32 eps times the reference RMS (theta uses its
300 K storage offset). Known T/QKE/SNOW, history-only and static residuals are reported
separately, as in the accepted o2-dfi evidence; no whole-state machine-precision claim.
"""
import hashlib
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

PRIMARY = ('U', 'V', 'W', 'PH', 'MU', 'THM', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE',
           'QSNOW', 'QGRAUP', 'QNICE', 'QNRAIN', 'P', 'AL', 'P_HYD',
           'TSLB', 'SMOIS', 'LAI', 'TSK', 'T2', 'PSFC', 'U10', 'V10', 'Q2')


def compare(port, oracle, raw):
    fields = {}
    failures = []
    with netCDF4.Dataset(port) as p, netCDF4.Dataset(oracle) as w, netCDF4.Dataset(raw) as r:
        missing = sorted(set(w.variables) - set(p.variables))
        for name in sorted(set(w.variables) & set(p.variables)):
            if not np.issubdtype(w[name].dtype, np.number):
                continue
            a, b = (np.asarray(ds[name][:], np.float64) for ds in (p, w))
            c = np.asarray(r[name][:], np.float64) if name in r.variables else None
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all()
                          and (c is None or np.isfinite(c).all()))
            if not finite:
                failures.append(name + ':nonfinite')
                fields[name] = {'finite': False}
                continue
            if a.shape != b.shape or c is not None and b.shape != c.shape:
                failures.append(name + ':shape')
                continue
            rmse = float(np.sqrt(np.mean((a-b)**2)))
            inc = float(np.sqrt(np.mean((b-c)**2))) if c is not None else 0.
            scale = float(np.sqrt(np.mean(b*b)))
            floor = 8 * np.finfo(np.float32).eps * max(scale, 300. if name in ('THM', 'T') else 0.)
            cap = .25 if name == 'Q2' else .10
            limit = max(cap*inc, floor)
            rec = dict(finite=finite, rmse=rmse, max_abs=float(np.abs(a-b).max()),
                       wrf_increment_rmse=inc, residual_fraction=rmse/inc if inc else None,
                       exact=bool(np.array_equal(a,b)), primary=name in PRIMARY)
            if name in PRIMARY:
                rec.update(limit=limit, passed=rmse <= limit)
                if not rec['passed']:
                    failures.append(name)
            if name == "RAD_TTEN_DFI" and not rec["exact"]:
                failures.append(name + ':nonzero-radar')
            fields[name] = rec
        missing_primary = sorted(set(PRIMARY) - set(fields))
        # Some schemes lack optional number/hydrometeor variables in both files.
        missing_primary = [n for n in missing_primary if n in w.variables]
        failures.extend(n+':missing' for n in missing_primary)
    return dict(fields=fields, missing_vs_wrf=missing, failures=failures,
                primary_passed=not failures and not missing,
                numeric_fields_compared=len(fields), exact_fields=sum(x.get('exact',False) for x in fields.values()))


if __name__ == '__main__':
    port, oracle, raw, out = sys.argv[1:]
    result = compare(port, oracle, raw)
    result['inputs'] = {key: {'path': value, 'sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest()}
                        for key, value in [('port',port),('oracle',oracle),('raw',raw)]}
    Path(out).write_text(json.dumps(result, indent=1)+'\n')
    for n, r in result['fields'].items():
        if r.get('primary'):
            print(n, {k:r[k] for k in ('rmse','residual_fraction','passed')})
    print('ALL FIELDS:', result['numeric_fields_compared'], 'EXACT:', result['exact_fields'],
          'PRIMARY PASS:',result['primary_passed'], 'FAILURES:', result['failures'])
    sys.exit(0 if result['primary_passed'] else 1)
