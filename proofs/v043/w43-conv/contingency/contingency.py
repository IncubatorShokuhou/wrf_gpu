#!/usr/bin/env python3
"""ALISIOS acceptance amendment 1 §3 contingency tables (CPU only, read-only on teacher extracts).

Frozen rule (cross/FROM_WRF_GPU2_V043_AMEND1_ACK_20261021.md §3), not changed here:
  events  (a) rain_rate_mm_h >= 0.1   (b) LWP >= 0.01 kg/m2 (proxy: LWP = column QCLOUD+QRAIN, disclosed)
  domains d01_crop, d02_crop, d03; tau 1..72; classes per cell x hour: both / CPU-only / GPU-only / neither
  ratio   R = GPU-only / CPU-only per case (pooled over domains+hours) and pooled over all cases (sum/sum)
  verdict symmetric (pooled R in [0.8, 1.25] AND no case outside [0.5, 2]) -> NOT_A_BUG, else FINDING with size
Diagnostics (not part of the verdict): +-1 h same-cell hits, +-1 cell same-hour hits, +-1 h x +-1 cell hits,
first-onset shift per cell, distribution by rate class.
Non-finite rule (extract readback rule, E200): rain_rate_mm_h may be NaN only at tau 0; LWP never; any other
non-finite value in a counted array -> the pair is REJECTED (raises), never silently classified.
"""
import json, os, sys, hashlib
import numpy as np

T = '<USER_HOME>/wrf_gpu2_lanes/w43-teacher'
TWIN = '<USER_HOME>/wrf_gpu2_lanes/w43-conv/contingency/twin0227_cpu'
WN3 = ['20260120', '20260220', '20260227', '20260502', '20260608', '20260614']
ERA5 = ['20240218', '20240302']
DOMS = ['d01_crop_sfc.npz', 'd02_crop_sfc.npz', 'd03_sfc.npz']
FAMS = {'rain': ('rain_rate_mm_h', 0.1, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]),
        'lwp': ('LWP', 0.01, [0.01, 0.05, 0.1, 0.5, 1.0, np.inf])}
TAU = np.arange(73)
SYM_POOLED = (0.8, 1.25)
SYM_CASE = (0.5, 2.0)


class Reject(Exception):
    pass


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def pairs():
    out = [(f'wn3_{c}', f'{T}/te_repro/{c}_cpu', f'{T}/te_gpu/{c}_gpu') for c in WN3]
    out += [(f'era5_{c}', f'{T}/te_cpu/{c}_cpu', f'{T}/te_gpu/{c}_gpu') for c in ERA5]
    return out


def readback_ok(d):
    rb = json.load(open(os.path.join(d, 'READBACK.json')))
    bad = [k for k, v in rb['domains'].items() if v.get('foreign_nonfinite_total', 1) != 0 or v.get('frames') != 73]
    if rb['failures'] or bad:
        raise Reject(f'READBACK {d}: failures={rb["failures"]} bad_domains={bad}')
    return True


def load_field(z, key, where):
    """Return (73,ny,nx) float64 with the non-finite rule enforced."""
    a = np.asarray(z[key])
    if a.ndim != 3 or a.shape[0] != 73:
        raise Reject(f'{where}/{key}: shape {a.shape}')
    a = a.astype(np.float64)
    fin = np.isfinite(a)
    if key == 'rain_rate_mm_h':
        if not np.isnan(a[0]).all():  # readback rule (cf54 :453-454 np.isnan): NaN only, never finite or Inf
            raise Reject(f'{where}/{key}: tau0 expected all-NaN, {int((~np.isnan(a[0])).sum())} not NaN')
        if not fin[1:].all():
            raise Reject(f'{where}/{key}: {int((~fin[1:]).sum())} non-finite at tau 1..72')
    elif not fin.all():
        raise Reject(f'{where}/{key}: {int((~fin).sum())} non-finite')
    return a


def check_pair_meta(zc, zg, where):
    if not (np.array_equal(zc['tau_h'], TAU) and np.array_equal(zg['tau_h'], TAU)):
        raise Reject(f'{where}: tau_h not 0..72 on both sides')
    if not np.array_equal(zc['lead_h'], TAU + 6):  # lead_h is issue-relative (18z issue, t0 = 00z next day)
        raise Reject(f'{where}: lead_h != tau_h + 6')
    for k in ('lead_h', 'actual_time_unix_s', 'XLAT', 'XLONG'):
        if not np.array_equal(zc[k], zg[k]):
            raise Reject(f'{where}: {k} differs')


def dilate(e, dt, ds):
    """Boolean max-filter over time (+-dt) and space (+-ds, 3x3 for ds=1); no wrap-around."""
    out = e.copy()
    if ds:
        p = np.pad(e, ((0, 0), (ds, ds), (ds, ds)))
        ny, nx = e.shape[1:]
        for oy in range(-ds, ds + 1):
            for ox in range(-ds, ds + 1):
                out |= p[:, ds + oy:ds + oy + ny, ds + ox:ds + ox + nx]
    if dt:
        base = out.copy()
        for o in range(1, dt + 1):
            out[o:] |= base[:-o]
            out[:-o] |= base[o:]
    return out


def first_onset(e):
    """Index (hour) of the first event per cell, -1 if none."""
    any_ = e.any(axis=0)
    idx = np.argmax(e, axis=0)
    return np.where(any_, idx, -1)


def table(c, g, thr, bins):
    """c, g: (72,ny,nx) at tau 1..72."""
    ec, eg = c >= thr, g >= thr
    both, co, go = ec & eg, ec & ~eg, eg & ~ec
    r = {'both': int(both.sum()), 'cpu_only': int(co.sum()), 'gpu_only': int(go.sum()),
         'neither': int((~ec & ~eg).sum()), 'cells_x_hours': int(ec.size)}
    # shift diagnostics: an X-only cell-hour is a 'hit' if the other side has an event within the window
    for name, (dt, ds) in {'pm1h': (1, 0), 'pm1cell': (0, 1), 'pm1h_pm1cell': (1, 1)}.items():
        dg, dc = dilate(eg, dt, ds), dilate(ec, dt, ds)
        r[f'cpu_only_hit_{name}'] = int((co & dg).sum())
        r[f'gpu_only_hit_{name}'] = int((go & dc).sum())
    fc, fg = first_onset(ec), first_onset(eg)
    m = (fc >= 0) & (fg >= 0)
    sh = (fg - fc)[m]
    r['first_onset'] = {'cells_both': int(m.sum()), 'sum_shift_h': int(sh.sum()),
                        'cells_cpu_only_ever': int(((fc >= 0) & (fg < 0)).sum()),
                        'cells_gpu_only_ever': int(((fg >= 0) & (fc < 0)).sum())}
    r['by_class'] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = f'[{lo},{hi})'
        r['by_class'][k] = {'cpu_only': int((co & (c >= lo) & (c < hi)).sum()),
                            'gpu_only': int((go & (g >= lo) & (g < hi)).sum()),
                            'both_cpu_rate': int((both & (c >= lo) & (c < hi)).sum())}
    # other side's value at X-only cell-hours, as a fraction of the threshold (near-threshold straddling vs real absence)
    edges = [0.0, 0.1, 0.5, 0.8, 1.0]
    r['other_side_frac_of_thr'] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = f'[{lo},{hi})'
        r['other_side_frac_of_thr'][k] = {'gpu_at_cpu_only': int((co & (g >= lo * thr) & (g < hi * thr)).sum()),
                                          'cpu_at_gpu_only': int((go & (c >= lo * thr) & (c < hi * thr)).sum())}
    r['_shifts'] = sh.tolist()
    return r


def add(acc, r):
    for k, v in r.items():
        if k == '_shifts':
            acc.setdefault(k, []).extend(v)
        elif isinstance(v, dict):
            add(acc.setdefault(k, {}), v)
        else:
            acc[k] = acc.get(k, 0) + v


def ratio(go, co):
    if co == 0:
        return None if go == 0 else float('inf')
    return go / co


def finalize(acc):
    s = acc.pop('_shifts', [])
    acc['ratio_gpu_only_over_cpu_only'] = ratio(acc['gpu_only'], acc['cpu_only'])
    acc['excess_gpu_minus_cpu_only'] = acc['gpu_only'] - acc['cpu_only']
    acc['first_onset']['mean_shift_h_gpu_minus_cpu'] = float(np.mean(s)) if s else None
    acc['first_onset']['median_shift_h'] = float(np.median(s)) if s else None
    acc['first_onset']['frac_shift_within_pm1h'] = float(np.mean(np.abs(s) <= 1)) if s else None
    return acc


def run_pair(case, cdir, gdir):
    readback_ok(cdir); readback_ok(gdir)
    res = {'cpu_dir': cdir, 'gpu_dir': gdir, 'domains': {}, 'files': {}}
    tot = {f: {} for f in FAMS}
    for dn in DOMS:
        pc, pg = os.path.join(cdir, dn), os.path.join(gdir, dn)
        res['files'][dn] = {'cpu_sha256': sha(pc), 'gpu_sha256': sha(pg)}
        zc, zg = np.load(pc), np.load(pg)
        where = f'{case}/{dn}'
        check_pair_meta(zc, zg, where)
        dres = {}
        neg = {'cpu': int(np.asarray(zc['rain_negative_flag']).sum()), 'gpu': int(np.asarray(zg['rain_negative_flag']).sum())}
        for fam, (key, thr, bins) in FAMS.items():
            c = load_field(zc, key, where + '/cpu')[1:]
            g = load_field(zg, key, where + '/gpu')[1:]
            r = table(c, g, thr, bins)
            add(tot[fam], r)
            r2 = {}
            add(r2, r)  # deep copy (add() rebuilds nested dicts and lists)
            dres[fam] = finalize(r2)
        dres['rain_negative_flag_count'] = neg
        res['domains'][dn.split('_sfc')[0]] = dres
    res['total'] = {f: finalize(tot[f]) for f in FAMS}
    return res


def verdict(per_case, pooled):
    out = {}
    for fam in FAMS:
        R = pooled[fam]['ratio_gpu_only_over_cpu_only']
        cases = {c: v['total'][fam]['ratio_gpu_only_over_cpu_only'] for c, v in per_case.items()}
        outside = {c: r for c, r in cases.items() if r is not None and not (SYM_CASE[0] <= r <= SYM_CASE[1])}
        undefined = [c for c, r in cases.items() if r is None]
        pooled_ok = R is not None and SYM_POOLED[0] <= R <= SYM_POOLED[1]
        sym = pooled_ok and not outside
        out[fam] = {'pooled_ratio': R, 'pooled_in_0.8_1.25': pooled_ok, 'cases_outside_0.5_2': outside,
                    'cases_ratio_undefined_0_over_0': undefined,
                    'verdict': 'NOT_A_BUG' if sym else 'FINDING'}
        if not sym:
            p = pooled[fam]
            out[fam]['size'] = {'excess_gpu_minus_cpu_only_cells_x_hours': p['excess_gpu_minus_cpu_only'],
                                'gpu_only': p['gpu_only'], 'cpu_only': p['cpu_only'], 'both': p['both'],
                                'mean_first_onset_shift_h_gpu_minus_cpu': p['first_onset']['mean_shift_h_gpu_minus_cpu'],
                                'by_class': p['by_class'], 'other_side_frac_of_thr': p['other_side_frac_of_thr']}
    return out


def nan_controls():
    """E200 self-test: one NaN/+Inf/-Inf per counted field at tau>=1 (and a finite rain value at tau 0) must be REJECTED."""
    rng = np.random.default_rng(0)
    base = {'rain_rate_mm_h': rng.random((73, 4, 5)).astype(np.float32), 'LWP': rng.random((73, 4, 5)).astype(np.float32)}
    base['rain_rate_mm_h'][0] = np.nan
    res = {}
    for key in base:
        for bad in (np.nan, np.inf, -np.inf):
            z = {k: v.copy() for k, v in base.items()}
            z[key][5, 1, 2] = bad
            try:
                load_field(z, key, 'control'); res[f'{key}={bad}@tau5'] = 'ACCEPTED(BAD)'
            except Reject:
                res[f'{key}={bad}@tau5'] = 'REJECTED'
    for name, bad in (('finite', 0.0), ('+inf', np.inf), ('-inf', -np.inf)):
        z = {k: v.copy() for k, v in base.items()}; z['rain_rate_mm_h'][0, 0, 0] = bad
        try:
            load_field(z, 'rain_rate_mm_h', 'control'); res[f'rain_rate {name}@tau0'] = 'ACCEPTED(BAD)'
        except Reject:
            res[f'rain_rate {name}@tau0'] = 'REJECTED'
    for k in base:  # healthy control must pass
        load_field(base, k, 'control')
    res['healthy'] = 'ACCEPTED'
    if any(v == 'ACCEPTED(BAD)' for v in res.values()):
        raise SystemExit(f'NaN controls failed: {res}')
    return res


def twin_static_check():
    """Twin extract vs reference: static fields + tau0 state must be bitwise (same wrfinput)."""
    out = {}
    for dn in DOMS:
        zr, zt = np.load(f'{T}/te_repro/20260227_cpu/{dn}'), np.load(f'{TWIN}/{dn}')
        out[dn] = {k: bool(np.array_equal(np.asarray(zr[k]), np.asarray(zt[k]), equal_nan=True))
                   for k in ('XLAT', 'XLONG', 'HGT', 'LANDMASK')}
        out[dn].update({f'{k}@tau0': bool(np.array_equal(np.asarray(zr[k])[0], np.asarray(zt[k])[0])) for k in ('LWP', 'RAINNC', 'T2')})
    return out


def main(outp):
    res = {'rule_source': '<DATA_ROOT>/alisios/state/manager/cross/FROM_WRF_GPU2_V043_AMEND1_ACK_20261021.md §3',
           'thresholds': {f: v[1] for f, v in FAMS.items()}, 'tau': [1, 72],
           'proxy_disclosure': 'LWP = f32(sum_k (QCLOUD+QRAIN)*dm) from the teacher extract: includes rain water, not cloud water alone',
           'nan_controls': nan_controls(), 'cases': {}, 'rejected': {}}
    for case, cdir, gdir in pairs():
        try:
            res['cases'][case] = run_pair(case, cdir, gdir)
        except Reject as e:
            res['rejected'][case] = str(e)
        print(case, 'done' if case in res['cases'] else 'REJECTED', flush=True)
    pooled = {f: {} for f in FAMS}
    for v in res['cases'].values():
        for f in FAMS:
            t = dict(v['total'][f]); fo = dict(t['first_onset'])
            add(pooled[f], {k: t[k] for k in ('both', 'cpu_only', 'gpu_only', 'neither', 'cells_x_hours') +
                            tuple(k for k in t if k.startswith(('cpu_only_hit', 'gpu_only_hit')))})
            add(pooled[f], {'by_class': t['by_class'], 'other_side_frac_of_thr': t['other_side_frac_of_thr'],
                            'first_onset': {k: fo[k] for k in ('cells_both', 'sum_shift_h', 'cells_cpu_only_ever', 'cells_gpu_only_ever')}})
    for f in FAMS:
        p = pooled[f]
        p['ratio_gpu_only_over_cpu_only'] = ratio(p['gpu_only'], p['cpu_only'])
        p['excess_gpu_minus_cpu_only'] = p['gpu_only'] - p['cpu_only']
        n = p['first_onset']['cells_both']
        p['first_onset']['mean_shift_h_gpu_minus_cpu'] = p['first_onset']['sum_shift_h'] / n if n else None
    res['pooled'] = pooled
    res['verdict'] = verdict(res['cases'], pooled) if not res['rejected'] else {'verdict': 'INCOMPLETE', 'rejected': res['rejected']}
    # noise reference: 0227 CPU twin (4x1) vs CPU reference (2x2); ERA5 0302 twin wrfouts deleted -> UNAVAILABLE
    twin = {'wn3_20260227': None, 'era5_20240302': 'UNAVAILABLE: CPU twin 2x2 wrfouts deleted by w43-teacher cleanup '
            '2026-10-21T21:18Z (MAILBOX); only per-lead aggregate stats remain in w43-teacher/score/C_era5_20240302/x, no cell-level fields'}
    try:
        twin['wn3_20260227'] = run_pair('twin_20260227', f'{T}/te_repro/20260227_cpu', TWIN)
        twin['wn3_20260227']['static_tau0_bitwise_vs_reference'] = twin_static_check()
        twin['wn3_20260227']['note'] = ('"gpu" side = CPU twin C2_4x1 (same inputs/executable, 4x1 decomposition) extracted with '
                                       'w43-teacher teacher_extract.py@cf54a11a42; "cpu" side = CPU reference extract')
    except Reject as e:
        twin['wn3_20260227'] = f'REJECTED: {e}'
    res['twin_noise_reference'] = twin
    json.dump(res, open(outp, 'w'), indent=1, allow_nan=True)
    print(json.dumps(res['verdict'], indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
