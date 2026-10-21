#!/usr/bin/env python3
"""Reproduction of the ALISIOS "alisios.teacher_extract.v2" extract format from wrfout files.

Inputs are ONLY native wrfout files (d01/d02/d03, 73 hourly frames each) plus a POINTS.parquet
point set (reused from the ALISIOS reference extract of the same issue). No ALISIOS code is used:
the formulas were inferred empirically against ALISIOS's own CPU extract of 20260227_18z
(see analysis/teacher_extract notes / te_repro/COMPARE_20260227.json).

Usage:
  teacher_extract.py --wrfout-dir DIR --issue YYYYMMDD_18z --attempt N --points POINTS.parquet \
                     --out OUTDIR --producer "<text>"
  teacher_extract.py --readback OUTDIR

Provenance fields are our own (producer string, our code sha, our frame sha list). ALISIOS-only
fields (w20, cpu_lock, registry, guard, era tuple, canonical cache path) are null; the reason is in
MANIFEST.producer_note.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

SCHEMA = 'alisios.teacher_extract.v2'
DOMAINS = ('d01', 'd02', 'd03')
NPZ_NAME = {'d01': 'd01_crop_sfc.npz', 'd02': 'd02_crop_sfc.npz', 'd03': 'd03_sfc.npz'}
CROP = {'d01': [24, 39, 43, 60], 'd02': [33, 68, 89, 130], 'd03': [0, 93, 0, 111]}  # y0,y1,x0,x1
NFRAMES = 73
PROFILE_DOMAIN = 'd03'
STATIC = ('XLAT', 'XLONG', 'HGT', 'LANDMASK', 'COSALPHA', 'SINALPHA')
TIMEV = ('lead_h', 'tau_h', 'actual_time_unix_s', 'time_offset_seconds')
NATIVE = ('U10', 'V10', 'T2', 'Q2', 'PSFC', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'HFX', 'LH', 'PBLH')
DERIVED = ('u10_east', 'v10_north', 'wind_speed', 'rh2_unclipped', 'rain_total', 'rain_increment',
           'cloud_max', 'cloud_low_max', 'cloud_mid_max', 'cloud_high_max', 'rain_rate_mm_h',
           'rain_negative_flag', 'wind_direction_from_deg', 'cloud_low_sum', 'cloud_mid_sum',
           'cloud_high_sum', 'LWP', 'IWP', 'QCLOUD_negative_column_flag')
FIELDS = STATIC + TIMEV + NATIVE + DERIVED
UINT8_FIELDS = ('rain_negative_flag', 'QCLOUD_negative_column_flag')
G = 9.81
CALM_WIND_MS = 2.0           # wind_direction is NaN where wind_speed < 2 m/s
CLOUD_LOW_TOP_M = 2000.0     # band by mass-level height above ground (z_agl, f32)
CLOUD_HIGH_BASE_M = 6000.0
SURFACE_UNITS = {'GLW': 'W m-2', 'HFX': 'W m-2', 'LH': 'W m-2', 'PBLH': 'm', 'PSFC': 'Pa', 'Q2': 'kg kg-1',
                 'RAINC': 'mm', 'RAINNC': 'mm', 'SWDOWN': 'W m-2', 'T2': 'K',
                 'U10': 'm s-1 grid relative', 'V10': 'm s-1 grid relative'}
RH_FORMULA = '100*(Q2*PSFC/(0.622+Q2))/(611.2*exp(17.67*(T2-273.15)/(T2-29.65)))'
TPROF_FORMULA = '(T+300)*((P+PB)/100000)^(2/7)'
MASS_FORMULA = '-(C1H*(MU+MUB)+C2H)*DNW/9.81; hybrid native dry mass'
PROFILE_COLUMNS = ('phys_id', 'role', 'issue', 'attempt', 'valid_utc', 'actual_utc', 'lead_h', 'tau_h',
                   'time_offset_seconds', 'level', 'grid_y', 'grid_x', 'T_K', 'theta_K', 'pressure_Pa',
                   'QV_kg_kg', 'U_ms', 'V_ms', 'QCLOUD_kg_kg', 'CLDFRA', 'z_msl_m', 'z_agl_m')
# Derived-field definitions as inferred (documented in MANIFEST.derived_definitions).
DERIVED_DEFS = {
    'u10_east': 'f32(U10*COSALPHA - V10*SINALPHA) computed in f64',
    'v10_north': 'f32(V10*COSALPHA + U10*SINALPHA) computed in f64',
    'wind_speed': 'f32 hypot(U10, V10) (grid relative == earth relative magnitude)',
    'rh2_unclipped': 'f32(rh_formula in f64), no clipping',
    'rain_total': 'RAINC + RAINNC (f32)',
    'rain_increment': 'f32(T[t] - T[t-1]) with T = f64(RAINC) + f64(RAINNC); NaN at tau 0',
    'rain_rate_mm_h': 'f32((T[t]-T[t-1])*3600/(actual_time_unix_s[t]-actual_time_unix_s[t-1])) in f64; NaN at tau 0',
    'rain_negative_flag': 'uint8(T[t]-T[t-1] < 0) in f64; 0 at tau 0',
    'wind_direction_from_deg': 'f32 mod(degrees(arctan2(-u10_east, -v10_north)), 360); NaN where wind_speed < 2 m/s',
    'cloud_max': 'max over levels of CLDFRA',
    'cloud_*_max/_sum': 'max / f32(f64 sum) of CLDFRA over levels with z_agl < 2000 m (low), 2000 <= z_agl < 6000 (mid), >= 6000 (high); '
                        'z_agl = f32(0.5*(zw[k]+zw[k+1])) - HGT, zw = (PH+PHB)/9.81 in f64',
    'LWP': 'f32(sum_k f64(QCLOUD+QRAIN)*dm), dm = column_mass_formula in f64 [kg m-2]',
    'IWP': 'f32(sum_k f64(QICE+QSNOW+QGRAUP)*dm) [kg m-2]',
    'QCLOUD_negative_column_flag': 'uint8(any_k QCLOUD < 0)',
    'profiles': 'd03 columns at POINTS grid_y/grid_x (profile_status AVAILABLE), all 44 mass levels; '
                'U/V destaggered (mean of faces, f64) and rotated to earth-relative; theta=T+300; '
                'pressure=P+PB; T_K=theta*(p/1e5)^(2/7) in f64; z_msl=f32(0.5*(zw[k]+zw[k+1])); '
                'z_agl=z_msl(f32)-HGT; one parquet row group per frame, rows point-major then level',
}


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, doc):
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write('\n')


# ----------------------------------------------------------------------------------------- frames
_NAME = re.compile(r'^wrfout_(d0[1-3])_(\d{4}-\d{2}-\d{2})_(\d{2}):(\d{2}):(\d{2})$')


def discover(wrfout_dir):
    frames = {d: [] for d in DOMAINS}
    for p in sorted(glob.glob(os.path.join(wrfout_dir, 'wrfout_d0*'))):
        m = _NAME.match(os.path.basename(p))
        if not m or m.group(1) not in frames:
            continue
        t = dt.datetime.strptime(m.group(2) + ' ' + ':'.join(m.group(3, 4, 5)), '%Y-%m-%d %H:%M:%S').replace(
            tzinfo=dt.timezone.utc)
        frames[m.group(1)].append((t, p))
    for d in DOMAINS:
        frames[d].sort()
        if len(frames[d]) != NFRAMES:
            raise SystemExit(f'FRAME_COUNT {d}: {len(frames[d])} != {NFRAMES} in {wrfout_dir}')
        prev = None
        for t, p in frames[d]:
            nominal = t.replace(minute=0, second=0)
            if t - nominal > dt.timedelta(minutes=5):
                raise SystemExit(f'FRAME_TIME_NOT_NEAR_HOUR {p}')
            if prev is not None and nominal - prev != dt.timedelta(hours=1):
                raise SystemExit(f'FRAME_NOT_HOURLY {p}')
            prev = nominal
    t0 = {d: frames[d][0][0].replace(minute=0, second=0) for d in DOMAINS}
    if len(set(t0.values())) != 1:
        raise SystemExit(f'DOMAIN_START_MISMATCH {t0}')
    return frames


def times_for(frames_d, issue_dt):
    actual = [t for t, _ in frames_d]
    nominal = [t.replace(minute=0, second=0) for t in actual]
    tau = np.arange(NFRAMES, dtype=np.int16)
    lead0 = int(round((nominal[0] - issue_dt).total_seconds() / 3600.0))
    lead = (tau.astype(np.int64) + lead0).astype(np.int16)
    unix = np.array([int(t.timestamp()) for t in actual], dtype=np.int64)
    off = np.array([(a - n).total_seconds() for a, n in zip(actual, nominal)], dtype=np.float32)
    return {'lead_h': lead, 'tau_h': tau, 'actual_time_unix_s': unix, 'time_offset_seconds': off,
            'nominal': nominal, 'actual': actual}


def check_times_var(ds, t_expected, path):
    raw = ds.variables['Times'][0]
    s = b''.join(raw.tolist()).decode() if hasattr(raw, 'tolist') else str(raw)
    s = s.replace('\x00', '').strip()
    if s != t_expected.strftime('%Y-%m-%d_%H:%M:%S'):
        raise SystemExit(f'TIMES_VAR_MISMATCH {path}: {s!r}')


# --------------------------------------------------------------------------------------- physics
def column_mass(ds, sl):
    c1h = ds.variables['C1H'][0].astype(np.float64)[:, None, None]
    c2h = ds.variables['C2H'][0].astype(np.float64)[:, None, None]
    dnw = ds.variables['DNW'][0].astype(np.float64)[:, None, None]
    mu = ds.variables['MU'][0][sl].astype(np.float64) + ds.variables['MUB'][0][sl]
    return -(c1h * mu + c2h) * dnw / G


def z_heights(ds, sl, hgt):
    ph = ds.variables['PH'][0][(slice(None),) + sl].astype(np.float64) + ds.variables['PHB'][0][(slice(None),) + sl]
    zw = ph / G
    zmsl = (0.5 * (zw[1:] + zw[:-1])).astype(np.float32)
    zagl = zmsl - hgt
    return zmsl, zagl


def surface_frame(ds, sl):
    """Native + 3-D-derived per-frame quantities for a crop (2-D outputs)."""
    v = lambda n: ds.variables[n][0][sl]
    v3 = lambda n: ds.variables[n][0][(slice(None),) + sl]
    out = {n: v(n) for n in NATIVE}
    hgt = v('HGT')
    cf = v3('CLDFRA')
    _, zagl = z_heights(ds, sl, hgt)
    low = zagl < np.float32(CLOUD_LOW_TOP_M)
    high = zagl >= np.float32(CLOUD_HIGH_BASE_M)
    mid = ~low & ~high
    zero = np.float32(0)
    out['cloud_max'] = cf.max(axis=0)
    for name, m in (('low', low), ('mid', mid), ('high', high)):
        sel = np.where(m, cf, zero)
        out[f'cloud_{name}_max'] = sel.max(axis=0)
        out[f'cloud_{name}_sum'] = sel.astype(np.float64).sum(axis=0).astype(np.float32)
    dm = column_mass(ds, sl)
    qc = v3('QCLOUD')
    liq = qc.astype(np.float64) + v3('QRAIN')
    ice = v3('QICE').astype(np.float64) + v3('QSNOW') + v3('QGRAUP')
    out['LWP'] = (liq * dm).sum(axis=0).astype(np.float32)
    out['IWP'] = (ice * dm).sum(axis=0).astype(np.float32)
    out['QCLOUD_negative_column_flag'] = (qc < 0).any(axis=0).astype(np.uint8)
    return out


def finish_surface(stack, static, tv):
    """Derived fields that only need the stacked 2-D natives."""
    u = stack['U10'].astype(np.float64)
    vv = stack['V10'].astype(np.float64)
    ca = static['COSALPHA'].astype(np.float64)
    sa = static['SINALPHA'].astype(np.float64)
    ue = (u * ca - vv * sa).astype(np.float32)
    vn = (vv * ca + u * sa).astype(np.float32)
    ws = np.hypot(stack['U10'], stack['V10'])
    t2 = stack['T2'].astype(np.float64)
    q2 = stack['Q2'].astype(np.float64)
    ps = stack['PSFC'].astype(np.float64)
    rh = (100 * (q2 * ps / (0.622 + q2)) / (611.2 * np.exp(17.67 * (t2 - 273.15) / (t2 - 29.65)))).astype(np.float32)
    total = stack['RAINC'] + stack['RAINNC']
    total64 = stack['RAINC'].astype(np.float64) + stack['RAINNC']
    inc64 = total64[1:] - total64[:-1]
    inc = np.full_like(total, np.nan)
    inc[1:] = inc64.astype(np.float32)
    dts = np.diff(tv['actual_time_unix_s']).astype(np.float64)
    rate = np.full_like(total, np.nan)
    rate[1:] = (inc64 * 3600.0 / dts[:, None, None]).astype(np.float32)
    neg = np.zeros(total.shape, dtype=np.uint8)
    neg[1:] = (inc64 < 0).astype(np.uint8)
    wd = np.mod(np.degrees(np.arctan2(-ue, -vn)), np.float32(360)).astype(np.float32)
    wd = np.where(ws < np.float32(CALM_WIND_MS), np.float32(np.nan), wd).astype(np.float32)
    return {'u10_east': ue, 'v10_north': vn, 'wind_speed': ws.astype(np.float32), 'rh2_unclipped': rh,
            'rain_total': total, 'rain_increment': inc, 'rain_rate_mm_h': rate, 'rain_negative_flag': neg,
            'wind_direction_from_deg': wd}


def profile_frame(ds, yy, xx):
    """Returns dict of (npts, nlev) arrays for the profile columns."""
    v3 = lambda n: ds.variables[n][0]
    t = v3('T')[:, yy, xx].astype(np.float64)
    p = v3('P')[:, yy, xx].astype(np.float64) + v3('PB')[:, yy, xx]
    theta = t + 300.0
    tk = theta * (p / 100000.0) ** (2.0 / 7.0)
    U = v3('U')
    V = v3('V')
    um = 0.5 * (U[:, yy, xx].astype(np.float64) + U[:, yy, xx + 1])
    vm = 0.5 * (V[:, yy, xx].astype(np.float64) + V[:, yy + 1, xx])
    ca = ds.variables['COSALPHA'][0][yy, xx].astype(np.float64)
    sa = ds.variables['SINALPHA'][0][yy, xx].astype(np.float64)
    ph = v3('PH')[:, yy, xx].astype(np.float64) + v3('PHB')[:, yy, xx]
    zw = ph / G
    zmsl = (0.5 * (zw[1:] + zw[:-1])).astype(np.float32)
    zagl = zmsl - ds.variables['HGT'][0][yy, xx]
    cols = {'T_K': tk, 'theta_K': theta, 'pressure_Pa': p, 'QV_kg_kg': v3('QVAPOR')[:, yy, xx],
            'U_ms': um * ca - vm * sa, 'V_ms': vm * ca + um * sa, 'QCLOUD_kg_kg': v3('QCLOUD')[:, yy, xx],
            'CLDFRA': v3('CLDFRA')[:, yy, xx], 'z_msl_m': zmsl, 'z_agl_m': zagl}
    return {k: np.asarray(a).astype(np.float32).T for k, a in cols.items()}  # (npts, nlev)


# ------------------------------------------------------------------------------------------ main
def extract(a):
    import netCDF4
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    t_start = time.time()
    cpu0 = time.process_time()
    wrfout_dir = os.path.abspath(a.wrfout_dir)
    out = os.path.abspath(a.out)
    m = re.match(r'^(\d{8})_(\d{2})z$', a.issue)
    if not m:
        raise SystemExit('ISSUE_FORMAT expected YYYYMMDD_HHz')
    issue_dt = dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H').replace(tzinfo=dt.timezone.utc)
    os.makedirs(out, exist_ok=False)
    frames = discover(wrfout_dir)
    code_sha, _ = sha256_file(os.path.abspath(__file__))

    # --- source frames (hash everything we read)
    sources = {}
    raw_bytes = {d: 0 for d in DOMAINS}
    for d in DOMAINS:
        for _, p in frames[d]:
            s, n = sha256_file(p)
            sources[os.path.basename(p)] = {'bytes': n, 'path': p, 'sha256': s}
            raw_bytes[d] += n

    # --- surface npz per domain
    metrics_domains = {}
    tv_by_dom = {}
    for d in DOMAINS:
        y0, y1, x0, x1 = CROP[d]
        sl = (slice(y0, y1), slice(x0, x1))
        tv = times_for(frames[d], issue_dt)
        tv_by_dom[d] = tv
        per = {k: [] for k in NATIVE + ('cloud_max', 'cloud_low_max', 'cloud_mid_max', 'cloud_high_max',
                                         'cloud_low_sum', 'cloud_mid_sum', 'cloud_high_sum', 'LWP', 'IWP',
                                         'QCLOUD_negative_column_flag')}
        static = None
        for i, (t, p) in enumerate(frames[d]):
            with netCDF4.Dataset(p) as ds:
                ds.set_auto_mask(False)
                check_times_var(ds, t, p)
                if static is None:
                    static = {n: np.asarray(ds.variables[n][0][sl], dtype=np.float32) for n in STATIC}
                fr = surface_frame(ds, sl)
            for k in per:
                per[k].append(fr[k])
        stack = {k: np.stack(vs).astype(np.uint8 if k in UINT8_FIELDS else np.float32) for k, vs in per.items()}
        stack.update(finish_surface(stack, static, tv))
        arrays = {}
        for k in FIELDS:
            if k in STATIC:
                arrays[k] = static[k]
            elif k in TIMEV:
                arrays[k] = tv[k]
            else:
                arrays[k] = stack[k]
        path = os.path.join(out, NPZ_NAME[d])
        np.savez_compressed(path, **arrays)
        metrics_domains[d] = {
            'QCLOUD_negative_cell_frames': int(stack['QCLOUD_negative_column_flag'].astype(np.int64).sum()),
            'bytes': os.path.getsize(path), 'crop_y0_y1_x0_x1': list(CROP[d]), 'fields': list(FIELDS),
            'raw_bytes_hashed': raw_bytes[d], 'shape': [y1 - y0, x1 - x0]}

    # --- points + profiles (d03)
    points_out = os.path.join(out, 'POINTS.parquet')
    shutil.copyfile(a.points, points_out)
    pts = pd.read_parquet(a.points)
    av = pts[pts['profile_status'] == 'AVAILABLE']
    yy = av['grid_y'].to_numpy(np.int64)
    xx = av['grid_x'].to_numpy(np.int64)
    npts = len(av)
    schema = pa.schema([('phys_id', pa.string()), ('role', pa.string()), ('issue', pa.string()),
                        ('attempt', pa.int32()), ('valid_utc', pa.string()), ('actual_utc', pa.string()),
                        ('lead_h', pa.int16()), ('tau_h', pa.int16()), ('time_offset_seconds', pa.float32()),
                        ('level', pa.int16()), ('grid_y', pa.int32()), ('grid_x', pa.int32())] +
                       [(c, pa.float32()) for c in PROFILE_COLUMNS[12:]])
    prof_path = os.path.join(out, 'points_profile.parquet')
    tv = tv_by_dom[PROFILE_DOMAIN]
    nlev = None
    rows = 0
    with pq.ParquetWriter(prof_path, schema, compression='zstd') as w:
        for i, (t, p) in enumerate(frames[PROFILE_DOMAIN]):
            with netCDF4.Dataset(p) as ds:
                ds.set_auto_mask(False)
                pr = profile_frame(ds, yy, xx)
            nlev = pr['T_K'].shape[1]
            n = npts * nlev
            cols = {
                'phys_id': np.repeat(av['phys_id'].to_numpy(object), nlev),
                'role': np.repeat(av['role'].to_numpy(object), nlev),
                'issue': [a.issue] * n,
                'attempt': np.full(n, a.attempt, np.int32),
                'valid_utc': [tv['nominal'][i].isoformat()] * n,
                'actual_utc': [tv['actual'][i].isoformat()] * n,
                'lead_h': np.full(n, tv['lead_h'][i], np.int16),
                'tau_h': np.full(n, tv['tau_h'][i], np.int16),
                'time_offset_seconds': np.full(n, tv['time_offset_seconds'][i], np.float32),
                'level': np.tile(np.arange(nlev, dtype=np.int16), npts),
                'grid_y': np.repeat(yy.astype(np.int32), nlev),
                'grid_x': np.repeat(xx.astype(np.int32), nlev),
            }
            for c in PROFILE_COLUMNS[12:]:
                cols[c] = pr[c].ravel()
            w.write_table(pa.table({c: cols[c] for c in PROFILE_COLUMNS}, schema=schema))
            rows += n

    # --- JSON
    created = utcnow()
    nullnote = ('ALISIOS-only provenance (w20 CPU budget/cpu_lock, case registry, train-fit guard, era tuple, '
                'canonical cache path, label admission, format/physics shas) is not reproduced and set to null: '
                'this extract was produced by ' + a.producer + ' with analysis/teacher_extract.py, not by ALISIOS.')
    inputs = [{'path': os.path.abspath(a.points), 'sha256': sha256_file(a.points)[0]}]
    src_doc = {'created_utc': created, 'era_scope': None, 'frames': sources, 'guard': None, 'inputs': inputs,
               'producer_code_sha256': code_sha, 'producer_sprint': a.producer, 'registry_metadata': None,
               'schema': 'alisios.teacher_extract.sources.v2', 'status': 'SEALED'}
    write_json(os.path.join(out, 'SOURCES.json'), src_doc)
    wall = time.time() - t_start
    met = {'cold_cache_proven': False, 'created_utc': utcnow(), 'domains': metrics_domains, 'era_scope': None,
           'inputs': [], 'network_bytes_measured': False, 'points_available': int(npts),
           'points_requested': int(len(pts)), 'producer_code_sha256': code_sha, 'producer_sprint': a.producer,
           'profile_rows': int(rows), 'schema': 'alisios.teacher_extract.metrics.v2', 'status': 'SEALED',
           'w20': None}
    write_json(os.path.join(out, 'METRICS.json'), met)
    arts = {}
    for name in sorted(os.listdir(out)):
        if name == 'MANIFEST.json':
            continue
        s, n = sha256_file(os.path.join(out, name))
        arts[name] = {'bytes': n, 'sha256': s}
    cfg = {'crop': CROP, 'calm_wind_ms': CALM_WIND_MS, 'cloud_low_top_m': CLOUD_LOW_TOP_M,
           'cloud_high_base_m': CLOUD_HIGH_BASE_M, 'g': G, 'profile_domain': PROFILE_DOMAIN, 'fields': list(FIELDS),
           'profile_columns': list(PROFILE_COLUMNS), 'derived': DERIVED_DEFS}
    man = {'artifacts': arts, 'attempt': a.attempt, 'canonical_cache_path': None,
           'column_mass_formula': MASS_FORMULA, 'created_utc': utcnow(), 'crop': CROP, 'era_scope': None,
           'extract_config_sha': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
           'format_sha256': None, 'inputs': inputs, 'issue': a.issue, 'label_admission': None,
           'physics_sha': None, 'point_set_sha256': arts['POINTS.parquet']['sha256'],
           'producer_code_sha256': code_sha, 'producer_sprint': a.producer,
           'profile_temperature_formula': TPROF_FORMULA, 'rh_formula': RH_FORMULA, 'schema': SCHEMA,
           'status': 'SEALED', 'storage_scope': 'W43_LOCAL_REPRODUCTION', 'surface_units': SURFACE_UNITS,
           'w20': None,
           'producer_note': nullnote + ' point_set_sha256 = sha256 of the POINTS.parquet bytes (ALISIOS basis '
                            'unknown). extract_wall_seconds=%.1f cpu_seconds=%.1f.' % (wall, time.process_time() - cpu0),
           'derived_definitions': cfg}
    write_json(os.path.join(out, 'MANIFEST.json'), man)
    print(json.dumps({'out': out, 'wall_s': round(time.time() - t_start, 1), 'profile_rows': rows}))
    return 0


def readback(outdir):
    import pyarrow.parquet as pq
    outdir = os.path.abspath(outdir)
    rep = {'outdir': outdir, 'utc': utcnow(), 'failures': [], 'artifacts': {}, 'domains': {},
           'rule': 'non-finite allowed only: rain_increment/rain_rate_mm_h at frame 0; wind_direction_from_deg where '
                   'wind_speed < 2 m/s same cell/frame; any other NaN/Inf = FAIL (ALISIOS confirm 2026-10-21)'}
    fail = rep['failures']
    try:
        man = json.load(open(os.path.join(outdir, 'MANIFEST.json')))
    except Exception as e:  # noqa
        fail.append(f'MANIFEST_UNREADABLE {e}')
        man = {'artifacts': {}}
    expected = set(NPZ_NAME.values()) | {'POINTS.parquet', 'points_profile.parquet', 'SOURCES.json', 'METRICS.json'}
    if set(man.get('artifacts', {})) != expected:
        fail.append(f'ARTIFACT_SET {sorted(man.get("artifacts", {}))}')
    for name, meta in man.get('artifacts', {}).items():
        p = os.path.join(outdir, name)
        if not os.path.exists(p):
            fail.append(f'MISSING {name}')
            continue
        s, n = sha256_file(p)
        ok = (s == meta.get('sha256') and n == meta.get('bytes'))
        rep['artifacts'][name] = {'sha256_ok': ok, 'bytes': n}
        if not ok:
            fail.append(f'SHA_MISMATCH {name}')
    for d, name in NPZ_NAME.items():
        p = os.path.join(outdir, name)
        if not os.path.exists(p):
            continue
        dd = {'nonfinite': {}, 'expected_nan': {}}
        try:
            z = np.load(p)
            keys = list(z.files)
            if keys != list(FIELDS):
                fail.append(f'NPZ_KEYS {d}')
            nt = int(z['tau_h'].shape[0]) if 'tau_h' in keys else -1
            dd['frames'] = nt
            if nt != NFRAMES:
                fail.append(f'FRAME_COUNT {d} {nt}')
            ws = z['wind_speed'] if 'wind_speed' in keys else None
            for k in keys:
                arr = z[k]
                if k in UINT8_FIELDS or arr.dtype.kind not in 'f':
                    continue
                if arr.ndim == 3 and arr.shape[0] != NFRAMES:
                    fail.append(f'FIELD_FRAMES {d}/{k} {arr.shape}')
                bad = ~np.isfinite(arr)
                allowed = np.zeros_like(bad)
                # designed NaNs of the format: tau 0 increments/rates, calm-wind directions
                if k in ('rain_increment', 'rain_rate_mm_h') and arr.ndim == 3:
                    allowed[0] = np.isnan(arr[0])
                if k == 'wind_direction_from_deg' and ws is not None and ws.shape == arr.shape:
                    allowed = np.isnan(arr) & (ws < np.float32(CALM_WIND_MS))
                if allowed.any():
                    dd['expected_nan'][k] = int(allowed.sum())
                unexpected = int((bad & ~allowed).sum())
                if unexpected:
                    dd['nonfinite'][k] = unexpected
                    fail.append(f'NONFINITE {d}/{k} {unexpected}')
        except Exception as e:  # noqa
            fail.append(f'NPZ_UNREADABLE {d} {e}')
        # ALISIOS readback rule (cross TO_WRF_GPU2_EXTRACT_FORMAT_CONFIRM_20261021.md): count both kinds per domain
        dd['allowed_nan_total'] = int(sum(dd['expected_nan'].values()))
        dd['foreign_nonfinite_total'] = int(sum(dd['nonfinite'].values()))
        rep['domains'][d] = dd
    pp = os.path.join(outdir, 'points_profile.parquet')
    if os.path.exists(pp):
        try:
            pf = pq.ParquetFile(pp)
            cols = [f.name for f in pf.schema_arrow]
            if cols != list(PROFILE_COLUMNS):
                fail.append('PROFILE_COLUMNS')
            nonfin = {}
            taus = set()
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i)
                taus.update(t.column('tau_h').to_numpy().tolist())
                for c in PROFILE_COLUMNS[12:]:
                    a = t.column(c).to_numpy()
                    n = int((~np.isfinite(a)).sum())
                    if n:
                        nonfin[c] = nonfin.get(c, 0) + n
            rep['profile'] = {'rows': pf.metadata.num_rows, 'row_groups': pf.num_row_groups,
                              'distinct_tau': len(taus), 'nonfinite': nonfin}
            if len(taus) != NFRAMES:
                fail.append(f'PROFILE_FRAMES {len(taus)}')
            if nonfin:
                fail.append(f'PROFILE_NONFINITE {nonfin}')
        except Exception as e:  # noqa
            fail.append(f'PROFILE_UNREADABLE {e}')
    rep['status'] = 'PASS' if not fail else 'FAIL'
    write_json(os.path.join(outdir, 'READBACK.json'), rep)
    print(json.dumps({'status': rep['status'], 'failures': fail, 'domains': rep['domains'],
                      'profile': rep.get('profile')}, indent=1))
    return 0 if not fail else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--readback')
    p.add_argument('--wrfout-dir')
    p.add_argument('--issue')
    p.add_argument('--attempt', type=int)
    p.add_argument('--points')
    p.add_argument('--out')
    p.add_argument('--producer')
    a = p.parse_args(argv)
    if a.readback:
        return readback(a.readback)
    for k in ('wrfout_dir', 'issue', 'attempt', 'points', 'out', 'producer'):
        if getattr(a, k) in (None, ''):
            p.error(f'--{k.replace("_", "-")} is required')
    return extract(a)


if __name__ == '__main__':
    sys.exit(main())
