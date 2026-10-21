"""Readback-mode tests for analysis/teacher_extract.py on tiny synthetic extracts (no wrfout needed)."""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_extract as te  # noqa: E402


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _make(out, nframes=te.NFRAMES, nan_field=None):
    os.makedirs(out)
    ny, nx = 2, 3
    for d, name in te.NPZ_NAME.items():
        arrays = {}
        ws = np.full((nframes, ny, nx), 5.0, np.float32)
        for k in te.FIELDS:
            if k in te.STATIC:
                arrays[k] = np.ones((ny, nx), np.float32)
            elif k in ('lead_h', 'tau_h'):
                arrays[k] = np.arange(nframes, dtype=np.int16)
            elif k == 'actual_time_unix_s':
                arrays[k] = np.arange(nframes, dtype=np.int64) * 3600
            elif k == 'time_offset_seconds':
                arrays[k] = np.zeros(nframes, np.float32)
            elif k in te.UINT8_FIELDS:
                arrays[k] = np.zeros((nframes, ny, nx), np.uint8)
            elif k == 'wind_speed':
                arrays[k] = ws
            else:
                a = np.zeros((nframes, ny, nx), np.float32)
                if k in ('rain_increment', 'rain_rate_mm_h'):
                    a[0] = np.nan  # designed NaN of the format
                arrays[k] = a
        if nan_field and d == 'd02':
            arrays[nan_field][5, 1, 1] = np.nan
        np.savez_compressed(os.path.join(out, name), **arrays)
    pq.write_table(pa.table({'phys_id': ['a'], 'profile_status': ['AVAILABLE']}), os.path.join(out, 'POINTS.parquet'))
    n = nframes * 2
    cols = {c: (['x'] * n if c in ('phys_id', 'role', 'issue', 'valid_utc', 'actual_utc') else
                np.repeat(np.arange(nframes), 2).astype(np.int16) if c == 'tau_h' else
                np.zeros(n, np.float32)) for c in te.PROFILE_COLUMNS}
    pq.write_table(pa.table(cols), os.path.join(out, 'points_profile.parquet'))
    for j in ('SOURCES.json', 'METRICS.json'):
        json.dump({'schema': j}, open(os.path.join(out, j), 'w'))
    arts = {f: {'bytes': os.path.getsize(os.path.join(out, f)), 'sha256': _sha(os.path.join(out, f))}
            for f in sorted(os.listdir(out))}
    json.dump({'artifacts': arts}, open(os.path.join(out, 'MANIFEST.json'), 'w'))
    return out


def test_clean_passes(tmp_path):
    out = _make(str(tmp_path / 'ok'))
    assert te.main(['--readback', out]) == 0
    rep = json.load(open(os.path.join(out, 'READBACK.json')))
    assert rep['status'] == 'PASS'
    assert rep['domains']['d01']['expected_nan']['rain_increment'] == 6
    assert all(d['foreign_nonfinite_total'] == 0 and d['allowed_nan_total'] > 0 for d in rep['domains'].values())


def test_corrupt_byte_fails(tmp_path):
    out = _make(str(tmp_path / 'bad'))
    p = os.path.join(out, 'd03_sfc.npz')
    b = bytearray(open(p, 'rb').read())
    b[len(b) // 2] ^= 0xFF
    open(p, 'wb').write(bytes(b))
    assert te.main(['--readback', out]) == 1
    rep = json.load(open(os.path.join(out, 'READBACK.json')))
    assert any(f.startswith('SHA_MISMATCH d03_sfc.npz') for f in rep['failures'])


def test_nan_in_field_fails(tmp_path):
    out = _make(str(tmp_path / 'nan'), nan_field='T2')
    assert te.main(['--readback', out]) == 1
    rep = json.load(open(os.path.join(out, 'READBACK.json')))
    assert rep['domains']['d02']['nonfinite'] == {'T2': 1}
    assert rep['domains']['d02']['foreign_nonfinite_total'] == 1 and rep['domains']['d01']['foreign_nonfinite_total'] == 0


def test_72_frames_fails(tmp_path):
    out = _make(str(tmp_path / 'f72'), nframes=72)
    assert te.main(['--readback', out]) == 1
    rep = json.load(open(os.path.join(out, 'READBACK.json')))
    assert 'FRAME_COUNT d01 72' in rep['failures']
    assert any(f.startswith('PROFILE_FRAMES') for f in rep['failures'])


def test_calm_wind_nan_allowed_only_when_calm(tmp_path):
    out = _make(str(tmp_path / 'wd'))
    p = os.path.join(out, 'd01_crop_sfc.npz')
    z = dict(np.load(p))
    z['wind_direction_from_deg'][3, 0, 0] = np.nan  # wind_speed there is 5 m/s -> not calm
    np.savez_compressed(p, **z)
    man = json.load(open(os.path.join(out, 'MANIFEST.json')))
    man['artifacts']['d01_crop_sfc.npz'] = {'bytes': os.path.getsize(p), 'sha256': _sha(p)}
    json.dump(man, open(os.path.join(out, 'MANIFEST.json'), 'w'))
    assert te.main(['--readback', out]) == 1
