"""Generate a compact independent WRF SMS coefficient fixture; no full-scheme claim.

Reuses tracked real Swiss crop/terrain inputs, with required hpbl/dlk operands
specified explicitly. Grey-zone spacings exercise scale blends and slope branches.
Tolerances pre-registered before generation: 32 eps32 relative + 32 eps32*maxabs
per field; all candidate AND reference arrays must be finite, including halos.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from proofs.o1_smag3d import oracle_io

SOURCE = ROOT / 'data/fixtures/les3d-smag3d-oracle-v1.npz'
OUT_NAMES = ('xkmh', 'xkmv', 'xkhh', 'xkhv', 'xkmv_meso', 'l_diss')
INPUT3_NAMES = ('tke', 'theta', 'dthrdn', 'bn2', 'rdzw', 'd11', 'd22', 'd12', 'zx', 'zy', 'dlk')
INPUT2_NAMES = ('msftx', 'msfty', 'hpbl')
CONFIGS = {
    'grey_spec': dict(dx=600., periodic=False, isfflx=1),
    'grey_periodic': dict(dx=1200., periodic=True, isfflx=0),
    'les_seed': dict(dx=80., periodic=True, isfflx=0, zero_tke=True),
    'drag_seed_edge': dict(dx=400., periodic=False, isfflx=0, zero_tke=True,
                          tke_drag_coefficient=5.e-11),
    'meso_cap': dict(dx=3000., periodic=False, isfflx=2, high_tke=True),
}


def source_case(z, *, high_tke=False, zero_tke=False):
    prefix = 'in/wind20/' if high_tke else 'in/real/'
    case = {k[len(prefix):]: np.array(v) for k, v in z.items()
            if k.startswith(prefix) and not k[len(prefix):].startswith('moist')}
    case['moist'] = [np.array(z[prefix + f'moist{i}']) for i in range(5)]
    if zero_tke:
        case['tke'][:] = 0.
    if high_tke:
        case['tke'][:] = 1000.
    nz, ny, nx = case['tke'].shape
    # Independent, explicit physical-length operands: zeros exercise h=0;
    # different positive heights exercise scale-adaptive blends. Not a PBLH port.
    case['hpbl'] = np.broadcast_to(np.resize(np.array([0., 150., 500., 1200., 3000.], np.float32), nx), (ny, nx)).copy()
    case['dlk'] = np.broadcast_to(np.linspace(5., 600., nz, dtype=np.float32)[:, None, None], (nz, ny, nx)).copy()
    return case


def case_for(z, cfg):
    case = source_case(z, high_tke=cfg.get('high_tke', False), zero_tke=cfg.get('zero_tke', False))
    case['dx'] = case['dy'] = np.float32(cfg['dx'])
    case['rdx'] = case['rdy'] = np.float32(1.) / case['dx']
    return case


def input_from_fixture(z, key):
    from gpuwrf.dynamics.sms3dtke import SmsInputs
    arrays = {n: z[key + '/in/' + n] for n in INPUT3_NAMES + INPUT2_NAMES}
    arrays['dx'] = arrays['dy'] = np.float32(CONFIGS[key]['dx'])
    return SmsInputs(**arrays)


def config_from_case(cfg):
    from gpuwrf.dynamics.sms3dtke import SmsConfig
    return SmsConfig(boundary='periodic' if cfg['periodic'] else 'specified',
        isfflx=cfg['isfflx'], tke_drag_coefficient=cfg.get('tke_drag_coefficient', 0.))


def generate(exe: Path, output: Path, work: Path):
    saved = {}
    with np.load(SOURCE, allow_pickle=False) as src:
        for name, cfg in CONFIGS.items():
            case = case_for(src, cfg)
            fcfg = dict(km_opt=2, vertical=True, periodic=cfg['periodic'], isfflx=cfg['isfflx'],
                        tke_drag_coefficient=cfg.get('tke_drag_coefficient', 0.))
            inp, out = work / (name + '.bin'), work / (name + '.out')
            oracle_io.write_input(case, inp, fcfg)
            with inp.open('ab') as f:
                f.write(case['hpbl'].T.tobytes(order='F'))
                f.write(oracle_io._fo(case['dlk']))
            subprocess.run([str(exe), str(inp), str(out)], check=True, timeout=30)
            raw = out.read_bytes()
            ims, ime, jms, jme, kms, kme = np.frombuffer(raw[:24], np.int32)
            ni, nj, nk = ime - ims + 1, jme - jms + 1, kme - kms + 1
            size = ni * nk * nj
            flat = np.frombuffer(raw[24:], np.float32)
            for idx, field in enumerate(OUT_NAMES):
                values = flat[idx*size:(idx+1)*size].reshape((ni, nk, nj), order='F').transpose(1, 2, 0)
                # The WRF driver has 5 halo points, JAX uses 3.
                saved[name + '/' + field] = np.array(values[:, 2:-2, 2:-2])
                assert np.isfinite(saved[name + '/' + field]).all(), (name, field)
            for idx, field in enumerate(INPUT3_NAMES, start=len(OUT_NAMES)):
                values = flat[idx*size:(idx+1)*size].reshape((ni,nk,nj), order='F').transpose(1,2,0)
                saved[name + '/in/' + field] = np.array(values[:,2:-2,2:-2])
            offset = (len(OUT_NAMES) + len(INPUT3_NAMES)) * size
            size2 = ni * nj
            for idx, field in enumerate(INPUT2_NAMES):
                values = flat[offset+idx*size2:offset+(idx+1)*size2].reshape((ni,nj), order='F').T
                saved[name + '/in/' + field] = np.array(values[2:-2,2:-2])
            inp.unlink()
            out.unlink()
    meta = {'source_fixture_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'configs': CONFIGS, 'scope': 'SMS-specific coefficient branches after common WRF preprocessing; not full scheme/forecast',
            'hpbl_dlk': 'explicit real-structured operands (see generator); no fallback to defaults',
            'eps_multiplier': 32}
    saved['meta_json'] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    np.savez_compressed(output, **saved)
    print(output, output.stat().st_size, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    generate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
