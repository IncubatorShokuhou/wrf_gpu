"""Pristine-WRF MYJSFC oracle on real columns of the ORIGINAL CPU-WRF c03 run (lane t-cfix-sfc, no JAX).

c03 = Swiss 2023-01-15, bl_pbl_physics=2 / sf_sfclay_physics=2 / Noah-MP: full 44-level
profiles (U V T P PB PH PHB QVAPOR QCLOUD TKE_PBL) at 7 frames, picks spanning HFX/UST
extremes + random columns, WATER variants (xland=2, skin colder/warmer than the air) and
calm variants.  MYJSFC is called THREE times like WRF from a cold start: call 1 with
itimestep=1 and zero INOUT memory (THZ0 QZ0 UZ0 VZ0 AKHS AKMS, Registry init), calls 2
and 3 with itimestep 2/3 consuming the previous call's own INOUT outputs (USTAR ZNT
QSFC THZ0 QZ0 UZ0 VZ0 AKHS AKMS).  Builds zero (reference) / inf / fma.
usage: python make_oracle_myj.py <lane_dir>   -> <lane>/oracle/oracle_myj.npz
"""
from pathlib import Path
import ctypes, hashlib, json, sys
import numpy as np
import netCDF4

LANE = Path(sys.argv[1])
CPU = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c03/cpu')
FRAMES = [f'2023-01-15_{h:02d}:00:00' for h in (3, 6, 9, 12, 15, 18, 21)]
R_D, CP, G = 287.0, 1004.5, 9.81
PROF = 'dz pmid pint th t qv qc u v q2'.split()
SFC = 'ht tsk qsfc thz0 qz0 uz0 vz0 xland ust znt z0base mavail akhs akms'.split()
OUT = ('ust znt pblh rmol akhs akms rib chs chs2 cqs2 hfx qfx lh flhc flqc qgh cpm ct u10 v10 '
       't02 th02 tshltr th10 q02 qshltr q10 pshltr u10e v10e qsfc thz0 qz0 uz0 vz0 spare').split()
MEMORY = ('ust', 'znt', 'qsfc', 'thz0', 'qz0', 'uz0', 'vz0', 'akhs', 'akms')


def columns():
    rng = np.random.default_rng(20261015)
    profs, sfcs, labels = [], [], []
    for frame in FRAMES:
        with netCDF4.Dataset(CPU / f'wrfout_d01_{frame}') as d:
            v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                'U', 'V', 'T', 'P', 'PB', 'PH', 'PHB', 'QVAPOR', 'QCLOUD', 'TKE_PBL', 'PSFC', 'TSK', 'UST',
                'HFX', 'HGT')}
        nz, ny, nx = v['T'].shape
        pick = {'max_hfx': np.argmax(v['HFX']), 'min_hfx': np.argmin(v['HFX']),
                'max_ust': np.argmax(v['UST']), 'min_ust': np.argmin(v['UST'])}
        for r in range(4):
            pick[f'random{r}'] = int(rng.integers(0, ny * nx))
        for tag, flat in pick.items():
            j, i = np.unravel_index(int(flat), (ny, nx))
            p = (v['P'] + v['PB'])[:, j, i]
            zw = (v['PH'] + v['PHB'])[:, j, i] / G
            dz = np.diff(zw)
            th = v['T'][:, j, i] + 300.
            t = th * (p / 1.e5) ** (R_D / CP)
            u = .5 * (v['U'][:, j, i] + v['U'][:, j, i + 1]); vv = .5 * (v['V'][:, j, i] + v['V'][:, j + 1, i])
            pw = np.empty(nz + 1); pw[0] = v['PSFC'][j, i]
            zm = .5 * (zw[:-1] + zw[1:]); wgt = (zw[1:-1] - zm[:-1]) / (zm[1:] - zm[:-1])
            pw[1:-1] = p[:-1] + wgt * (p[1:] - p[:-1]); pw[-1] = p[-1] - .5 * (p[-2] - p[-1])
            prof = np.zeros((nz + 1, len(PROF)))
            for c, val in enumerate((dz, p, None, th, t, v['QVAPOR'][:, j, i], v['QCLOUD'][:, j, i], u, vv,
                                     v['TKE_PBL'][:nz, j, i])):
                if val is None:
                    prof[:, c] = pw
                else:
                    prof[:-1, c] = val; prof[-1, c] = val[-1]
            base = dict(ht=v['HGT'][j, i], tsk=v['TSK'][j, i], qsfc=0., thz0=0., qz0=0., uz0=0., vz0=0.,
                        xland=1., ust=v['UST'][j, i], znt=float(rng.choice([.03, .1, .25, .5])), z0base=0.,
                        mavail=float(rng.uniform(.3, 1.)), akhs=0., akms=0.)
            base['z0base'] = base['znt']
            variants = [('land', prof, base)]
            if tag.startswith('random') or tag in ('max_hfx', 'min_hfx'):
                for dt in (-4., -.5, 1.5, 6.):
                    variants.append((f'water{dt:+}', prof, dict(base, xland=2., tsk=t[0] + dt,
                                     znt=float(rng.choice([1.e-4, 4.e-4, 2.e-3])), mavail=1.)))
            if tag in ('random0', 'min_ust'):
                calm = prof.copy(); calm[:, 7] *= .03; calm[:, 8] *= .03
                variants.append(('calm', calm, base))
                variants.append(('calm_water', calm, dict(base, xland=2., tsk=t[0] + 1., znt=1.e-4, mavail=1.)))
            for surface, pr, s in variants:
                profs.append(pr); sfcs.append([s[k] for k in SFC])
                labels.append(dict(frame=frame, pick=tag, j=int(j), i=int(i), surface=surface))
    return np.array(profs, np.float32), np.array(sfcs, np.float32), labels


def load(variant):
    so = LANE / 'oracle' / 'sf2' / variant / 'oracle.so'
    fn = ctypes.CDLL(str(so)).probe
    a = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, a, a, a]
    return fn


def call(fn, prof, sfc, itimestep):
    n, nz1 = prof.shape[0], prof.shape[1]
    out = np.zeros((n, len(OUT)), np.float32, order='F')
    fn(n, nz1 - 1, itimestep, np.asfortranarray(prof), np.asfortranarray(sfc), out)
    return out


def chain(fn, prof, sfc, calls=3):
    outs, sfcs = [], [sfc]
    for c in range(1, calls + 1):
        out = call(fn, prof, sfcs[-1], c)
        outs.append(out)
        nxt = sfcs[-1].copy()
        for k in MEMORY:
            nxt[:, SFC.index(k)] = out[:, OUT.index(k)]
        sfcs.append(nxt)
    return outs, sfcs[:-1]


def main():
    prof, sfc, labels = columns()
    result = {'prof': prof}
    for variant in ('zero', 'inf', 'fma'):
        outs, sfcs = chain(load(variant), prof, sfc)
        for c, (o, s) in enumerate(zip(outs, sfcs), 1):
            result[f'{variant}_out{c}'] = o
            result[f'{variant}_sfc{c}'] = s
    same = all(np.array_equal(result[f'zero_out{c}'], result[f'inf_out{c}']) for c in (1, 2, 3))
    result['init_byte_identical'] = np.array(same)
    path = LANE / 'oracle' / 'oracle_myj.npz'
    np.savez_compressed(path, **result)
    meta = dict(prof=PROF, sfc=SFC, outputs=OUT, labels=labels,
                npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (LANE / 'oracle' / 'oracle_myj.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(prof.shape, 'zero==inf', same, 'finite', bool(np.isfinite(result['zero_out3']).all()), meta['npz_sha256'])


if __name__ == '__main__':
    main()
