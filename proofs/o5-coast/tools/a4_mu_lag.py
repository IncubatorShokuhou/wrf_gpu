"""A4: is the d01 GPU−CPU dry-mass (MU) offset a boundary-timing lag? (o5-coast)

usage: python a4_mu_lag.py GPU_DIR CPU_DIR DOM NFRAMES OUT.json
Per output frame t and per distance-to-boundary ring k (0..9, interior ≥10): mean(MU_gpu − MU_cpu) and the CPU mass
tendency dMU/dt (central difference of CPU frames, Pa/s). Fits diff ≈ τ_lag · dMU/dt per ring (least squares over
frames 1..N-2); τ_lag in seconds (positive = GPU ahead in time). Also PSFC. No JAX."""
import glob
import json
import sys

import netCDF4 as nc
import numpy as np


def frames(d, dom, n):
    files = sorted(glob.glob(f"{d}/wrfout_{dom}_*"))
    return files[:n] if n else files


def secs(p):
    with nc.Dataset(p) as f:
        s = f['Times'][0].tobytes().decode()
    import datetime as dt
    return dt.datetime.strptime(s, '%Y-%m-%d_%H:%M:%S').timestamp()


def main():
    gd, cd, dom, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    G = {p[-19:]: p for p in frames(gd, dom, None)}
    Call = frames(cd, dom, None)
    # pair by timestamp; CPU tendency from the full CPU series (central difference), so a GPU run without the
    # t0 frame (emit_initial_history off) still pairs with the CPU frames at the same valid times.
    C = [p for p in Call if p[-19:] in G][:n]
    idx = [Call.index(p) for p in C]
    G = [G[p[-19:]] for p in C]
    assert len(C) >= 2, 'need >= 2 common frames'
    var = {}
    tsall = np.array([secs(p) for p in Call])
    for v in ('MU', 'PSFC'):
        call = np.stack([np.asarray(nc.Dataset(p)[v][0], np.float64) for p in Call])
        tend_all = np.full_like(call, np.nan)
        tend_all[1:-1] = (call[2:] - call[:-2]) / (tsall[2:] - tsall[:-2])[:, None, None]
        var[v] = (np.stack([np.asarray(nc.Dataset(p)[v][0], np.float64) for p in G]), call[idx], tend_all[idx])
    ts = tsall[idx]
    ny, nx = var['MU'][0].shape[1:]
    jj, ii = np.mgrid[0:ny, 0:nx]
    dist = np.minimum(np.minimum(jj, ny - 1 - jj), np.minimum(ii, nx - 1 - ii))
    rings = {f'ring{k}': dist == k for k in range(10)}
    rings['interior'] = dist >= 10
    res = {'dom': dom, 'frames': [p[-19:] for p in C], 'vars': {}}
    for v, (g, c, tend) in var.items():
        rv = {}
        for rk, m in rings.items():
            dm = (g - c)[:, m].mean(1)
            tm = tend[:, m].mean(1)
            sel = np.isfinite(tm) & (np.abs(dm) + np.abs(tm) > 0)
            lag = float((dm[sel] * tm[sel]).sum() / (tm[sel] ** 2).sum())
            resid = dm[sel] - lag * tm[sel]
            r = float(np.corrcoef(dm[sel], tm[sel])[0, 1])
            rv[rk] = dict(n_cells=int(m.sum()), diff=dm.tolist(), tend=tm.tolist(), lag_s=lag, corr=r,
                          resid_rms=float(np.sqrt((resid ** 2).mean())), diff_rms=float(np.sqrt((dm[sel] ** 2).mean())))
        res['vars'][v] = rv
        print(v, ' '.join(f"{rk}: lag {x['lag_s']:+.1f}s r {x['corr']:+.2f}" for rk, x in rv.items()))
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
