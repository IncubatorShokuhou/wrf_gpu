"""Load one domain of a native wrfout directory into a compact npz cache (o5-coast).

usage: python load_raw.py WRFOUT_DIR DOMAIN OUT.npz [NFRAMES]
       python load_raw.py --frames FRAME_NPZ_DIR DOMAIN OUT.npz   (stack per-frame npz written by stream_extract.py)
Per frame (hourly files sorted by name): 2-D natives U10 V10 T2 Q2 PSFC HFX LH PBLH UST TSK SWDOWN GLW RAINNC MU,
earth-rotated 10 m wind (u10_east/v10_north, same rotation as teacher_extract), level 0/1 mass-point wind
(destaggered, earth-rotated), theta level 0, z_agl of mass levels 0/1, QV level 0. Statics: XLAT XLONG HGT LANDMASK
XLAND LAKEMASK IVGTYP LU_INDEX COSALPHA SINALPHA, times. No JAX."""
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

G = 9.81
NAT = ('U10', 'V10', 'T2', 'Q2', 'PSFC', 'HFX', 'LH', 'PBLH', 'UST', 'TSK', 'SWDOWN', 'GLW', 'RAINNC', 'MU')
STAT = ('XLAT', 'XLONG', 'HGT', 'LANDMASK', 'XLAND', 'LAKEMASK', 'IVGTYP', 'LU_INDEX', 'COSALPHA', 'SINALPHA')
DER = ('u10_east', 'v10_north', 'ul0', 'vl0', 'ul1', 'vl1', 'th0', 'z0', 'z1', 'qv0')


def frame(path):
    """One wrfout file -> dict of 2-D float32 fields, statics (s_*) and the Times string."""
    out = {}
    with nc.Dataset(path) as f:
        v = lambda n: np.asarray(f.variables[n][0])
        stat = {k: v(k) for k in STAT if k in f.variables}
        for k in NAT:
            out[k] = v(k).astype(np.float32)
        ca = stat['COSALPHA'].astype(np.float64)
        sa = stat['SINALPHA'].astype(np.float64)
        u10 = v('U10').astype(np.float64)
        v10 = v('V10').astype(np.float64)
        out['u10_east'] = (u10 * ca - v10 * sa).astype(np.float32)
        out['v10_north'] = (v10 * ca + u10 * sa).astype(np.float32)
        U = np.asarray(f.variables['U'][0, :2], np.float64)
        V = np.asarray(f.variables['V'][0, :2], np.float64)
        um = 0.5 * (U[:, :, :-1] + U[:, :, 1:])
        vm = 0.5 * (V[:, :-1, :] + V[:, 1:, :])
        ue = um * ca - vm * sa
        vn = vm * ca + um * sa
        out['ul0'], out['vl0'] = ue[0].astype(np.float32), vn[0].astype(np.float32)
        out['ul1'], out['vl1'] = ue[1].astype(np.float32), vn[1].astype(np.float32)
        out['th0'] = (np.asarray(f.variables['T'][0, 0], np.float64) + 300.0).astype(np.float32)
        out['qv0'] = np.asarray(f.variables['QVAPOR'][0, 0], np.float32)
        ph = np.asarray(f.variables['PH'][0, :3], np.float64) + np.asarray(f.variables['PHB'][0, :3], np.float64)
        zw = ph / G
        hgt = stat['HGT'].astype(np.float64)
        out['z0'] = (0.5 * (zw[0] + zw[1]) - hgt).astype(np.float32)
        out['z1'] = (0.5 * (zw[1] + zw[2]) - hgt).astype(np.float32)
        out['times'] = np.array(b''.join(np.asarray(f.variables['Times'][0]).tolist()).decode())
    out.update({f's_{k}': x for k, x in stat.items()})
    return out


def stack(frames):
    arr = {k: np.stack([fr[k] for fr in frames]) for k in NAT + DER}
    arr['times'] = np.array([str(fr['times']) for fr in frames])
    arr.update({k: frames[0][k] for k in frames[0] if k.startswith('s_')})
    return arr


def load(wdir, dom, nframes=None):
    files = sorted(glob.glob(os.path.join(wdir, f'wrfout_{dom}_*')))
    if nframes:
        files = files[:nframes]
    return stack([frame(p) for p in files])


def load_frames(fdir, dom):
    files = sorted(glob.glob(os.path.join(fdir, f'wrfout_{dom}_*.npz')))
    frames = []
    for p in files:
        z = np.load(p)
        frames.append({k: z[k] for k in z.files})
    return stack(frames)


if __name__ == '__main__':
    if sys.argv[1] == '--frames':
        a = load_frames(sys.argv[2], sys.argv[3])
        out = sys.argv[4]
    else:
        nf = int(sys.argv[4]) if len(sys.argv) > 4 else None
        a = load(sys.argv[1], sys.argv[2], nf)
        out = sys.argv[3]
    np.savez_compressed(out, **a)
    print(out, {k: v.shape for k, v in a.items() if k in ('U10', 'times')}, a['times'][0], a['times'][-1])
