"""Build tests/fixtures/o2_moistadv_swiss_h06.npz: real-structured advect_scalar operands (o2-moistadv).

Source: original CPU-WRF Swiss RD11 wrfout_d01_2023-01-15_06:00:00 (42x42, dx 3 km), lowest NZ mass levels.
Operands follow tests/v025/b_core/bench_scalar_flux.inputs (the BC30 pristine-parity recipe): coupled
ru/rv = (c1h*mu_face+c2h)*wind/map, rom diagnosed by the calc_ww_cp column recurrence, all REAL.
Fields: QVAPOR (smooth, everywhere > 0) and QCLOUD (sparse; positive-definite limiter active).
usage: python build_fixture.py <wrfout> <out.npz>
"""
import sys

import numpy as np
from netCDF4 import Dataset

NZ = 12
src, out = sys.argv[1], sys.argv[2]
with Dataset(src) as d:
    g = lambda k: np.asarray(d.variables[k][0], np.float32)  # noqa: E731
    qv, qc = g("QVAPOR")[:NZ], np.maximum(g("QCLOUD")[:NZ], 0).astype(np.float32)
    u, v = g("U")[:NZ], g("V")[:NZ]
    mu = (g("MU") + g("MUB")).astype(np.float32)
    mapm, mapu, mapv = g("MAPFAC_M"), g("MAPFAC_U"), g("MAPFAC_V")
    c1, c2 = g("C1H")[:NZ], g("C2H")[:NZ]
    dnw, rdnw, fnm, fnp = g("DNW")[:NZ], g("RDNW")[:NZ], g("FNM")[:NZ], g("FNP")[:NZ]
    rdx = np.float32(1.0 / float(d.DX))
nz, ny, nx = qv.shape
mp = np.pad(mu, ((0, 0), (1, 1)), mode="edge")
muu = (0.5 * (mp[:, 1:] + mp[:, :-1])).astype(np.float32)          # (ny, nx+1)
mp = np.pad(mu, ((1, 1), (0, 0)), mode="edge")
muv = (0.5 * (mp[1:] + mp[:-1])).astype(np.float32)                # (ny+1, nx)
ru = ((c1[:, None, None] * muu + c2[:, None, None]) * u / mapu).astype(np.float32)
rv = ((c1[:, None, None] * muv + c2[:, None, None]) * v / mapv).astype(np.float32)
div = dnw[:, None, None] * mapm[None] * (rdx * (ru[:, :, 1:] - ru[:, :, :-1]) + rdx * (rv[:, 1:] - rv[:, :-1]))
dm = np.zeros((ny, nx), np.float32)
for k in range(nz):
    dm = dm + div[k]
rom = np.zeros((nz + 1, ny, nx), np.float32)
for k in range(1, nz):
    rom[k] = rom[k - 1] - dnw[k - 1] * c1[k - 1] * dm - div[k - 1]
assert np.count_nonzero(rom) > 0 and np.count_nonzero(qc) > 0
np.savez_compressed(out, qv=qv, qc=qc, ru=ru[:, :, :nx], rv=rv[:, :ny], rom=rom, mu=mu, map=mapm,
                    fzm=fnm, fzp=fnp, rdzw=rdnw, c1=c1, c2=c2, rdx=np.float32(rdx),
                    source=np.array(src))
print(out, qv.shape, int(np.count_nonzero(qc)))
