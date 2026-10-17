"""Pristine WRF phy_prep / conv_t_tendf_to_moist oracle on real CPU-WRF columns (lane t-cfix-phyprep).

usage: python build_phyprep_oracle.py <build_dir> <fixture.npz>

Independent reference for the legacy surface/PBL/cumulus caller staging (CALLER_AUDIT F4/F5/F15/F16):
* the UNTOUCHED pristine ``SUBROUTINE phy_prep`` and ``SUBROUTINE conv_t_tendf_to_moist`` text
  (dyn_em/module_big_step_utilities_em.F) compiled in a private wrapper with the real
  share/module_model_constants.F; only the two WRF modules phy_prep USEs for flags are stubbed
  (grid_config_rec_type%use_theta_m=1, PARAM_FIRST_SCALAR/P_QV);
* ALT from the literal calc_p_rho_phi hypsometric_opt=2 ``al`` statement (with alb=0, so al == alt up to
  the al+alb round trip WRF does elsewhere);
* inputs: the whole real d01 domain of the c03 composite (Swiss 2023-01-15 12 UTC, mp=2 Lin, cu=2 BMJ,
  bl=2 MYJ) CPU-WRF history frame: THM (=t_2 theta_m-T0), QVAPOR..QGRAUP, P, PB, PH, PHB, MU+MUB, U, V and
  the hybrid/stagger coefficients. Rates for conv_t_tendf_to_moist are deterministic adversarial values.
Builds: REAL*4 (WRF RWORDSIZE=4, -O0 -ffp-contract=off) and REAL*8 (-fdefault-real-8).
The fixture keeps a column subset (cloud/ice/moist/terrain extremes + a fixed random draw).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import netCDF4
import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FRAME = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c03/cpu/wrfout_d01_2023-01-15_12:00:00')
MOIST = ('QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP')
OUT3 = ('th_phy', 'p_phy', 'pi_phy', 't_phy', 'rho', 'z', 'p_hyd', 'u_phy', 'v_phy', 'alt', 'theta_m_rate')
OUTW = ('z_at_w', 'dz8w', 'p8w', 'p_hyd_w')


def _sub(source, name):
    return re.search(rf'(?im)^\s*SUBROUTINE {name}\s*\(.*?^END SUBROUTINE {name}', source, re.S | re.M).group()


def _wrapper(source, nx, ny, nz):
    prep, conv = _sub(source, 'phy_prep'), _sub(source, 'conv_t_tendf_to_moist')
    al = next(l.strip() for l in source.splitlines()
              if l.strip().startswith('al(i,k,j) = (ph(i,k+1,j)-ph(i,k,j)+phb(i,k+1,j)-phb(i,k,j))/phm/LOG(pfd/pfu)'))
    pfu = next(l.strip() for l in source.splitlines() if l.strip().startswith('pfu = c3f(k+1)*MUTS(i,j)'))
    pfd = next(l.strip() for l in source.splitlines() if l.strip().startswith('pfd = c3f(k  )*MUTS(i,j)'))
    phm = next(l.strip() for l in source.splitlines() if l.strip().startswith('phm = c3h(k  )*MUTS(i,j)'))
    nm = len(MOIST) + 1
    return f'''module stub_wrf
implicit none
! Only the flag carriers phy_prep USEs from module_configure/module_state_description.
type grid_config_rec_type
  integer :: use_theta_m
end type
integer, parameter :: PARAM_FIRST_SCALAR=2, P_QV=2
end module
module prep_mod
use module_model_constants
use stub_wrf
implicit none
contains
{prep}
{conv}
end module
program probe
use prep_mod
implicit none
integer, parameter :: nx={nx}, ny={ny}, nz={nz}, nm={nm}
integer :: ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte,i,j,k,u_in
type(grid_config_rec_type) :: config_flags
real :: u(nx+1,nz+1,ny+1), v(nx+1,nz+1,ny+1), p(nx+1,nz+1,ny+1), pb(nx+1,nz+1,ny+1), t(nx+1,nz+1,ny+1)
real :: ph(nx+1,nz+1,ny+1), phb(nx+1,nz+1,ny+1), moist(nx+1,nz+1,ny+1,nm), alt(nx+1,nz+1,ny+1), al(nx+1,nz+1,ny+1)
real :: alb(nx+1,nz+1,ny+1), rth(nx+1,nz+1,ny+1), rqv(nx+1,nz+1,ny+1)
real :: mut(nx+1,ny+1), muts(nx+1,ny+1), muu(nx+1,ny+1), muv(nx+1,ny+1)
real, dimension(nz+1) :: c1h,c2h,c1f,c2f,c3h,c4h,c3f,c4f,fzm,fzp,znw,dnw
real :: p_top, ptop, pfu, pfd, phm
real, dimension(nx+1,nz+1,ny+1) :: rho,th_phy,th_phy_m_t0,p_phy,pi_phy,u_phy,v_phy,p8w,t_phy,t8w,z,z_at_w,dz8w,p_hyd,p_hyd_w
ids=1;ide=nx+1;jds=1;jde=ny+1;kds=1;kde=nz+1;ims=1;ime=nx+1;jms=1;jme=ny+1;kms=1;kme=nz+1
its=1;ite=nx;jts=1;jte=ny;kts=1;kte=nz+1
config_flags%use_theta_m=1
u=0.;v=0.;p=0.;pb=0.;t=0.;ph=0.;phb=0.;moist=0.;alb=0.;rth=0.;rqv=0.;mut=0.;muu=0.;muv=0.
c1h=0.;c2h=0.;c1f=0.;c2f=0.;c3h=0.;c4h=0.;c3f=0.;c4f=0.;fzm=0.;fzp=0.;znw=0.;dnw=0.
open(newunit=u_in,file='input.bin',access='stream',form='unformatted',status='old')
read(u_in) u(1:nx+1,1:nz,1:ny), v(1:nx,1:nz,1:ny+1), p(1:nx,1:nz,1:ny), pb(1:nx,1:nz,1:ny), t(1:nx,1:nz,1:ny)
read(u_in) ph(1:nx,1:nz+1,1:ny), phb(1:nx,1:nz+1,1:ny)
do k=2,nm
  read(u_in) moist(1:nx,1:nz,1:ny,k)
enddo
read(u_in) mut(1:nx,1:ny), c1h(1:nz), c2h(1:nz), c1f(1:nz+1), c2f(1:nz+1), c3h(1:nz), c4h(1:nz), c3f(1:nz+1), c4f(1:nz+1)
read(u_in) fzm(1:nz), fzp(1:nz), znw(1:nz+1), dnw(1:nz), p_top
read(u_in) rth(1:nx,1:nz,1:ny), rqv(1:nx,1:nz,1:ny)
close(u_in)
ptop=p_top;muts=mut
! calc_p_rho_phi hypsometric_opt=2 (literal statements, alb=0 so al is the total inverse density)
do j=1,ny
do k=1,nz
do i=1,nx
{pfu}
{pfd}
{phm}
{al}
enddo
enddo
enddo
alt=al
call phy_prep(config_flags, mut, muu, muv, c1h, c2h, c1f, c2f, u, v, p, pb, alt, ph, phb, t, moist, nm, &
  rho, th_phy, th_phy_m_t0, p_phy, pi_phy, u_phy, v_phy, p8w, t_phy, t8w, z, z_at_w, dz8w, p_hyd, p_hyd_w, dnw, &
  fzm, fzp, znw, p_top, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
call conv_t_tendf_to_moist(t, moist(:,:,:,P_QV), rth, rqv, ids, ide, jds, jde, kds, kde, &
  ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
open(newunit=u_in,file='output.bin',access='stream',form='unformatted',status='replace')
write(u_in) th_phy(1:nx,1:nz,1:ny), p_phy(1:nx,1:nz,1:ny), pi_phy(1:nx,1:nz,1:ny), t_phy(1:nx,1:nz,1:ny)
write(u_in) rho(1:nx,1:nz,1:ny), z(1:nx,1:nz,1:ny), p_hyd(1:nx,1:nz,1:ny), u_phy(1:nx,1:nz,1:ny), v_phy(1:nx,1:nz,1:ny)
write(u_in) alt(1:nx,1:nz,1:ny), rth(1:nx,1:nz,1:ny)
write(u_in) z_at_w(1:nx,1:nz+1,1:ny), dz8w(1:nx,1:nz+1,1:ny), p8w(1:nx,1:nz+1,1:ny), p_hyd_w(1:nx,1:nz+1,1:ny)
close(u_in)
end program
'''


def _f(a):
    """(nz, ny, nx) C-order numpy -> Fortran (nx, nz, ny) stream bytes."""
    return np.asfortranarray(np.transpose(np.asarray(a, np.float32), (2, 0, 1))).tobytes(order='F')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('build', type=Path); ap.add_argument('fixture', type=Path)
    a = ap.parse_args(); a.build.mkdir(parents=True, exist_ok=True)
    source = (WRF / 'dyn_em/module_big_step_utilities_em.F').read_text()
    constants = WRF / 'share/module_model_constants.F'
    with netCDF4.Dataset(FRAME) as d:
        g = lambda n: np.asarray(d.variables[n][0], np.float32)
        nz, ny, nx = g('THM').shape
        fields = dict(U=g('U'), V=g('V'), P=g('P'), PB=g('PB'), THM=g('THM'), PH=g('PH'), PHB=g('PHB'),
                      MUT=(g('MU') + g('MUB')).astype(np.float32), MU=g('MU'), MUB=g('MUB'),
                      **{m: g(m) for m in MOIST})
        for n in ('C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'FNM', 'FNP', 'ZNW', 'DNW'):
            fields[n] = g(n)
        fields['P_TOP'] = np.float32(g('P_TOP'))
        frame_sha = hashlib.sha256(FRAME.read_bytes()).hexdigest()
    rng = np.random.default_rng(20261015)
    fields['RTH'] = rng.uniform(-2.e-3, 2.e-3, (nz, ny, nx)).astype(np.float32)
    fields['RQV'] = rng.uniform(-3.e-6, 3.e-6, (nz, ny, nx)).astype(np.float32)
    pad1 = lambda v, n: np.concatenate([np.asarray(v, np.float32), np.zeros(n - len(v), np.float32)])
    blob = b''.join([_f(fields['U']), _f(fields['V']), _f(fields['P']), _f(fields['PB']), _f(fields['THM']),
                     _f(fields['PH']), _f(fields['PHB'])] + [_f(fields[m]) for m in MOIST] +
                    [np.asfortranarray(fields['MUT'].T).tobytes(order='F')] +
                    [np.asarray(fields[n], np.float32).tobytes() for n in
                     ('C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'FNM', 'FNP', 'ZNW', 'DNW')] +
                    [np.asarray([fields['P_TOP']], np.float32).tobytes(), _f(fields['RTH']), _f(fields['RQV'])])
    wrapper = _wrapper(source, nx, ny, nz)
    rows = {}
    for kind in (32, 64):
        b = a.build / f'r{kind}'; b.mkdir(exist_ok=True)
        (b / 'oracle.F90').write_text(wrapper)
        flags = ['-O0', '-cpp', '-ffree-form', '-ffree-line-length-none', '-ffp-contract=off',
                 '-finit-real=zero', '-finit-integer=0', '-finit-logical=false', f'-J{b}']
        if kind == 64:
            flags += ['-fdefault-real-8', '-fdefault-double-8']
        subprocess.run([FC, *flags, str(constants), str(b / 'oracle.F90'), '-o', str(b / 'oracle')], cwd=b, check=True)
        real = np.float32 if kind == 32 else np.float64
        inp = blob if kind == 32 else np.frombuffer(blob, np.float32).astype(np.float64).tobytes()
        (b / 'input.bin').write_bytes(inp)
        subprocess.run([str(b / 'oracle')], cwd=b, check=True)
        out = np.frombuffer((b / 'output.bin').read_bytes(), real)
        res, off = {}, 0
        for name in OUT3:
            n = nx * nz * ny
            res[name] = np.transpose(out[off:off + n].reshape((nx, nz, ny), order='F'), (1, 2, 0)); off += n
        for name in OUTW:
            n = nx * (nz + 1) * ny
            res[name] = np.transpose(out[off:off + n].reshape((nx, nz + 1, ny), order='F'), (1, 2, 0)); off += n
        assert off == out.size, (off, out.size)
        assert all(np.isfinite(v).all() for k, v in res.items() if k not in ('dz8w', 'p8w')), kind
        rows[kind] = res
    # Column subset: cloud/ice/vapour/terrain/hydrostatic-gap extremes + fixed random draw.
    hgt = fields['PHB'][0] / np.float32(9.81)
    picks = [np.unravel_index(int(np.argmax(x)), (ny, nx)) for x in
             (fields['QCLOUD'].max(0), fields['QICE'].max(0), fields['QVAPOR'][0], hgt, -hgt,
              fields['QSNOW'].max(0), np.abs(rows[32]['p_hyd'][0] - (fields['P'] + fields['PB'])[0]))]
    picks += [tuple(x) for x in zip(*np.unravel_index(rng.choice(ny * nx, 9, replace=False), (ny, nx)))]
    seen, cols = set(), []
    for c in picks:
        if (int(c[0]), int(c[1])) not in seen and 1 <= c[0] < ny - 1 and 1 <= c[1] < nx - 1:
            seen.add((int(c[0]), int(c[1]))); cols.append((int(c[0]), int(c[1])))
    jj, ii = np.asarray([c[0] for c in cols]), np.asarray([c[1] for c in cols])
    # Centre columns only: every compared phy_prep field is column-local (u_phy/v_phy are not compared).
    win = lambda x: np.stack([x[..., j, i] for j, i in cols])
    fixture = dict(cols=np.asarray(cols, np.int32), U=win(fields['U']), V=win(fields['V']),
                   **{n: win(fields[n]) for n in ('P', 'PB', 'THM', 'PH', 'PHB', 'MUT', 'MU', 'MUB', 'RTH', 'RQV', *MOIST)},
                   **{n: np.asarray(fields[n]) for n in ('C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F',
                                                       'FNM', 'FNP', 'ZNW', 'DNW')},
                   P_TOP=np.asarray(fields['P_TOP'], np.float32))
    for kind, res in rows.items():
        for name, value in res.items():
            fixture[f'wrf{kind}_{name}'] = win(value)
    a.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.fixture, **fixture)
    manifest = dict(
        reference='UNTOUCHED pristine WRF 4.7.1 SUBROUTINE phy_prep + conv_t_tendf_to_moist + calc_p_rho_phi '
                  'hypsometric_opt=2 al statements (alb=0), private wrapper; flags stubbed use_theta_m=1/P_QV=2',
        frame=str(FRAME), frame_sha256=frame_sha, columns=cols, window='centre column [col, k] (U/V: west/south face of the column)', rates='RTH U(-2e-3,2e-3) K/s, RQV U(-3e-6,3e-6) kg/kg/s, seed 20261015',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        constants_sha256=hashlib.sha256(constants.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(), compiler=FC,
        flags='-O0 -cpp -ffp-contract=off -finit-real=zero (+ -fdefault-real-8 -fdefault-double-8 for r64)',
        fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
        outputs=dict(mass=OUT3, faces=OUTW),
    )
    a.fixture.with_suffix('.json').write_text(json.dumps(manifest, indent=1, allow_nan=False) + '\n')
    print(json.dumps(dict(columns=len(cols), fixture_sha256=manifest['fixture_sha256'],
                          max_moist_minus_dry_T=float(np.max(fields['THM'] + 300 - rows[32]['th_phy'])))))


if __name__ == '__main__':
    main()
