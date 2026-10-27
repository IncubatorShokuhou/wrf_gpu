"""Verbatim pristine WRF 4.7.1 final-RK scalar chain, REAL4, one C-callable driver.

Routines are extracted unchanged at build time (never edited):
  dyn_em/module_big_step_utilities_em.F  horizontal_diffusion, sixth_order_diffusion
  dyn_em/module_em.F                     rk_update_scalar, rk_update_scalar_pd
  dyn_em/module_advect_em.F              advect_scalar_pd
  share/module_bc.F                      flow_dep_bdy

``chain`` reproduces solve_em's caller wiring for ONE positive-definite species of a
specified root domain (moist_adv_opt = scalar_adv_opt = 1, diff_opt = 1, diff_6th_opt):

  moist_tend  = physics (update_phy_ten, operand)                       first_rk_step_part2
  moist_tend += horizontal_diffusion('m', moist, ..., muts, khdif/prandtl, xkhh)   module_em.F:1380
  moist_tend += sixth_order_diffusion('m', moist, ..., muts, dt_rk1)                module_em.F:1423
  QV only: relax_bdy_scalar adds (operand), spec_bdy_scalar overwrites rings < spec_zone  solve_em.F:2345-2380
  rk_update_scalar_pd(moist_old, moist_tend, mu_1, mu_1, mub, rk_step=3, dt)        solve_em.F:1942
  advect_scalar_pd(moist, moist_old, ..., muts, mub, mu_1, time_step, dt)           module_em.F:1257
  rk_update_scalar(rk_step=3, moist_old, moist, sc_tend(consumed = 0), mu_1, mu_2)  solve_em.F:2393
  flow_dep_bdy(moist, ru_m, rv_m, spec_zone) for non-QV species                     solve_em.F:2426-2438

Operand mapping for port captures: the port carries FULL dry masses, so every WRF
(perturbation, base) pair is fed as (full, 0): MUOLD = MU_OLD + MU_BASE reproduces the
port's full mass exactly instead of re-splitting it.  khdif = 0 (Swiss/PROD namelist).
Memory: i -3:nx+4, k 1:nz+1, j -3:ny+4 (edge-repeated halo); WRF serial tile its=1..ide.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
FC = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
SPEC = {
    "dyn_em/module_big_step_utilities_em.F": ("horizontal_diffusion", "sixth_order_diffusion"),
    "dyn_em/module_em.F": ("rk_update_scalar", "rk_update_scalar_pd"),
    "dyn_em/module_advect_em.F": ("advect_scalar_pd",),
    "share/module_bc.F": ("flow_dep_bdy",),
}
INTEGER_FLAGS = {"h_sca_adv_order": 5, "v_sca_adv_order": 3, "diff_6th_slopeopt": 0, "rk_ord": 3}
REAL_FLAGS = {"diff_6th_thresh": "0."}

DRIVER = r"""
subroutine chain(nx,ny,nz,spec_zone,flow,ownqv,opt,iso, &
  qold,qcur,xkhh,phys,bdy,ru,rv,ww,sc_in,pre_in,adv_in,upd_in,muts1,mu1,mut3,mu2, &
  mx,my,ux,uy,vx,vy,fnm,fnp,rdnw,c1,c2,rdx,rdy,dt,dt1,factor, &
  o_hd,o_d6,o_sc,o_pre,o_adv,o_upd,o_fin) bind(c)
integer(c_int),value :: nx,ny,nz,spec_zone,flow,ownqv,opt,iso
real(c_float),value :: rdx,rdy,dt,dt1,factor
real(c_float),dimension(-3:nx+4,1:nz+1,-3:ny+4),intent(in) :: qold,qcur,xkhh,phys,bdy,ru,rv,ww
real(c_float),dimension(-3:nx+4,1:nz+1,-3:ny+4),intent(in) :: sc_in,pre_in,adv_in,upd_in
real(c_float),dimension(-3:nx+4,-3:ny+4),intent(in) :: muts1,mu1,mut3,mu2,mx,my,ux,uy,vx,vy
real(c_float),dimension(nz+1),intent(in) :: fnm,fnp,rdnw,c1,c2
real(c_float),dimension(-3:nx+4,1:nz+1,-3:ny+4),intent(out) :: o_hd,o_d6,o_sc,o_pre,o_adv,o_upd,o_fin
real :: sc(-3:nx+4,1:nz+1,-3:ny+4,1),s1(-3:nx+4,1:nz+1,-3:ny+4,1),s2(-3:nx+4,1:nz+1,-3:ny+4,1)
real :: ht(-3:nx+4,1:nz+1,-3:ny+4),zt(-3:nx+4,1:nz+1,-3:ny+4),ph0(-3:nx+4,1:nz+1,-3:ny+4)
real :: vxinv(-3:nx+4,-3:ny+4),mub0(-3:nx+4,-3:ny+4),advu(-3:nx+4,1:nz+1,-3:ny+4)
integer :: i,j,k,ring
type(grid_config_rec_type) :: cfg
cfg%specified=.true.
vxinv=1./vx; mub0=0.; ph0=0.; ht=0.; zt=0.
! isolated operator diagnostics (each from a zero tendency)
o_hd=0.
call horizontal_diffusion('m',qold,o_hd,muts1,c1,c2,cfg,ux,uy,vx,vxinv,vy,mx,my,0.,xkhh,rdx,rdy, &
  1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
o_d6=0.
if(opt/=0) call sixth_order_diffusion('m',qold,o_d6,muts1,dt1,cfg,c1,c2,opt,factor,ph0,ph0,rdx,rdy, &
  mx,my,ux,uy,vx,vy,1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
! WRF-order cumulative moist_tend: physics, += hdiff, += sixth, QV relax/spec
sc(:,:,:,1)=phys
call horizontal_diffusion('m',qold,sc(:,:,:,1),muts1,c1,c2,cfg,ux,uy,vx,vxinv,vy,mx,my,0.,xkhh,rdx,rdy, &
  1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
if(opt/=0) call sixth_order_diffusion('m',qold,sc(:,:,:,1),muts1,dt1,cfg,c1,c2,opt,factor,ph0,ph0,rdx,rdy, &
  mx,my,ux,uy,vx,vy,1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
if(ownqv/=0)then
  do j=1,ny; do k=1,nz; do i=1,nx
    ring=min(min(i-1,nx-i),min(j-1,ny-j))
    if(ring<spec_zone)then
      sc(i,k,j,1)=bdy(i,k,j)
    else
      sc(i,k,j,1)=sc(i,k,j,1)+bdy(i,k,j)
    endif
  enddo; enddo; enddo
endif
! iso/=0: every operator below receives the PORT's own input (sc_in, pre_in, adv_in, upd_in)
if(iso/=0) sc(:,:,:,1)=sc_in
o_sc=sc(:,:,:,1)
s1(:,:,:,1)=qold
call rk_update_scalar_pd(1,1,s1,sc,c1,c2,mu1,mu1,mub0,3,dt,spec_zone,cfg, &
  1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
o_pre=s1(:,:,:,1)
if(iso/=0) s1(:,:,:,1)=pre_in
o_adv=0.
call advect_scalar_pd(qcur,s1(:,:,:,1),o_adv,ht,zt,ru,rv,ww,c1,c2,mut3,mub0,mu1,1,cfg,.false., &
  ux,uy,vx,vy,mx,my,fnm,fnp,rdx,rdy,rdnw,dt, &
  1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
s2(:,:,:,1)=qcur
advu=o_adv
if(iso/=0) advu=adv_in
call rk_update_scalar(scs=1,sce=1,scalar_1=s1,scalar_2=s2,sc_tend=sc,advect_tend=advu, &
  h_tendency=ht,z_tendency=zt,msftx=mx,msfty=my,c1=c1,c2=c2,mu_old=mu1,mu_new=mu2,mu_base=mub0, &
  rk_step=3,dt=dt,spec_zone=spec_zone,config_flags=cfg,tenddec=.false., &
  ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1,ims=-3,ime=nx+4,jms=-3,jme=ny+4,kms=1,kme=nz+1, &
  its=1,ite=nx+1,jts=1,jte=ny+1,kts=1,kte=nz+1)
o_upd=s2(:,:,:,1)
if(iso/=0) s2(:,:,:,1)=upd_in
if(flow/=0) call flow_dep_bdy(s2(:,:,:,1),ru,rv,cfg,spec_zone, &
  1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
o_fin=s2(:,:,:,1)
end subroutine

subroutine wrf_error_fatal(message)
character(*),intent(in) :: message
print *,message
error stop 99
end subroutine
"""


def routines() -> dict[str, str]:
    out = {}
    for rel, names in SPEC.items():
        text = (WRF / rel).read_text()
        for name in names:
            out[name] = re.search(r"(?ims)^\s*SUBROUTINE " + name + r"\s*\(.*?^\s*END SUBROUTINE " + name + r"\b",
                                  text).group(0)
    return out


def build(directory: Path, contraction: str = "off") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bodies = routines()
    flags = sorted({f.lower() for b in bodies.values() for f in re.findall(r"config_flags%(\w+)", b, re.I)})
    decl = []
    for flag in flags:
        if flag in INTEGER_FLAGS:
            decl.append(f"integer :: {flag}={INTEGER_FLAGS[flag]}")
        elif flag in REAL_FLAGS:
            decl.append(f"real :: {flag}={REAL_FLAGS[flag]}")
        else:
            decl.append(f"logical :: {flag}=.false.")
    header = ("module o5_pristine_chain\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n"
              + "\n".join(decl) + "\nend type\ncharacter(512) :: wrf_err_message\ncontains\n")
    source = directory / "pristine_chain.F90"
    source.write_text(header + "\n".join(bodies.values()) + "\n" + DRIVER + "\nend module\n")
    lib = directory / f"chain_{contraction}.so"
    command = [FC, "-cpp", "-O2", "-march=native", "-fPIC", "-shared", "-ffree-line-length-none", "-ffp-contract=" + contraction,
               "-fcheck=bounds", "-J", str(directory), str(source), "-o", str(lib)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = dict(
        sources={rel: hashlib.sha256((WRF / rel).read_bytes()).hexdigest() for rel in SPEC},
        routine_sha256={n: hashlib.sha256(b.encode()).hexdigest() for n, b in bodies.items()},
        driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest(),
        flags={f: INTEGER_FLAGS.get(f, REAL_FLAGS.get(f, ".false.")) for f in flags}, specified=True,
        command=command, real_kind_bytes=4)
    (directory / f"manifest_{contraction}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return lib


H = 4  # halo cells each side in i and j


class Chain:
    def __init__(self, lib: Path):
        self.lib = ctypes.CDLL(str(lib))
        self.fn = self.lib.chain
        # E117 diagnostic only: x86 MXCSR FTZ|DAZ for this thread (emulates XLA:CPU denormal flushing).
        mx = Path(lib).parent / "mxcsr" / "mxcsr.so"
        if not mx.exists():
            mx.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gcc", "-O2", "-shared", "-fPIC", str(Path(__file__).with_name("mxcsr.c")), "-o", str(mx)],
                           check=True)
        self.mx = ctypes.CDLL(str(mx))
        self.fn.restype = None
        # 8 ints, 12 volumes, 10 planes, 5 level vectors, 5 REAL scalars, 7 output volumes
        self.fn.argtypes = ([ctypes.c_int] * 8 + [ctypes.c_void_p] * (12 + 10 + 5) + [ctypes.c_float] * 5
                            + [ctypes.c_void_p] * 7)

    def __call__(self, *, nx, ny, nz, spec_zone, flow, ownqv, opt, qold, qcur, xkhh, phys, bdy, ru, rv, ww,
                 muts1, mu1, mut3, mu2, mx, my, ux, uy, vx, vy, fnm, fnp, rdnw, c1, c2, rdx, rdy, dt, dt1, factor,
                 sc_in=None, pre_in=None, adv_in=None, upd_in=None, ftz=False):
        """iso mode when sc_in/pre_in/adv_in/upd_in are given: each operator gets the port's own input."""
        iso = sc_in is not None
        if iso:
            assert pre_in is not None and adv_in is not None and upd_in is not None
        else:
            sc_in = pre_in = adv_in = upd_in = np.zeros_like(np.asarray(qold, np.float32))
        def vol(a):
            a = np.asarray(a, np.float32)
            a = np.pad(a, ((0, nz + 1 - a.shape[0]), (H, ny + H - a.shape[1]), (H, nx + H - a.shape[2])), mode="edge")
            return np.asfortranarray(a.transpose(2, 0, 1))

        def plane(a):
            a = np.asarray(a, np.float32)
            return np.asfortranarray(np.pad(a, ((H, ny + H - a.shape[0]), (H, nx + H - a.shape[1])), mode="edge").T)

        def vec(a):
            a = np.asarray(a, np.float32).reshape(-1)
            return np.ascontiguousarray(np.pad(a, (0, nz + 1 - a.size), mode="edge"))

        ins = [vol(a) for a in (qold, qcur, xkhh, phys, bdy, ru, rv, ww, sc_in, pre_in, adv_in, upd_in)]
        ins += [plane(a) for a in (muts1, mu1, mut3, mu2, mx, my, ux, uy, vx, vy)]
        ins += [vec(a) for a in (fnm, fnp, rdnw, c1, c2)]
        outs = [np.zeros((nx + 2 * H, nz + 1, ny + 2 * H), np.float32, order="F") for _ in range(7)]
        if ftz:
            self.mx.o5_ftz_on()
        self.fn(nx, ny, nz, spec_zone, int(flow), int(ownqv), opt, int(iso), *[a.ctypes.data for a in ins],
                np.float32(rdx), np.float32(rdy), np.float32(dt), np.float32(dt1), np.float32(factor),
                *[o.ctypes.data for o in outs])
        if ftz:
            self.mx.o5_ftz_off()
        names = ("hdiff", "diff6", "sc", "pre", "adv", "upd", "fin")
        return {n: o[H:H + nx, :nz, H:H + ny].transpose(1, 2, 0).copy() for n, o in zip(names, outs)}
