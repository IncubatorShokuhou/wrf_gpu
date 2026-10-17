"""F20 pristine public-call oracle: WRF 4.7.1 ``mynnedmf_driver`` (unmodified objects) on real columns.

``build(out, instrument=False)`` links a small batch caller against the pristine
module_bl_mynnedmf_driver.o / module_bl_mynnedmf.o (the caller mirrors module_pbl_driver: w-staggered
``w`` = grid%w_2 with kme = nz+1, Registry-default switches, initflag=0). ``instrument=True`` swaps in a
lane-dir copy of module_bl_mynnedmf.F whose ONLY edits are two logging WRITEs (bit patterns of
maxw/Psig_w/Psig_shcu/pblh after F90:5879 and of hfx/qfx/flt/flqv/fltv after F90:876), compiled with the
pristine configure.wrf FCOPTIM/FCBASEOPTS; callers must prove its public outputs BYTE-equal the unmodified
build (E26).
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WRF = ROOT / "data/wrf_pristine/WRF"
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
WRF_FLAGS = ["-O2", "-ftree-vectorize", "-funroll-loops", "-w", "-ffree-form", "-ffree-line-length-none",
             "-fconvert=big-endian", "-frecord-marker=4"]
FIELDS3 = ("u", "v", "w", "th", "t3d", "p", "exner", "rho", "dz", "qv", "qc", "qi", "qs", "qni", "qke", "el_pbl")
FIELDS2 = ("xland", "tsk", "qsfc", "psfc", "ust", "ch", "hfx", "qfx", "wspd", "znt")
OUT2 = ("pblh", "maxmf", "maxwidth", "ztop_plume", "kpbl")
OUT3 = ("rthblten", "rqvblten", "rqcblten", "rqiblten", "rublten", "rvblten", "rqniblten", "qke_out",
        "edmf_a", "edmf_w", "exch_h", "qc_bl", "cldfra_bl")

CALLER = """program f20_public
  use module_bl_mynnedmf_common, only: kind_phys
  use module_bl_mynnedmf_driver, only: mynnedmf_driver
  implicit none
  integer :: n, nz, nzp, ui, uo
  real(kind_phys) :: delt, dxc
  real(kind_phys), allocatable, dimension(:,:,:) :: u,v,w,th,t3d,p,exner,rho,dz,rthraten, &
     qv,qc,qi,qs,qnc,qni,qnwfa,qnifa,qnbca,qke,qke_adv,tsq,qsq,cov,el_pbl,sh3d,sm3d,exch_h,exch_m, &
     rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,rqncblten,rqniblten,rqnwfablten, &
     rqnifablten,rqnbcablten,qc_bl,qi_bl,cldfra_bl,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc, &
     sub_thl3d,sub_sqv3d,det_thl3d,det_sqv3d,dqke,qwt,qshear,qbuoy,qdiss,pattern_spp_pbl
  real(kind_phys), allocatable, dimension(:,:) :: xland,ts,qsfc,ps,ust,ch,hfx,qfx,wspd,znt,uoce,voce, &
     pblh,maxwidth,maxmf,ztop_plume
  integer, allocatable :: kpbl(:,:)
  open(newunit=ui, file='in.bin', access='stream', form='unformatted', status='old', convert='native')
  read(ui) n, nz
  read(ui) delt, dxc
  nzp = nz + 1
  allocate(u(n,nzp,1),v(n,nzp,1),w(n,nzp,1),th(n,nzp,1),t3d(n,nzp,1),p(n,nzp,1),exner(n,nzp,1), &
     rho(n,nzp,1),dz(n,nzp,1),rthraten(n,nzp,1),qv(n,nzp,1),qc(n,nzp,1),qi(n,nzp,1),qs(n,nzp,1), &
     qnc(n,nzp,1),qni(n,nzp,1),qnwfa(n,nzp,1),qnifa(n,nzp,1),qnbca(n,nzp,1),qke(n,nzp,1),qke_adv(n,nzp,1), &
     tsq(n,nzp,1),qsq(n,nzp,1),cov(n,nzp,1),el_pbl(n,nzp,1),sh3d(n,nzp,1),sm3d(n,nzp,1),exch_h(n,nzp,1), &
     exch_m(n,nzp,1),rublten(n,nzp,1),rvblten(n,nzp,1),rthblten(n,nzp,1),rqvblten(n,nzp,1), &
     rqcblten(n,nzp,1),rqiblten(n,nzp,1),rqsblten(n,nzp,1),rqncblten(n,nzp,1),rqniblten(n,nzp,1), &
     rqnwfablten(n,nzp,1),rqnifablten(n,nzp,1),rqnbcablten(n,nzp,1),qc_bl(n,nzp,1),qi_bl(n,nzp,1), &
     cldfra_bl(n,nzp,1),edmf_a(n,nzp,1),edmf_w(n,nzp,1),edmf_qt(n,nzp,1),edmf_thl(n,nzp,1), &
     edmf_ent(n,nzp,1),edmf_qc(n,nzp,1),sub_thl3d(n,nzp,1),sub_sqv3d(n,nzp,1),det_thl3d(n,nzp,1), &
     det_sqv3d(n,nzp,1),dqke(n,nzp,1),qwt(n,nzp,1),qshear(n,nzp,1),qbuoy(n,nzp,1),qdiss(n,nzp,1), &
     pattern_spp_pbl(n,nzp,1))
  allocate(xland(n,1),ts(n,1),qsfc(n,1),ps(n,1),ust(n,1),ch(n,1),hfx(n,1),qfx(n,1),wspd(n,1),znt(n,1), &
     uoce(n,1),voce(n,1),pblh(n,1),maxwidth(n,1),maxmf(n,1),ztop_plume(n,1),kpbl(n,1))
  rthraten=0.; qnc=0.; qnwfa=0.; qnifa=0.; qnbca=0.; qke_adv=0.; tsq=0.; qsq=0.; cov=0.; sh3d=0.; sm3d=0.
  exch_h=0.; exch_m=0.; rublten=0.; rvblten=0.; rthblten=0.; rqvblten=0.; rqcblten=0.; rqiblten=0.
  rqsblten=0.; rqncblten=0.; rqniblten=0.; rqnwfablten=0.; rqnifablten=0.; rqnbcablten=0.
  qc_bl=0.; qi_bl=0.; cldfra_bl=0.; edmf_a=0.; edmf_w=0.; pattern_spp_pbl=0.; uoce=0.; voce=0.
  pblh=0.; maxwidth=0.; maxmf=0.; ztop_plume=0.; kpbl=1
  read(ui) u, v, w, th, t3d, p, exner, rho, dz, qv, qc, qi, qs, qni, qke, el_pbl
  read(ui) xland, ts, qsfc, ps, ust, ch, hfx, qfx, wspd, znt
  close(ui)
  ! module_pbl_driver.F:1715 CASE(MYNNPBLSCHEME); physics kte = kde-1; Registry defaults, mp8 flags.
  call mynnedmf_driver(ids=1, ide=n+1, jds=1, jde=2, kds=1, kde=nzp, ims=1, ime=n, jms=1, jme=1, &
     kms=1, kme=nzp, its=1, ite=n, jts=1, jte=1, kts=1, kte=nz, flag_qc=.true., flag_qi=.true., &
     flag_qs=.true., flag_qnc=.false., flag_qni=.true., flag_qnifa=.false., flag_qnwfa=.false., &
     flag_qnbca=.false., initflag=0, restart=.false., cycling=.false., delt=delt, dxc=dxc, &
     xland=xland, ps=ps, ts=ts, qsfc=qsfc, ust=ust, ch=ch, hfx=hfx, qfx=qfx, wspd=wspd, znt=znt, &
     uoce=uoce, voce=voce, dz=dz, u=u, v=v, w=w, th=th, t3d=t3d, p=p, exner=exner, rho=rho, qv=qv, &
     qc=qc, qi=qi, qs=qs, qnc=qnc, qni=qni, qnifa=qnifa, qnwfa=qnwfa, qnbca=qnbca, rthraten=rthraten, &
     pblh=pblh, kpbl=kpbl, cldfra_bl=cldfra_bl, qc_bl=qc_bl, qi_bl=qi_bl, maxwidth=maxwidth, &
     maxmf=maxmf, ztop_plume=ztop_plume, qke=qke, qke_adv=qke_adv, tsq=tsq, qsq=qsq, cov=cov, &
     el_pbl=el_pbl, rublten=rublten, rvblten=rvblten, rthblten=rthblten, rqvblten=rqvblten, &
     rqcblten=rqcblten, rqiblten=rqiblten, rqsblten=rqsblten, rqncblten=rqncblten, &
     rqniblten=rqniblten, rqnifablten=rqnifablten, rqnwfablten=rqnwfablten, rqnbcablten=rqnbcablten, &
     edmf_a=edmf_a, edmf_w=edmf_w, edmf_qt=edmf_qt, edmf_thl=edmf_thl, edmf_ent=edmf_ent, &
     edmf_qc=edmf_qc, sub_thl3d=sub_thl3d, sub_sqv3d=sub_sqv3d, det_thl3d=det_thl3d, &
     det_sqv3d=det_sqv3d, exch_h=exch_h, exch_m=exch_m, dqke=dqke, qwt=qwt, qshear=qshear, &
     qbuoy=qbuoy, qdiss=qdiss, sh3d=sh3d, sm3d=sm3d, spp_pbl=0, pattern_spp_pbl=pattern_spp_pbl, &
     icloud_bl=1, bl_mynn_tkeadvect=.false., tke_budget=0, bl_mynn_cloudpdf=2, bl_mynn_mixlength=1, &
     bl_mynn_closure=2.6_kind_phys, bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, &
     bl_mynn_output=1, bl_mynn_mixscalars=1, bl_mynn_cloudmix=1, bl_mynn_mixqt=0)
  open(newunit=uo, file='out.bin', access='stream', form='unformatted', status='replace', convert='native')
  write(uo) pblh, maxmf, maxwidth, ztop_plume, real(kpbl, kind_phys)
  write(uo) rthblten, rqvblten, rqcblten, rqiblten, rublten, rvblten, rqniblten, qke, edmf_a, edmf_w, &
     exch_h, qc_bl, cldfra_bl
  close(uo)
end program
"""

PSIG_ANCHOR = " Psig_w = min(Psig_w, Psig_shcu)\n"
PSIG_LOG = (" write(*,'(A,I8,4(1X,Z8.8))') 'F20PSIG', i, transfer(maxw,1), transfer(Psig_w,1), "
            "transfer(Psig_shcu,1), transfer(pblh,1)\n")
FLUX_ANCHOR = "    fltv=flt + flqv*p608*th_sfc      !! Virtual temperature flux\n"
FLUX_LOG = ("    write(*,'(A,I8,5(1X,Z8.8))') 'F20FLUX', i, transfer(hfx,1), transfer(qfx,1), "
            "transfer(flt,1), transfer(flqv,1), transfer(fltv,1)\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(cmd, cwd, timeout):
    return subprocess.run(["timeout", str(timeout), *cmd], cwd=cwd, check=True, capture_output=True, text=True)


def build(out, *, instrument=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    phys = WRF / "phys"
    core = phys / "module_bl_mynnedmf.o"
    provenance = {"wrf_driver_F_sha256": _sha(phys / "module_bl_mynnedmf_driver.F"),
                  "wrf_core_F_sha256": _sha(phys / "module_bl_mynnedmf.F")}
    if instrument:
        text = (phys / "module_bl_mynnedmf.F").read_text()
        assert text.count(PSIG_ANCHOR) == 1 and text.count(FLUX_ANCHOR) == 1
        text = text.replace(PSIG_ANCHOR, PSIG_ANCHOR + PSIG_LOG).replace(FLUX_ANCHOR, FLUX_ANCHOR + FLUX_LOG)
        (out / "module_bl_mynnedmf.F").write_text(text)
        _run([FC, "-cpp", *WRF_FLAGS, "-I" + str(phys), "-c", "module_bl_mynnedmf.F", "-o", "core.o"], out, 600)
        core = out / "core.o"
        provenance["instrumented_core_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    (out / "caller.f90").write_text(CALLER)
    objects = [phys / "module_bl_mynnedmf_driver.o", core, phys / "module_bl_mynnedmf_common.o",
               phys / "ccpp_kind_types.o", WRF / "share/module_model_constants.o"]
    include = [f"-I{out}"] if instrument else []
    _run([FC, *include, "-I" + str(phys), "-ffree-line-length-none", "caller.f90",
          *[str(p) for p in objects], "-o", "oracle"], out, 120)
    provenance["objects"] = {str(p): _sha(p) for p in objects}
    provenance["caller_sha256"] = hashlib.sha256(CALLER.encode()).hexdigest()
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    return out / "oracle"


def run(executable, workdir, cols, dt, dx):
    """``cols``: dict of (n, nz) / (n, nz+1) for ``w`` / (n,) float32 arrays. Returns (outputs, log)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n, nz = cols["th"].shape
    blob = [np.asarray([n, nz], np.int32).tobytes(), np.asarray([dt, dx], np.float32).tobytes()]
    for name in FIELDS3:
        a = np.zeros((n, nz + 1), np.float32)
        value = np.asarray(cols[name], np.float32)
        a[:, :value.shape[1]] = value
        blob.append(a.T.tobytes())  # Fortran (n, nz+1, 1) order
    for name in FIELDS2:
        blob.append(np.asarray(cols[name], np.float32).tobytes())
    (workdir / "in.bin").write_bytes(b"".join(blob))
    result = _run([str(executable)], workdir, 120)
    raw = np.frombuffer((workdir / "out.bin").read_bytes(), np.float32)
    outputs, offset = {}, 0
    for name in OUT2:
        outputs[name] = raw[offset:offset + n].copy()
        offset += n
    for name in OUT3:
        outputs[name] = raw[offset:offset + n * (nz + 1)].reshape(nz + 1, n).T[:, :nz].copy()
        offset += n * (nz + 1)
    assert offset == raw.size
    log = {"F20PSIG": {}, "F20FLUX": {}}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] in log:
            bits = np.asarray([int(x, 16) for x in parts[2:]], np.uint32).view(np.float32)
            log[parts[0]][int(parts[1])] = bits
    return outputs, log
