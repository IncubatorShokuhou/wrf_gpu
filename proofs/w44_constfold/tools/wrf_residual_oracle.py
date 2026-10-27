"""Pristine-gfortran oracle for the w44-constfold residual constants (WRF's OWN folded REAL bits).

Every entry reproduces the cited WRF 4.7.1 declaration/statement verbatim (same operands, parentheses and
left-to-right order).  PARAMETER chains come from the pristine modules themselves (USE of the build's .mod
files: module_model_constants, module_bl_mynnedmf_common); module-PRIVATE parameters are redeclared verbatim
with their source line.  Thompson gamma constants use WRF's own GAMMLN/WGAMMA copied verbatim
(proofs/thompson/wrf_constants/wrf_constants.F90 module wg_probe).  Compiler = the pristine build compiler.

usage: python wrf_residual_oracle.py OUT_JSON WORKDIR
"""
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WRF = Path(os.environ.get("WRF_PRISTINE", "<USER_HOME>/src/wrf_pristine/WRF"))
FC = os.environ.get("FC", "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/gfortran")

# (label, Fortran expression, WRF source reference)
ENTRIES = [
    # MYNN (module_bl_mynnedmf.F90, kind_phys = REAL)
    ("mynn_qsat_blend_chi_den", "(t0c-6.) - tice", "MYNN-EDMF/module_bl_mynnedmf.F90:7664 chi=((t0c-6.)-t)/((t0c-6.)-tice)"),
    ("mynn_qsat_blend_t0c_m6", "t0c-6.", "MYNN-EDMF/module_bl_mynnedmf.F90:7648/7664"),
    ("mynn_xl_blend_cpv_cliq", "cpv-cliq", "MYNN-EDMF/module_bl_mynnedmf.F90:7686/7690"),
    ("mynn_xl_blend_cpv_cice", "cpv-cice", "MYNN-EDMF/module_bl_mynnedmf.F90:7688/7691"),
    ("mynn_xl_blend_chi_den", "t0c - tice", "MYNN-EDMF/module_bl_mynnedmf.F90:7693 chi=(t0c-t)/(t0c-tice)"),
    ("mynn_q1_rh_den", "one-rhcrit", "MYNN-EDMF/module_bl_mynnedmf.F90:3669/3677 q1_rh=-3.+3.*(rh-rhcrit)/(one-rhcrit)"),
    ("mynn_level2_one_m_c2", "one -c2", "MYNN-EDMF/module_bl_mynnedmf.F90:1555 f1 ... *( one -c2 )"),
    ("mynn_level2_f2", "b1*( g1+g2 ) -3.0*a1*( one -c2 )", "MYNN-EDMF/module_bl_mynnedmf.F90:1557 f2"),
    ("mynn_level2_b1_g1_m_c1", "b1*( g1-c1 )", "MYNN-EDMF/module_bl_mynnedmf.F90:1555/1558"),
    ("mynn_level2_2a1_3m2c2", "2.0*a1*(3.0-2.0*c2)", "MYNN-EDMF/module_bl_mynnedmf.F90:1555 (tail term, folded)"),
    ("mynn_level2_3a2", "3.0*a2", "MYNN-EDMF/module_bl_mynnedmf.F90:1555 (prefix)"),
    ("mynn_level2_one_m_c5", "one-c5", "MYNN-EDMF/module_bl_mynnedmf.F90:1555"),
    ("mynn_psi_dydz_m", "-0.33333*am_unst", "MYNN-EDMF/module_bl_mynnedmf.F90:7730 dummy_11=-0.33333*am_unst*..."),
    ("mynn_psi_dydz_h", "-0.33333*ah_unst", "MYNN-EDMF/module_bl_mynnedmf.F90:7782"),
    # MYNN surface layer (module_sf_mynn.F)
    ("mynnsl_inv_27315", "1./273.15", "module_sf_mynn.F:525/543 E1=SVP1*EXP(4648*(1./273.15-1./TSK)..."),
    ("mynnsl_ep_3", "1.-ep_2", "module_sf_mynn.F:81 ep_3=1.-ep_2"),
    # Noah-MP (module_sf_noahmplsm.F, module_sf_noahmpdrv.F)
    ("noahmp_vkc2", "VKC*VKC", "module_sf_noahmplsm.F:4734/4735 CM=VKC*VKC/(CMFM*CMFM)"),
    ("noahmp_rsurf_den", "( 2.71828 - 1.0 )", "module_sf_noahmplsm.F:2185 L_RSURF ... /( 2.71828 - 1.0 )"),
    ("noahmp_snowh2o_ratio", "max_liq_mass_fraction/(1.0 - max_liq_mass_fraction)", "module_sf_noahmplsm.F:7216/7217"),
    ("noahmp_frzfact_tail", "(0.412 / 0.468)", "module_sf_noahmpdrv.F:1705 FRZFACT=(SMCMAX/SMCREF)*(0.412/0.468)"),
    # KF (module_cu_kfeta.F)
    ("kf_rhh_slope", "0.2/1000.", "module_cu_kfeta.F:1743 RHH=1.-0.2/1000.*(Z0(KSTART)-Z0(ND))"),
    ("kf_cliq", "SVP2*SVPT0", "module_cu_kfeta.F:732/3212 CLIQ=SVP2*SVPT0"),
    ("kf_cliq_m_bliq_dliq", "(SVP2*SVPT0)-SVP2*SVP3", "module_cu_kfeta.F:1748/2111/2841 (CLIQ-BLIQ*DLIQ), runtime REAL"),
    ("kf_rdpr", "1./((110000.0-5000.)/REAL(220-1))", "module_cu_kfeta.F:3208-3225 DPR=(PBOT-PLUTOP)/REAL(KFNP-1); rdpr=1./dpr"),
    # RRTMG SW (module_ra_rrtmg_sw.F, rb = REAL)
    ("rrtmg_sw_bpade", "1.0 / 0.278", "module_ra_rrtmg_sw.F:1229/4755 bpade=1.0_rb/pade, pade=0.278_rb"),
    # Thompson effective radius / size distributions (module_mp_thompson.F)
    ("thompson_oams", "1./am_s", "module_mp_thompson.F:747 oams=1./am_s"),
    ("thompson_lami_prefix", "am_i*cig(2)*oig1", "module_mp_thompson.F:1860/2647/2711/3038/3679 lami=(am_i*cig(2)*oig1*ni/ri)**obmi"),
    ("thompson_lamr_prefix", "am_r*crg(3)*org2", "module_mp_thompson.F:1888/2147/3075/3240/3385/3617"),
    ("thompson_lamg_prefix", "am_g(idx_bg1)*cgg(3,1)*ogg2", "module_mp_thompson.F:1929/2136/3126/3374"),
    ("thompson_t1_vts_prefix", "Kap0*csg(4)", "module_mp_thompson.F:3715 t1_vts=Kap0*csg(4)*ils1**cse(4)"),
    ("thompson_4218_olfus", "4218.*olfus", "module_mp_thompson.F:2787 prr_sml+4218.*olfus*(twet-T_0)"),
    # advance_w dampmag = dts*config_flags%dampcoef (module_small_step_em.F:1284), REAL run-time product; release
    # dts_rk values: Swiss dt 18 -> 6 / 4.5, PROD d01 dt 54 -> 18 / 13.5, d02 dt 18 -> 6 / 4.5; dampcoef 0.2.
    ("acoustic_dampmag_dts4p5", "dts_a*dampcoef_a", "dyn_em/module_small_step_em.F:1284 (dts=4.5, dampcoef=0.2)"),
    ("acoustic_dampmag_dts18", "dts_b*dampcoef_a", "dyn_em/module_small_step_em.F:1284 (dts=18., dampcoef=0.2)"),
    ("acoustic_dampmag_dts13p5", "dts_c*dampcoef_a", "dyn_em/module_small_step_em.F:1284 (dts=13.5, dampcoef=0.2; equal control)"),
    # Noah-MP CLASS snow albedo EXP argument and snow water release, REAL run-time expressions of DT
    # (module_sf_noahmplsm.F:3261 / :7211), release dt: Swiss 18, PROD d01 54 / d02 18, WN3 d03 6.
    ("noahmp_class_alb_arg_dt54", "-0.01*dt_54/3600.0", "module_sf_noahmplsm.F:3261 EXP(-0.01*DT/3600.0), DT=54"),
    ("noahmp_class_alb_arg_dt18", "-0.01*dt_18/3600.0", "module_sf_noahmplsm.F:3261, DT=18 (equal control)"),
    ("noahmp_snow_ret_dt54", "snow_ret_fac*dt_54", "module_sf_noahmplsm.F:7211 SNOW_RET_FAC*DT, DT=54"),
    ("noahmp_snow_ret_dt6", "snow_ret_fac*dt_6", "module_sf_noahmplsm.F:7211 SNOW_RET_FAC*DT, DT=6"),
    ("noahmp_snow_ret_dt18", "snow_ret_fac*dt_18", "module_sf_noahmplsm.F:7211 SNOW_RET_FAC*DT, DT=18 (equal control)"),
]

PRELUDE = """
      USE wg_probe
      USE module_model_constants, ONLY: cpv_m => cpv, cliq_m => cliq, cice_m => cice, ep_2, SVP2, SVPT0, SVP3
      USE module_bl_mynnedmf_common, ONLY: t0c, tice, one, cpv, cliq, cice
      IMPLICIT NONE
      ! module_bl_mynnedmf.F90:278-301 (module PARAMETERs, verbatim)
      real, parameter :: pr = 0.74, g1 = 0.235, b1 = 24.0, b2 = 15.0, c2 = 0.729, c3 = 0.340, c4 = 0.0, c5 = 0.2
      real, parameter :: a1 = b1*( 1.0-3.0*g1 )/6.0
      real, parameter :: c1 = g1 -1.0/( 3.0*a1*2.88449914061481660)
      real, parameter :: a2 = a1*( g1-c1 )/( g1*pr )
      real, parameter :: g2 = b2/b1*( 1.0-c3 ) +2.0*a1/b1*( 3.0-2.0*c2 )
      real, parameter :: rhcrit = 0.83                  ! :3431
      real, parameter :: am_unst = 10., ah_unst = 34.   ! :7714
      real, parameter :: VKC = 0.40                     ! module_sf_noahmplsm.F:206
      real, parameter :: max_liq_mass_fraction = 0.4    ! module_sf_noahmplsm.F:7135
      ! module_mp_thompson.F PRIVATE parameters (verbatim)
      REAL, PARAMETER:: PI = 3.1415926536, rho_w = 1000.0, rho_i = 890.0
      REAL, DIMENSION(9), PARAMETER:: rho_g = (/50., 100., 200., 300., 400., 500., 600., 700., 800./)
      REAL, PARAMETER:: am_r = PI*rho_w/6.0, am_i = PI*rho_i/6.0, am_s = 0.069
      INTEGER, PARAMETER:: idx_bg1 = 5
      REAL, PARAMETER:: am_g5 = PI*rho_g(idx_bg1)/6.0
      REAL, PARAMETER:: mu_r = 0.0, mu_g = 0.0, mu_i = 0.0, bm_r = 3.0, bm_g = 3.0, bm_i = 3.0, bm_s = 2.0
      REAL, PARAMETER:: bv_s = 0.55, bv_i = 1.0, Kap0 = 490.6
      REAL, PARAMETER:: lsub = 2.834E6, lvap0 = 2.5E6, lfus = lsub - lvap0, olfus = 1./lfus
      REAL:: x
      REAL:: dts_a = 4.5, dts_b = 18., dts_c = 13.5, dampcoef_a = 0.2   ! namelist REAL variables
      REAL:: dt_54 = 54., dt_18 = 18., dt_6 = 6., snow_ret_fac = 5.e-5  ! DT; MPTABLE SNOW_RET_FAC
      REAL:: am_g(9), cie(7), cig(7), cre(13), crg(13), cge(12), cgg(12,1), cse(18), csg(18), oig1, org2, ogg2
      INTEGER:: n
      am_g(idx_bg1) = am_g5
      cie(1) = mu_i + 1.; cie(2) = bm_i + mu_i + 1.
      do n = 1, 2
         cig(n) = WGAMMA(cie(n))
      enddo
      oig1 = 1./cig(1)
      cre(2) = mu_r + 1.; cre(3) = bm_r + mu_r + 1.
      do n = 2, 3
         crg(n) = WGAMMA(cre(n))
      enddo
      org2 = 1./crg(2)
      cge(2) = mu_g + 1.; cge(3) = bm_g + mu_g + 1.
      cgg(2,1) = WGAMMA(cge(2)); cgg(3,1) = WGAMMA(cge(3))
      ogg2 = 1./cgg(2,1)
      cse(4) = bm_s + bv_s + 1.                          ! :730
      csg(4) = WGAMMA(cse(4))                            ! :745
"""


def main():
    out_json, work = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    wg = (ROOT / "proofs/thompson/wrf_constants/wrf_constants.F90").read_text()
    module = wg[wg.index("      MODULE wg_probe"): wg.index("      END MODULE wg_probe") + len("      END MODULE wg_probe")]
    lines = [module, "      PROGRAM residual", PRELUDE]
    for idx, (label, expr, _ref) in enumerate(ENTRIES):
        lines.append(f"      x = {expr}")
        lines.append(f"      write(*,'(I0,1X,I0)') {idx}, transfer(x, 1)")
    lines.append("      END PROGRAM residual")
    src = "\n".join(lines) + "\n"
    f90 = work / "residual.f90"
    f90.write_text(src)
    inc = [f"-I{WRF / d}" for d in ("frame", "phys", "share", "main")]
    # module_bl_mynnedmf_common needs its object (no executable code inside, parameters only) -> link not required
    subprocess.run([FC, "-ffree-line-length-none", "-O2", *inc, "-o", str(work / "residual"), str(f90)],
                   check=True, cwd=work)
    out = subprocess.run([str(work / "residual")], check=True, capture_output=True, text=True).stdout
    res = {}
    for line in out.split("\n"):
        if not line.strip():
            continue
        idx, bits = (int(v) for v in line.split())
        bits &= 0xFFFFFFFF
        label, expr, ref = ENTRIES[idx]
        res[label] = {"bits": bits, "value": struct.unpack("<f", struct.pack("<I", bits))[0], "expr": expr, "wrf": ref}
    import hashlib
    res["_meta"] = {"compiler": FC, "wrf": str(WRF), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
                    "program": str(f90)}
    out_json.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if k != "_meta":
            print(f"{k:28s} {v['value']!r:>24} 0x{v['bits']:08x}")


if __name__ == "__main__":
    main()
