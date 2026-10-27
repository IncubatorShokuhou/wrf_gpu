# W5P01 compile census, per module (appendix to PORT_PLAN)

| Family | Module | Kernels | gfx942 PASS · max VGPR · VGPR/SGPR spills (scratch) · min occ | gfx1100 same | sm_75 PASS · max regs · spill B | sm_61 PASS | gfx90a / gfx950 / gfx1201 PASS |
|---|---|---:|---|---|---|---|---|
| acoustic | `dyn_acoustic_fp32.py` | 7 | 7/7 · 366 · 0/14 · 1 | 7/7 · 256 · 112/0 (452 B) · 5 | 7/7 · 255 · 260 | 6/6 | 5/5 / 5/5 / 5/5 |
| lateral boundary | `dyn_boundary_fp32.py` | 4 | 4/4 · 36 · 0/0 · 8 | 4/4 · 49 · 0/0 · 16 | 4/4 · 49 · 0 | 4/4 | 4/4 / 4/4 / 4/4 |
| diffusion | `dyn_diff_fp32.py` | 2 | 2/2 · 36 · 0/0 · 8 | 2/2 · 75 · 0/0 · 16 | 2/2 · 64 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| advection | `dyn_flux_fp32.py` | 3 | 3/3 · 56 · 0/0 · 8 | 3/3 · 114 · 0/0 · 12 | 3/3 · 203 · 0 | 3/3 | 3/3 / 3/3 / 3/3 |
| advection | `dyn_momflux_fp32.py` | 3 | 3/3 · 62 · 0/0 · 8 | 3/3 · 117 · 0/0 · 12 | 3/3 · 111 · 0 | 3/3 | 3/3 / 3/3 / 3/3 |
| RK/EOS | `dyn_omega_fp32.py` | 1 | 1/1 · 39 · 0/0 · 8 | 1/1 · 43 · 0/0 · 16 | 1/1 · 46 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| PD transport | `dyn_pd_fp32.py` | 4 | 4/4 · 90 · 0/2 · 5 | 4/4 · 168 · 0/0 · 9 | 4/4 · 245 · 0 | 4/4 | 4/4 / 4/4 / 4/4 |
| RK/EOS | `dyn_rhsph_fp32.py` | 1 | 1/1 · 96 · 0/0 · 5 | 1/1 · 191 · 0/0 · 8 | 1/1 · 113 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| RK/EOS | `dyn_rk_fp32.py` | 5 | 5/5 · 32 · 0/0 · 8 | 5/5 · 46 · 0/0 · 16 | 5/5 · 44 · 0 | 5/5 | 5/5 / 5/5 / 5/5 |
| RK/EOS | `dyn_uvtend_fp32.py` | 2 | 2/2 · 92 · 0/0 · 5 | 2/2 · 189 · 0/0 · 8 | 2/2 · 96 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| MYNN | `fused_edmf_plume.py` | 1 | 1/1 · 159 · 0/2 · 3 | 1/1 · 96 · 0/56 · 16 | 1/1 · 116 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| acoustic (legacy advance_w) | `fused_vertical_implicit.py` | 2 | 2/2 · 384 · 0/0 · 1 | 2/2 · 256 · 126/0 (508 B) · 5 | 2/2 · 255 · 32 | 1/1 | 1/1 / 1/1 / 1/1 |
| GWDO | `phys_gwdo_column.py` | 1 | 1/1 · 77 · 0/0 · 6 | 1/1 · 73 · 0/0 · 16 | 1/1 · 64 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| physics coupling | `phys_hydrostatic_real.py` | 1 | 1/1 · 17 · 0/0 · 8 | 1/1 · 14 · 0/0 · 16 | 1/1 · 32 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| KF | `phys_kf_column.py` | 1 | 1/1 · 243 · 0/91 · 2 | 1/1 · 177 · 0/15 · 8 | 1/1 · 238 · 0 | 1/1 | – / – / – |
| MYNN | `phys_mynn_boulac.py` | 1 | 1/1 · 39 · 0/0 · 8 | 1/1 · 36 · 0/0 · 16 | 1/1 · 32 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| MYNN | `phys_mynn_cloudmix.py` | 1 | 1/1 · 27 · 0/0 · 8 | 1/1 · 25 · 0/0 · 16 | 1/1 · 63 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| MYNN | `phys_mynn_plume.py` | 2 | 2/2 · 64 · 0/0 · 8 | 2/2 · 63 · 0/0 · 16 | 2/2 · 69 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| MYNN | `phys_mynn_tridiag.py` | 1 | 1/1 · 109 · 0/0 · 4 | 1/1 · 118 · 0/0 · 12 | 1/1 · 96 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| Noah-MP | `phys_noahmp_columns.py` | 3 | 3/3 · 185 · 0/20 · 2 | 2/3 · 111 · 0/0 · 12 | 2/3 · 206 · 0 | 2/2 | 1/1 / 1/1 / 1/1 |
| Noah-MP | `phys_noahmp_radiation.py` | 1 | 1/1 · 80 · 0/0 · 6 | 1/1 · 96 · 0/0 · 16 | 1/1 · 64 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| Thompson | `phys_thompson_full.py` | 1 | 1/1 · 223 · 0/76 · 2 | 1/1 · 189 · 0/2 · 8 | 1/1 · 242 · 0 | 1/1 | – / – / – |
| Thompson | `phys_thompson_prep.py` | 1 | 1/1 · 120 · 0/0 · 4 | 1/1 · 104 · 0/0 · 12 | 1/1 · 72 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| Thompson | `phys_thompson_sedimentation.py` | 2 | 2/2 · 27 · 0/0 · 8 | 2/2 · 29 · 0/0 · 16 | 2/2 · 36 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| RRTMG | `rad_lw_transfer.py` | 2 | 2/2 · 59 · 0/0 · 8 | 2/2 · 70 · 0/0 · 16 | 2/2 · 64 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| RRTMG | `rad_mcica.py` | 1 | 1/1 · 58 · 0/0 · 8 | 1/1 · 82 · 0/0 · 16 | 1/1 · 56 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| RRTMG | `rad_sw_band_sums.py` | 2 | 2/2 · 135 · 0/32 · 3 | 2/2 · 121 · 0/0 · 10 | 2/2 · 154 · 0 | 2/2 | 2/2 / 2/2 / 2/2 |
| RRTMG | `rad_sw_quadrature.py` | 1 | 1/1 · 25 · 0/0 · 8 | 1/1 · 20 · 0/0 · 16 | 1/1 · 38 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |
| nest force-down | `nesting/sint_kernel.py` | 1 | 1/1 · 90 · 0/0 · 5 | 1/1 · 176 · 0/0 · 8 | 1/1 · 205 · 0 | 1/1 | 1/1 / 1/1 / 1/1 |

Totals (kernel sites PASS/compiled): gfx942: 58/58, gfx1100: 57/58, 75: 57/58, 61: 55/55, gfx90a: 51/51, gfx950: 51/51, gfx1201: 51/51

Non-PASS: `diagnostics/nwp.py` nwp_helicity_column gfx942 FAIL: RuntimeError: PassManager::run failed; `kernels/phys_noahmp_columns.py` b_core_noah_canopy_flux gfx1100 TIMEOUT: ; `kernels/phys_noahmp_columns.py` b_core_noah_canopy_flux 75 TIMEOUT: ; `kernels/phys_p3.py` p3_mp50_column gfx942 TIMEOUT: ; `kernels/phys_p3.py` p3_mp51_column gfx942 TIMEOUT: ; `kernels/phys_p3.py` p3_mp52_column gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `kernels/phys_p3.py` p3_mp53_column gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `kernels/phys_qnse_columns.py` qnse_sfc_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `kernels/phys_qnse_columns.py` qnse_pbl_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `kernels/phys_qnse_columns.py` qnse_mf_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `physics/ferrier/native.py` ferrier_mp5_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
; `physics/ferrier/native.py` ferrier_mp15_columns gfx942 CRASH: error: couldn't allocate output register for constraint 'f'
