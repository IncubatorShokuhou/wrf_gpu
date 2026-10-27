| target | kernels | PASS | FAIL | TIMEOUT | other | spill/scratch kernels | max regs (VGPR) | max compile s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hip:gfx90a | 61 | 61 | 0 | 0 | 0 | 1 | 366 | 173 |
| hip:gfx942 | 61 | 61 | 0 | 0 | 0 | 1 | 366 | 181 |
| hip:gfx1100 | 61 | 61 | 0 | 0 | 0 | 1 | 256 | 163 |

| family | kernel (src) | TTIR chars | hip:gfx90a | hip:gfx942 | hip:gfx1100 |
|---|---|---:|---|---|---|
| GWDO | `kernels/phys_gwdo_column.py:282` gwdo_column_fp32 | 138179 | 71v o7 | 77v o6 | 73v o16 |
| KF | `kernels/phys_kf_column.py:1273` kf_column_fp32 | 682293 | 259v 80ssp o1 | 243v 71ssp o2 | 178v 17ssp o8 |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33970 | 35v o8 | 35v o8 | 32v o16 |
| MYNN | `kernels/phys_mynn_cloudmix.py:59` _lambda_ | 27268 | 26v o8 | 21v o8 | 46v o16 |
| MYNN | `kernels/phys_mynn_plume.py:406` _lambda_ | 107184 | 79v o6 | 67v o7 | 66v o16 |
| MYNN | `kernels/phys_mynn_tridiag.py:31` mynn_tridiag2_real | 281016 | 105v o4 | 104v o4 | 96v o16 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_soilwater | 266452 | 180v o2 | 176v o2 | 164v o9 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_snowwater | 448956 | 200v 84ssp o2 | 200v 78ssp o2 | 207v 2ssp o7 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_canopy_flux | 451596 | 234v o2 | 210v o2 | 178v o8 |
| Noah-MP | `kernels/phys_noahmp_radiation.py:28` b_noahmp_radiation_real | 184106 | 88v o5 | 88v o5 | 76v o16 |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311302 | 80v o6 | 82v o5 | 155v o9 |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326124 | 88v o5 | 90v 2ssp o5 | 168v o9 |
| RK/EOS | `kernels/dyn_rhsph_fp32.py:19` b_diff_rhs_ph_fp32 | 393963 | 96v o5 | 96v o5 | 191v o8 |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53019 | 15v o8 | 14v o8 | 23v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53105 | 14v o8 | 14v o8 | 23v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_theta | 16994 | 10v o8 | 10v o8 | 17v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_u | 18042 | 10v o8 | 10v o8 | 20v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_w | 18042 | 10v o8 | 10v o8 | 20v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_v | 18332 | 10v o8 | 10v o8 | 18v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 36103 | 21v o8 | 21v o8 | 36v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 45696 | 24v o8 | 24v o8 | 55v o16 |
| RRTMG | `kernels/rad_lw_transfer.py:396` rrtmg_lw_band_sums | 124620 | 64v o8 | 62v o8 | 72v o16 |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 44v o8 | 57v o8 | 79v o16 |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 44v o8 | 57v o8 | 79v o16 |
| RRTMG | `kernels/rad_sw_band_sums.py:397` rrtmg_sw_band_sums | 396140 | 132v o3 | 128v o4 | 131v o10 |
| Thompson | `kernels/phys_thompson_full.py:320` thompson_full_column | 898096 | 270v **2vsp** 34ssp o1 | 269v **2vsp** 32ssp o1 | 187v 2ssp o8 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_pressure_fp32 | 65658 | 22v o8 | 22v o8 | 28v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p1_fp32 | 67588 | 32v o8 | 32v o8 | 32v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p2_fp32 | 111404 | 60v o8 | 60v o8 | 61v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_uv_fp32 | 216168 | 70v o7 | 70v o7 | 70v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_mass_fp32 | 338495 | 149v o3 | 149v o3 | 155v o9 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_fp32 | 413426 | 67v o7 | 64v o8 | 55v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_coef_fp32 | 3514725 | 366v o1 | 366v o1 | 256v **112vsp** **452B scr** o5 |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 752979 | 56v o8 | 56v o8 | 114v o12 |
| advection | `kernels/dyn_flux_fp32.py:58` b_core_scalar_flux_fp32 | 103491 | 52v o8 | 52v o8 | 96v o16 |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 695155 | 12v o8 | 12v o8 | 12v o16 |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 718751 | 12v o8 | 12v o8 | 12v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_m | 50874 | 33v o8 | 32v o8 | 55v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_w | 51328 | 32v o8 | 32v o8 | 54v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_u | 57434 | 32v o8 | 33v o8 | 56v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_v | 57588 | 31v o8 | 32v o8 | 59v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_m | 66250 | 34v o8 | 34v o8 | 72v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_v | 75550 | 34v o8 | 34v o8 | 74v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_u | 79298 | 36v o8 | 36v o8 | 75v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_w | 82210 | 40v o8 | 40v o8 | 77v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18488 | 17v o8 | 17v o8 | 14v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18510 | 17v o8 | 17v o8 | 14v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18510 | 17v o8 | 17v o8 | 14v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18510 | 17v o8 | 17v o8 | 14v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18516 | 17v o8 | 17v o8 | 14v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23145 | 16v o8 | 16v o8 | 16v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23197 | 16v o8 | 16v o8 | 16v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19229 | 12v o8 | 11v o8 | 20v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19268 | 12v o8 | 11v o8 | 21v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19268 | 12v o8 | 11v o8 | 21v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 50057 | 23v o8 | 23v o8 | 23v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 66578 | 28v o8 | 28v o8 | 28v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 70988 | 28v o8 | 28v o8 | 28v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 80445 | 28v o8 | 28v o8 | 37v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 80468 | 28v o8 | 28v o8 | 37v o16 |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:36` boundary_ring_relax_real | 48472 | 22v o8 | 22v o8 | 21v o16 |
