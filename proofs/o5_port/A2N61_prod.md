| target | kernels | PASS | FAIL | TIMEOUT | other | spill/scratch kernels | max regs (VGPR) | max compile s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda:61 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 271 |

| family | kernel (src) | TTIR chars | cuda:61 |
|---|---|---:|---|
| GWDO | `kernels/phys_gwdo_column.py:282` gwdo_column_fp32 | 138119 | 56r |
| KF | `kernels/phys_kf_column.py:1273` kf_column_fp32 | 681863 | 254r |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33955 | 27r |
| MYNN | `kernels/phys_mynn_cloudmix.py:59` _lambda_ | 27268 | 32r |
| MYNN | `kernels/phys_mynn_plume.py:406` _lambda_ | 107174 | 72r |
| MYNN | `kernels/phys_mynn_tridiag.py:31` mynn_tridiag2_real | 293712 | 128r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_soilwater | 266152 | 203r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_snowwater | 448911 | 128r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_canopy_flux | 450976 | 220r |
| Noah-MP | `kernels/phys_noahmp_radiation.py:28` b_noahmp_radiation_real | 183931 | 64r |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311272 | 225r |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326094 | 216r |
| RK/EOS | `kernels/dyn_rhsph_fp32.py:19` b_diff_rhs_ph_fp32 | 393963 | 128r |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53047 | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53135 | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_theta | 17008 | 32r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_u | 18196 | 32r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_w | 18196 | 32r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_v | 18488 | 32r |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 36554 | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 46719 | 42r |
| RRTMG | `kernels/rad_lw_transfer.py:396` rrtmg_lw_band_sums | 124550 | 64r |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 32r |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 32r |
| RRTMG | `kernels/rad_sw_band_sums.py:397` rrtmg_sw_band_sums | 395905 | 105r |
| Thompson | `kernels/phys_thompson_full.py:320` thompson_full_column | 897049 | 254r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_pressure_fp32 | 65648 | 29r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p1_fp32 | 67588 | 32r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p2_fp32 | 111389 | 48r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_uv_fp32 | 216168 | 56r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_mass_fp32 | 338495 | 168r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_fp32 | 413406 | 64r st32 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_coef_fp32 | 3514725 | 255r **492/492B sp** st496 |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 752979 | 192r |
| advection | `kernels/dyn_flux_fp32.py:58` b_core_scalar_flux_fp32 | 103491 | 71r |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 696299 | 32r |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 719895 | 32r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_m | 50874 | 48r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_w | 51328 | 40r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_u | 57434 | 42r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_v | 57588 | 48r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_m | 66250 | 40r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_v | 75550 | 55r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_u | 79298 | 48r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_w | 82210 | 56r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18769 | 18r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18797 | 21r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23572 | 22r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23624 | 22r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19514 | 26r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19553 | 29r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19553 | 29r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 50940 | 32r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 67649 | 41r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 72089 | 41r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 83468 | 40r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 83491 | 40r |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:36` boundary_ring_relax_real | 48528 | 49r |
