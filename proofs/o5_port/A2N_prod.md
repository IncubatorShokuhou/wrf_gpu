| target | kernels | PASS | FAIL | TIMEOUT | other | spill/scratch kernels | max regs (VGPR) | max compile s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda:70 | 61 | 61 | 0 | 0 | 0 | 3 | 255 | 226 |
| cuda:75 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 230 |
| cuda:80 | 61 | 61 | 0 | 0 | 0 | 3 | 255 | 245 |
| cuda:86 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 209 |
| cuda:89 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 285 |
| cuda:90 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 184 |
| cuda:120 | 61 | 61 | 0 | 0 | 0 | 2 | 255 | 230 |

| family | kernel (src) | TTIR chars | cuda:70 | cuda:75 | cuda:80 | cuda:86 | cuda:89 | cuda:90 | cuda:120 |
|---|---|---:|---|---|---|---|---|---|---|
| GWDO | `kernels/phys_gwdo_column.py:282` gwdo_column_fp32 | 138119 | 56r | 64r | 56r | 64r | 64r | 56r | 64r |
| KF | `kernels/phys_kf_column.py:1273` kf_column_fp32 | 681863 | 246r | 240r | 241r | 236r | 236r | 237r | 242r |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33955 | 31r | 31r | 32r | 32r | 32r | 32r | 36r |
| MYNN | `kernels/phys_mynn_cloudmix.py:59` _lambda_ | 27268 | 48r | 64r | 48r | 48r | 48r | 40r | 42r |
| MYNN | `kernels/phys_mynn_plume.py:406` _lambda_ | 107174 | 80r | 72r | 72r | 64r | 64r | 72r | 71r |
| MYNN | `kernels/phys_mynn_tridiag.py:31` mynn_tridiag2_real | 293712 | 104r | 104r | 104r | 103r | 103r | 128r | 104r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_soilwater | 266152 | 216r | 216r | 216r | 217r | 217r | 217r | 158r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_snowwater | 448911 | 96r | 96r | 96r | 111r | 111r | 110r | 104r |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_canopy_flux | 450976 | 209r | 209r | 208r | 208r | 208r | 208r | 201r |
| Noah-MP | `kernels/phys_noahmp_radiation.py:28` b_noahmp_radiation_real | 183931 | 64r | 64r | 64r | 70r | 70r | 70r | 64r |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311272 | 243r | 243r | 244r | 244r | 244r | 242r | 168r |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326094 | 212r | 209r | 209r | 221r | 221r | 225r | 164r |
| RK/EOS | `kernels/dyn_rhsph_fp32.py:19` b_diff_rhs_ph_fp32 | 393963 | 115r | 117r | 116r | 115r | 115r | 124r | 154r |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53047 | 40r | 48r | 40r | 40r | 40r | 40r | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53135 | 40r | 56r | 40r | 40r | 40r | 48r | 48r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_theta | 17008 | 32r | 36r | 32r | 32r | 32r | 32r | 36r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_u | 18196 | 32r | 36r | 32r | 34r | 34r | 32r | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_w | 18196 | 32r | 36r | 32r | 34r | 34r | 32r | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_v | 18488 | 32r | 37r | 32r | 34r | 34r | 32r | 38r |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 36554 | 32r | 39r | 32r | 38r | 38r | 32r | 40r |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 46719 | 46r | 48r | 46r | 46r | 46r | 46r | 46r |
| RRTMG | `kernels/rad_lw_transfer.py:396` rrtmg_lw_band_sums | 124550 | 64r | 64r | 56r | 56r | 56r | 56r | 56r |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 32r | 56r | 32r | 40r | 40r | 32r | 40r |
| RRTMG | `kernels/rad_mcica.py:167` mcica_kiss_lanes | 75082 | 32r | 56r | 32r | 40r | 40r | 32r | 40r |
| RRTMG | `kernels/rad_sw_band_sums.py:397` rrtmg_sw_band_sums | 395905 | 96r **16/16B sp** st16 | 96r | 96r **16/16B sp** st16 | 80r | 80r | 92r | 80r |
| Thompson | `kernels/phys_thompson_full.py:320` thompson_full_column | 897049 | 242r | 242r | 242r | 244r | 244r | 246r | 246r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_pressure_fp32 | 65648 | 32r | 34r | 30r | 31r | 31r | 32r | 32r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p1_fp32 | 67588 | 32r | 46r | 32r | 40r | 40r | 32r | 44r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p2_fp32 | 111389 | 40r | 64r | 42r | 42r | 42r | 44r | 44r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_uv_fp32 | 216168 | 39r | 64r | 40r | 42r | 42r | 56r | 44r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_mass_fp32 | 338495 | 161r | 162r | 160r | 160r | 160r | 138r | 130r |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_fp32 | 413406 | 64r st32 | 62r st32 | 62r st32 | 56r st32 | 56r st32 | 56r st32 | 48r st32 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_coef_fp32 | 3514725 | 255r **320/320B sp** st320 | 255r **260/260B sp** st264 | 255r **312/312B sp** st312 | 255r **32/32B sp** st32 | 255r **32/32B sp** st32 | 255r **24/24B sp** st24 | 255r **112/112B sp** st112 |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 752979 | 209r | 209r | 208r | 210r | 210r | 207r | 146r |
| advection | `kernels/dyn_flux_fp32.py:58` b_core_scalar_flux_fp32 | 103491 | 71r | 79r | 72r | 72r | 72r | 101r | 82r |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 696299 | 32r | 30r | 32r | 32r | 32r | 40r | 40r |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 719895 | 32r | 30r | 32r | 32r | 32r | 40r | 40r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_m | 50874 | 40r | 70r | 40r | 40r | 40r | 56r | 64r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_w | 51328 | 34r | 64r | 32r | 40r | 40r | 40r | 58r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_u | 57434 | 40r | 64r | 40r | 40r | 40r | 40r | 66r |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_v | 57588 | 34r | 64r | 32r | 40r | 40r | 42r | 78r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_m | 66250 | 34r | 64r | 32r | 40r | 40r | 64r | 56r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_v | 75550 | 64r | 64r | 56r | 48r | 48r | 64r | 64r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_u | 79298 | 54r | 64r | 48r | 48r | 48r | 56r | 64r |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_w | 82210 | 40r | 64r | 40r | 40r | 40r | 66r | 72r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18769 | 20r | 21r | 20r | 22r | 21r | 21r | 21r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r | 21r | 21r | 22r | 23r | 23r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r | 21r | 21r | 22r | 23r | 23r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18791 | 21r | 21r | 21r | 22r | 23r | 23r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 18797 | 21r | 21r | 21r | 22r | 23r | 23r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23572 | 24r | 23r | 22r | 25r | 23r | 23r | 24r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:232` boundary_ring_update_real | 23624 | 22r | 22r | 22r | 25r | 23r | 23r | 24r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19514 | 22r | 22r | 22r | 24r | 24r | 22r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19553 | 24r | 22r | 23r | 23r | 23r | 23r | 23r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 19553 | 26r | 24r | 24r | 24r | 24r | 25r | 24r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 50940 | 33r | 39r | 34r | 39r | 35r | 35r | 38r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 67649 | 41r | 42r | 42r | 43r | 40r | 41r | 44r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 72089 | 44r | 42r | 41r | 45r | 41r | 41r | 45r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 83468 | 48r | 43r | 48r | 45r | 48r | 48r | 46r |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 83491 | 50r | 44r | 48r | 45r | 44r | 48r | 46r |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:36` boundary_ring_relax_real | 48528 | 40r | 42r | 38r | 37r | 38r | 40r | 43r |
