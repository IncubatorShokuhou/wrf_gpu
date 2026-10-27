| target | kernels | PASS | FAIL | TIMEOUT | other | spill/scratch kernels | max regs (VGPR) | max compile s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda:61 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 137 |
| cuda:70 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 134 |
| cuda:75 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 135 |
| cuda:80 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 136 |
| cuda:86 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 137 |
| cuda:89 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 136 |
| cuda:90 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 132 |
| cuda:120 | 62 | 62 | 0 | 0 | 0 | 2 | 255 | 127 |
| hip:gfx90a | 62 | 62 | 0 | 0 | 0 | 1 | 366 | 103 |
| hip:gfx942 | 62 | 62 | 0 | 0 | 0 | 1 | 366 | 103 |
| hip:gfx1100 | 62 | 62 | 0 | 0 | 0 | 1 | 256 | 102 |

| family | kernel (src) | TTIR chars | cuda:61 | cuda:70 | cuda:75 | cuda:80 | cuda:86 | cuda:89 | cuda:90 | cuda:120 | hip:gfx90a | hip:gfx942 | hip:gfx1100 |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33970 | 27r | 31r | 31r | 32r | 32r | 32r | 32r | 36r | – | – | – |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33976 | 27r | 31r | 31r | 32r | 32r | 32r | 32r | 36r | – | – | – |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33985 | – | – | – | – | – | – | – | – | 35v o8 | 35v o8 | 32v o16 |
| MYNN | `kernels/phys_mynn_boulac.py:107` _lambda_ | 33991 | – | – | – | – | – | – | – | – | 35v o8 | 35v o8 | 32v o16 |
| MYNN | `kernels/phys_mynn_cloudmix.py:59` _lambda_ | 27301 | 32r | 48r | 64r | 48r | 48r | 48r | 40r | 42r | – | – | – |
| MYNN | `kernels/phys_mynn_cloudmix.py:59` _lambda_ | 27301 | – | – | – | – | – | – | – | – | 26v o8 | 21v o8 | 46v o16 |
| MYNN | `kernels/phys_mynn_plume.py:406` _lambda_ | 107388 | 72r | 80r | 72r | 72r | 64r | 64r | 72r | 71r | – | – | – |
| MYNN | `kernels/phys_mynn_plume.py:406` _lambda_ | 107398 | – | – | – | – | – | – | – | – | 79v o6 | 67v o7 | 66v o16 |
| MYNN | `kernels/phys_mynn_tridiag.py:31` mynn_tridiag2_real | 293733 | 128r | 104r | 104r | 104r | 103r | 103r | 128r | 104r | – | – | – |
| MYNN | `kernels/phys_mynn_tridiag.py:31` mynn_tridiag2_real | 281037 | – | – | – | – | – | – | – | – | 105v o4 | 104v o4 | 96v o16 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_soilwater | 266365 | 203r | 216r | 216r | 216r | 217r | 217r | 217r | 158r | – | – | – |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_snowwater | 449107 | 128r | 96r | 96r | 96r | 111r | 111r | 110r | 104r | – | – | – |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_canopy_flux | 451330 | 220r | 209r | 209r | 208r | 208r | 208r | 208r | 201r | – | – | – |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_soilwater | 266665 | – | – | – | – | – | – | – | – | 180v o2 | 176v o2 | 164v o9 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_snowwater | 449152 | – | – | – | – | – | – | – | – | 200v 84ssp o2 | 200v 78ssp o2 | 207v 2ssp o7 |
| Noah-MP | `kernels/phys_noahmp_columns.py:107` b_core_noah_canopy_flux | 451950 | – | – | – | – | – | – | – | – | 234v o2 | 210v o2 | 178v o8 |
| Noah-MP | `kernels/phys_noahmp_radiation.py:28` b_noahmp_radiation_real | 184117 | 64r | 64r | 64r | 64r | 70r | 70r | 70r | 64r | – | – | – |
| Noah-MP | `kernels/phys_noahmp_radiation.py:28` b_noahmp_radiation_real | 184292 | – | – | – | – | – | – | – | – | 88v o5 | 88v o5 | 76v o16 |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311822 | 225r | 243r | 243r | 244r | 244r | 244r | 242r | 168r | – | – | – |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311832 | 232r | 245r | 242r | 244r | 244r | 244r | 246r | 168r | – | – | – |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311852 | – | – | – | – | – | – | – | – | 80v o6 | 82v o5 | 155v o9 |
| PD | `kernels/dyn_pd_fp32.py:327` b_core_pd_scale_sl_fp32 | 311862 | – | – | – | – | – | – | – | – | 90v o5 | 91v o5 | 167v o9 |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326694 | 216r | 212r | 209r | 209r | 221r | 221r | 225r | 164r | – | – | – |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326704 | 229r | 236r | 237r | 233r | 221r | 221r | 228r | 167r | – | – | – |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326724 | – | – | – | – | – | – | – | – | 88v o5 | 90v 2ssp o5 | 168v o9 |
| PD | `kernels/dyn_pd_fp32.py:358` b_core_pd_tend_sl_fp32 | 326734 | – | – | – | – | – | – | – | – | 98v o4 | 98v o4 | 192v o8 |
| RK/EOS | `kernels/dyn_rhsph_fp32.py:19` b_diff_rhs_ph_fp32 | 394803 | 128r | 115r | 117r | 117r | 115r | 115r | 124r | 154r | – | – | – |
| RK/EOS | `kernels/dyn_rhsph_fp32.py:19` b_diff_rhs_ph_fp32 | 394803 | – | – | – | – | – | – | – | – | 96v o5 | 96v o5 | 191v o8 |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53152 | 40r | 40r | 48r | 40r | 40r | 40r | 40r | 40r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53240 | 40r | 40r | 56r | 40r | 40r | 40r | 48r | 48r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53124 | – | – | – | – | – | – | – | – | 15v o8 | 14v o8 | 23v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:153` b_diff_stage_masses | 53210 | – | – | – | – | – | – | – | – | 14v o8 | 14v o8 | 23v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_theta | 17044 | 32r | 32r | 36r | 32r | 32r | 32r | 32r | 36r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_u | 18232 | 32r | 32r | 36r | 32r | 34r | 34r | 32r | 40r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_w | 18232 | 32r | 32r | 36r | 32r | 36r | 36r | 32r | 40r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_v | 18524 | 32r | 32r | 37r | 32r | 34r | 34r | 32r | 38r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_theta | 17030 | – | – | – | – | – | – | – | – | 10v o8 | 10v o8 | 17v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_u | 18078 | – | – | – | – | – | – | – | – | 10v o8 | 10v o8 | 20v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_w | 18078 | – | – | – | – | – | – | – | – | 10v o8 | 10v o8 | 20v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:201` b_diff_coupled_work_v | 18368 | – | – | – | – | – | – | – | – | 10v o8 | 10v o8 | 18v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 36617 | 40r | 32r | 39r | 32r | 38r | 38r | 32r | 40r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 46788 | 42r | 44r | 48r | 44r | 45r | 45r | 46r | 46r | – | – | – |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 36166 | – | – | – | – | – | – | – | – | 21v o8 | 21v o8 | 36v o16 |
| RK/EOS | `kernels/dyn_rk_fp32.py:78` b_diff_eos_real | 45765 | – | – | – | – | – | – | – | – | 24v o8 | 24v o8 | 55v o16 |
| RK/EOS | `kernels/dyn_uvtend_fp32.py:37` b_diff_large_step_u_fp32 | 189110 | 108r | 80r | 90r | 80r | 96r | 96r | 110r | 110r | – | – | – |
| RK/EOS | `kernels/dyn_uvtend_fp32.py:37` b_diff_large_step_u_fp32 | 189110 | – | – | – | – | – | – | – | – | 88v o5 | 91v o5 | 181v o8 |
| RK/EOS | `kernels/dyn_uvtend_fp32.py:98` b_diff_large_step_v_fp32 | 189375 | 116r | 92r | 92r | 94r | 80r | 80r | 112r | 108r | – | – | – |
| RK/EOS | `kernels/dyn_uvtend_fp32.py:98` b_diff_large_step_v_fp32 | 189375 | – | – | – | – | – | – | – | – | 85v o5 | 85v o5 | 178v o8 |
| Thompson | `kernels/phys_thompson_full.py:320` thompson_full_column | 897055 | 254r | 242r | 242r | 242r | 244r | 244r | 246r | 246r | – | – | – |
| Thompson | `kernels/phys_thompson_full.py:320` thompson_full_column | 898102 | – | – | – | – | – | – | – | – | 270v **2vsp** 34ssp o1 | 269v **2vsp** 32ssp o1 | 187v 2ssp o8 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_pressure_fp32 | 65105 | 29r | 32r | 34r | 30r | 30r | 30r | 32r | 32r | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p1_fp32 | 67744 | 32r | 32r | 46r | 32r | 40r | 40r | 32r | 44r | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p2_fp32 | 111596 | 48r | 40r | 64r | 42r | 42r | 42r | 40r | 42r | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_uv_fp32 | 215814 | 56r | 40r | 64r | 40r | 40r | 40r | 56r | 42r | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_mass_fp32 | 338296 | 168r | 161r | 162r | 160r | 160r | 160r | 138r | 130r | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_fp32 | 413859 | 64r st32 | 64r st32 | 62r st32 | 62r st32 | 56r st32 | 56r st32 | 56r st32 | 48r st32 | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_coef_fp32 | 3517890 | 255r **492/492B sp** st496 | 255r **320/320B sp** st320 | 255r **260/260B sp** st264 | 255r **312/312B sp** st312 | 255r **32/32B sp** st32 | 255r **32/32B sp** st32 | 255r **24/24B sp** st24 | 255r **112/112B sp** st112 | – | – | – |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_pressure_fp32 | 65115 | – | – | – | – | – | – | – | – | 22v o8 | 22v o8 | 28v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p1_fp32 | 67744 | – | – | – | – | – | – | – | – | 32v o8 | 32v o8 | 32v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_p2_fp32 | 111611 | – | – | – | – | – | – | – | – | 60v o8 | 60v o8 | 61v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_uv_fp32 | 215814 | – | – | – | – | – | – | – | – | 70v o7 | 70v o7 | 70v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_mass_fp32 | 338296 | – | – | – | – | – | – | – | – | 149v o3 | 149v o3 | 155v o9 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_w_fp32 | 413879 | – | – | – | – | – | – | – | – | 67v o7 | 64v o8 | 55v o16 |
| acoustic | `kernels/dyn_acoustic_fp32.py:92` b_core_coef_fp32 | 3517890 | – | – | – | – | – | – | – | – | 366v o1 | 366v o1 | 256v **112vsp** **452B scr** o5 |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 204487 | 152r | 157r | 157r | 158r | 148r | 148r | 152r | 124r | – | – | – |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 570231 | 189r | 207r | 207r | 208r | 168r | 168r | 210r | 146r | – | – | – |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 204487 | – | – | – | – | – | – | – | – | 54v o8 | 54v o8 | 114v o12 |
| advection | `kernels/dyn_flux_fp32.py:139` b_core_scalar_flux_fp32_stac | 570231 | – | – | – | – | – | – | – | – | 56v o8 | 56v o8 | 114v o12 |
| advection | `kernels/dyn_flux_fp32.py:58` b_core_scalar_flux_fp32 | 103666 | 71r | 71r | 79r | 72r | 72r | 72r | 101r | 82r | – | – | – |
| advection | `kernels/dyn_flux_fp32.py:58` b_core_scalar_flux_fp32 | 103666 | – | – | – | – | – | – | – | – | 52v o8 | 52v o8 | 96v o16 |
| advection | `kernels/dyn_momflux_fp32.py:107` b_diff_advect_v_fp32 | 138943 | 117r | 94r | 94r | 93r | 93r | 93r | 107r | 103r | – | – | – |
| advection | `kernels/dyn_momflux_fp32.py:107` b_diff_advect_v_fp32 | 138943 | – | – | – | – | – | – | – | – | 56v o8 | 56v o8 | 109v o12 |
| advection | `kernels/dyn_momflux_fp32.py:77` b_diff_advect_u_fp32 | 137449 | 113r | 103r | 103r | 103r | 108r | 108r | 112r | 113r | – | – | – |
| advection | `kernels/dyn_momflux_fp32.py:77` b_diff_advect_u_fp32 | 137449 | – | – | – | – | – | – | – | – | 54v o8 | 54v o8 | 115v o12 |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 697388 | 32r | 32r | 30r | 32r | 32r | 32r | 40r | 40r | – | – | – |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 720984 | 32r | 32r | 30r | 32r | 32r | 32r | 40r | 40r | – | – | – |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 696244 | – | – | – | – | – | – | – | – | 12v o8 | 12v o8 | 12v o16 |
| coupling | `kernels/phys_hydrostatic_real.py:19` phy_prep_hydrostatic_real | 719840 | – | – | – | – | – | – | – | – | 12v o8 | 12v o8 | 12v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_m | 50961 | 48r | 40r | 64r | 40r | 40r | 40r | 56r | 64r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_w | 51415 | 40r | 34r | 64r | 32r | 40r | 40r | 40r | 58r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_u | 57530 | 42r | 40r | 64r | 40r | 40r | 40r | 40r | 66r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_v | 57684 | 48r | 32r | 64r | 32r | 40r | 40r | 42r | 78r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_m | 50961 | – | – | – | – | – | – | – | – | 33v o8 | 32v o8 | 55v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_w | 51415 | – | – | – | – | – | – | – | – | 32v o8 | 32v o8 | 54v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_u | 57530 | – | – | – | – | – | – | – | – | 32v o8 | 33v o8 | 56v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:15` b_diff_sixth_v | 57684 | – | – | – | – | – | – | – | – | 31v o8 | 32v o8 | 59v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_m | 66358 | 40r | 34r | 64r | 32r | 40r | 40r | 64r | 56r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_v | 75664 | 55r | 56r | 64r | 56r | 48r | 48r | 64r | 64r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_u | 79418 | 48r | 54r | 64r | 48r | 48r | 48r | 56r | 64r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_w | 82342 | 56r | 40r | 64r | 40r | 40r | 40r | 66r | 72r | – | – | – |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_m | 66358 | – | – | – | – | – | – | – | – | 34v o8 | 34v o8 | 72v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_v | 75664 | – | – | – | – | – | – | – | – | 34v o8 | 34v o8 | 74v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_u | 79418 | – | – | – | – | – | – | – | – | 36v o8 | 36v o8 | 75v o16 |
| diffusion | `kernels/dyn_diff_fp32.py:96` b_diff_horizontal_w | 82342 | – | – | – | – | – | – | – | – | 40v o8 | 40v o8 | 77v o16 |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 189767 | 64r | 67r | 67r | 69r | 69r | 69r | 68r | 72r | – | – | – |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 189778 | 64r | 68r | 68r | 64r | 64r | 64r | 64r | 66r | – | – | – |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 189778 | 64r | 67r | 67r | 71r | 70r | 70r | 70r | 70r | – | – | – |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 189778 | 64r | 68r | 68r | 64r | 64r | 64r | 64r | 66r | – | – | – |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 189778 | 64r | 68r | 68r | 64r | 64r | 64r | 64r | 66r | – | – | – |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 187115 | – | – | – | – | – | – | – | – | 37v o8 | 37v o8 | 72v o16 |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 187126 | – | – | – | – | – | – | – | – | 38v o8 | 38v o8 | 73v o16 |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 187126 | – | – | – | – | – | – | – | – | 38v o8 | 38v o8 | 73v o16 |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 187126 | – | – | – | – | – | – | – | – | 38v o8 | 38v o8 | 73v o16 |
| force-down | `nesting/sint_kernel.py:77` nest_sint_sides | 187126 | – | – | – | – | – | – | – | – | 38v o8 | 38v o8 | 73v o16 |
| lateral bdy | `coupling/boundary_apply.py:1577` nest_scalar_couple_real4 | 12813 | 28r | 25r | 27r | 26r | 26r | 26r | 28r | 30r | – | – | – |
| lateral bdy | `coupling/boundary_apply.py:1577` nest_scalar_couple_real4 | 12837 | 28r | 26r | 27r | 26r | 26r | 26r | 28r | 30r | – | – | – |
| lateral bdy | `coupling/boundary_apply.py:1577` nest_scalar_couple_real4 | 12807 | – | – | – | – | – | – | – | – | 8v o8 | 8v o8 | 15v o16 |
| lateral bdy | `coupling/boundary_apply.py:1577` nest_scalar_couple_real4 | 12831 | – | – | – | – | – | – | – | – | 8v o8 | 8v o8 | 15v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 18275 | 23r | 22r | 21r | 22r | 23r | 23r | 21r | 24r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:56` boundary_leaf_interp_real | 17990 | – | – | – | – | – | – | – | – | 11v o8 | 10v o8 | 20v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 49669 | 32r | 35r | 39r | 34r | 41r | 35r | 35r | 38r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 64478 | 41r | 44r | 43r | 41r | 45r | 41r | 40r | 45r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 65958 | 40r | 42r | 44r | 40r | 45r | 42r | 42r | 46r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 78335 | 39r | 43r | 43r | 43r | 43r | 45r | 44r | 46r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 78343 | 39r | 42r | 43r | 42r | 43r | 46r | 43r | 46r | – | – | – |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 48786 | – | – | – | – | – | – | – | – | 22v o8 | 22v o8 | 22v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 63427 | – | – | – | – | – | – | – | – | 28v o8 | 28v o8 | 28v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 64897 | – | – | – | – | – | – | – | – | 28v o8 | 28v o8 | 28v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 76712 | – | – | – | – | – | – | – | – | 27v o8 | 27v o8 | 32v o16 |
| lateral bdy | `kernels/dyn_boundary_fp32.py:93` boundary_dry_relax_real | 76720 | – | – | – | – | – | – | – | – | 27v o8 | 27v o8 | 32v o16 |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:109` boundary_spec_final_real | 15697 | 17r | 22r | 22r | 22r | 22r | 22r | 22r | 24r | – | – | – |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:109` boundary_spec_final_real | 15555 | – | – | – | – | – | – | – | – | 10v o8 | 9v o8 | 9v o16 |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:36` boundary_ring_relax_real | 48583 | 49r | 40r | 41r | 38r | 38r | 37r | 42r | 42r | – | – | – |
| lateral bdy | `kernels/dyn_ring_relax_fp32.py:36` boundary_ring_relax_real | 48527 | – | – | – | – | – | – | – | – | 22v o8 | 22v o8 | 21v o16 |
