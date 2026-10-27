| Family (historical share) | Modules | Vendor-specific today | Verdict | LD | t ratio scenario MI300X / RX 7900 XTX [I] | Device validation (frozen tolerances) |
|---|---|---|---|---:|---|---|
| Acoustic (29.7 %) | `dyn_acoustic_fp32` (7) | none in body; geometry | reuse; geometry sweep (`coef`: gfx942 366 VGPR, occ 1; gfx1100 VGPR spills + 452 B scratch) | 0.5 | 0.6–1.0 / 1.8–2.6 (latency-bound w/mass loops, A54) | D7 pristine acoustic oracle; mass guards 0 |
| MYNN (12.2 %) | `phys_mynn_plume/tridiag/cloudmix/boulac`, `fused_edmf_plume` | tridiag2 uses `dyn_rk._rn` (RN ops) | small change (inherits `vendor_ops.rn`) | 0.5 | 0.5–0.9 / 1.6–2.4 | 48 WRF mean checks; tridiag2 exact vs literal WRF REAL (rule 3) |
| Advection (10 %) | `dyn_flux_fp32`, `dyn_momflux_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | momentum 12/12; pristine REAL4 operators |
| XLA glue/layout/DtoD (≈ 18 %) | XLA | XLA backend | XLA-ROCm; executing-binary scan (R5) | 0.5 | [U] | D6 + integrity |
| PD (6.3 %) | `dyn_pd_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | PD operator gates |
| Lateral boundary (5.2 %) | `dyn_ring_relax_fp32`, `dyn_boundary_fp32`, `boundary_apply` | RN ops; 2 kernels do not lower off CUDA | small change (M2a) | 1.0 | 0.4–0.7 / 1.9–2.6 | B36 ring tests; NumPy scalar-mass gate; D6 rings |
| RK/EOS (3.8 %) | `dyn_rk_fp32`, `dyn_uvtend/rhsph/omega_fp32` | `dyn_rk._rn` (PTX) | small change (M2a) | 0.5 | 0.4–0.7 / 1.9–2.6 | RK/EOS 222 checks; rhs_ph E133 gate |
| Thompson (3.8 %) | `phys_thompson_full/prep/sedimentation` | `tt.gather`; DOUBLE islands | reuse (`tt.gather` compiles) | 0.5 | 0.7–1.3 / 1.2–2.5 (register-bound; f64 cost [U]) | gate-2/L3a/L3b; mixed-phase pristine; adversarial ice/graupel (E154) |
| Diffusion (3.4 %) | `dyn_diff_fp32` | none | reuse | 0.25 | 0.4–0.6 / 1.9–2.4 | 246 REAL field checks |
| Noah-MP / GWDO / coupling (3.0 %) | `phys_noahmp_columns/radiation`, `phys_gwdo_column`, `phys_hydrostatic_real` | hydrostatic uses `dyn_rk._rn` | reuse; small change (hydrostatic); Noah canopy compile time | 0.5 | 0.7–1.3 / 1.5–2.5 | Noah energy/phenology + REAL water rule; GWDO pristine |
| Nest force-down (0.9 %) | `nesting/sint_kernel` | PTX rn/div mode | small change (M2c) | 0.5 | 0.5–1.0 / 1.5–2.5 | SINT exact host-transcription gate |
| RRTMG (radiation root) | `rad_lw_transfer`, `rad_sw_band_sums/quadrature`, `rad_mcica` | `cuda_native()` code-shape branch (KISS kernel, lane clamp) | small change (`triton_native`) | 0.5 | 0.5–1.0 / 1.5–2.5 | LW/SW 18 + clear-sky 4 WRF oracles; KISS bitwise (rule 2) |
| KF (d01) | `phys_kf_column` | `tt.gather` | reuse; geometry (243 VGPR) | 0.5 | 0.8–1.5 / 1.5–3.0 | 5 frozen + 16,800 real + 8,400 deep-active columns |
| Options (19 modules) | see §1.1 | PTX typed memory/asm, cuda-only externs, adapters | per-scheme state (§3.2) | 2.0 | n/a | own component gates; `experimental` until qualified |
