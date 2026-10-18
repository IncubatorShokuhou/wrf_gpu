# BEP+BEM rooftop PV and green roofs — pristine WRF oracle (lane w41-writergaps)

v0.4.0 wrote WRF's inactive-branch values for `EP_PV_URB3D`, `QGR_URB4D`, `TGR_URB4D` and
`DRAINGR_URB3D` and refused `PV_FRAC_ROOF>0` / `GR_FLAG=1`. v0.4.1 ports both WRF branches.

**Port** (`physics/urban_bem_building.py`, `physics/urban_green_roof.py`, `physics/urban_bem_model.py`):

| WRF routine | Port |
| --- | --- |
| `radfluxspv`, `hsfluxpv` (module_sf_bem.F) | `roof_radiation_pv`, `pv_panel_step` |
| `upward_rad` PV/green terms, `buildings()` PV share and green blend (module_sf_bep_bem.F) | `_upward`, flux loop of `bem_column_step` |
| `roof_temp_veg`, `soil_temp_veg`, `soil_moist`, `flux_flat_roof` | `urban_green_roof.py` |
| `wall_gr` (module_sf_bem.F) | `roof_wall_gr` |
| bep_bem `EP_PV/QGR/TGR/DRAINGR` sums, `urban_var_init` seeds | `_green_history`, `initial_extra` |

Literal REAL operation order; gfortran keeps IEEE division by constants and libm `powf` for real
exponents, so constant divisors/real exponents ride an optimization barrier
(`real_literal`): XLA:CPU otherwise rewrites `x/c` to `x*(1/c)` and `x**.5` to `sqrt`
([M] 25,305/200,000 and 19/40,000 f32 mismatches; XLA `pow` with a barriered operand equals glibc
`powf` on 40,000 operands for exponents 4 and 1/3).

**Oracle** — `pvgr_oracle.F90` is the qualified v040 caller (`proofs/v040/urban_bep_bem/bep_bem_oracle.F90`)
plus history dumps and a constructed `SMOIS_URB=0.3` (green-roof soil seed). `build_oracle.py`
stages the unmodified pristine modules with the v040 flags; `run_oracle.py` regenerates each arm's
`URBPARM.TBL` from the pristine table (only the listed lines change) and saves `savepoints/`.
Arms (six regimes × six dt=6 s steps, REAL4 and REAL8): `ctl_ac_on/off` (extensions off),
`pv04_ac_on`, `pv10_ac_off`, `gr05_ac_on`, `gr10pv03_ac_on`, `gr05irr_ac_off` (IRHO all hours) and
`gr05dt_ac_on`: `urban_var_init` seeds the green-roof soil at `TBLEND`, in equilibrium with the roof, so
`wall_gr`'s coupling flux `hfgr` stays below 1e-11 W/m² over six steps in the other arms (unexercised);
this arm adds a constructed 4 K green-soil contrast after `urban_var_init` (driver env `PVGR_TRV_OFFSET`;
unset reproduces the other fixtures byte for byte) and moves the roof layers by up to 0.95 K;
`gr05t1_ac_on`: pristine `GR_TYPE: 1` (grass: `rsmin` 40, LAI 2, albedo .3; the pristine table holds type 2); and
`gr05dry_ac_on`: constructed `SMOIS_URB=0.04` (driver env `PVGR_SMOIS_URB`), below `flux_flat_roof`'s wilting point
`qw=0.06` in every layer 5–10 at every step, so `gx` clamps to 0, `f4=0` and `rsveg=rsmax` (latent flux ≤ 6.6 vs
194 W/m² wet).
The control arms reproduce the committed v040 `urban3_*` fixtures byte for byte except
`QR_URB4D` (the constructed seed).

**Gates** (`pvgr_parity.py`, `tests/v040/test_w41_urban_pv_green_roof.py`):

- REAL8: every compared field and step within the v040 gate (2e-11+2e-12|ref|) — **0 violations** on all eight
  PV and green-roof arms (54–62 fields × 6 steps each).
- REAL4: the history producers `EP_PV_URB3D`, `T_PV_URB3D`, `QGR_URB3D`, `TGR_URB3D`, `TRV_URB4D`,
  `DRAIN_URB4D` stay inside WRF's own REAL4-vs-REAL8 envelope (2|ref4-ref8|+8eps) at every step.
  The conditioned green-roof fluxes deviate from step 1 because XLA's `exp`/`log10` differ from
  glibc in the last ulp (REAL8 is exact, so the operation structure is); they carry frozen caps
  `SFRV` 0.05 W/m², `LFRV` 0.02 W/m², `QR_URB4D` 3e-6, `DRAINGR` 2e-6 mm. The ordinary-roof
  fluxes keep their v040 ROUND01 qualification; the control arms show the same raw REAL4 class.
- Genuine mutants (REAL8 assertion kills): PV panel sensible share removed; green-roof
  `wall_gr` coupling removed (on the contrast arm; in the TBLEND-seeded arms that mutant survives two steps);
  grass with sedum stomata; dry soil without the `rsmax` cap. (Deleting the `gx<0` clamp alone is not detectable:
  `max(lai*f1*f2*f3*f4,1e-9)` maps a negative `f4` to the same `rsmax`.)
- Default program: PV/GR-off BEM jaxpr identical to the base commit; release identities unchanged.

GPU smoke (`gpu_smoke.sh`, Madrid 42×42 BEP+BEM, `PV_FRAC_ROOF=0.3`, `GR_FLAG=1`/`GR_FRAC_ROOF=1`, 9 h cold CLI
on an RTX 5090): rc 0, 10 hourly files, 0 nonfinite values, `EP_PV_URB3D`/`QGR_URB4D`/`TGR_URB4D`/`DRAINGR_URB3D`
written every frame (`EP_PV_URB3D` 0 until sunrise, domain maximum 0.6/2.5/4.2 W m⁻² at 07/08/09 UTC). This proves finite
execution and written fields only, not fidelity; the values are gated by the oracle above. No CPU-WRF forecast
comparison with these extensions exists yet. E41 (`wave_cubin_gate.py`, rich JIT + d01 AOT): FAIL, the class already
disclosed for the v0.4.0 BEP+BEM composite c57 — one Noah-MP water/BEM/sfclay `broadcast_subtract_transpose` kInput
fusion with STACK 976 B (c57: 968 B) and 1032 HLO instructions (c57: 1015), which the green-roof code joins (182 of
its 904 framed instructions); the other 12 conservative kInput flags are revised-MM5 surface-layer fusions. No
PV/GR-off control was compiled on this case.
`DRAINGR_URB3D` is WRF's per-call green-roof drainage: bep_bem zeroes it every call (module_sf_bep_bem.F:1058) and
sums that call's drainage (:1101), although the Registry describes it as accumulated.
Cost: only PV/GR configurations trace the new code (31 urban columns in the Madrid case).
