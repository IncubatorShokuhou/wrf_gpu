W2 research candidate adds opt-in Noah-MP soil hydraulics (`smcmax`, `dksat`,
`bexp`, `psisat`), canopy heights (`hvt`, `hvb`), Jarvis `rsmin`, and snow energy
parameters (`z0sno`, `snow_emis`) to surface controls. Soil class factors use
1-based ISLTYP; vegetation factors use IVGTYP. Changed porosity rechecks retained
SMOIS/SH2O through the shared input validator. Values of one selected family
reuse its executable; omitting the fields preserves the default program.

The ON-default CPU program has a disclosed constant-folding round-off class:
BEXP produces1–2ULP differences in a few LH/QFX/Q2 cells, PSISAT1ULP differences
in QSFC/LH/QFX. The measured causal control restores all static output bytes
when those same values are captured as constants. OFF identity remains binding.
Changed-value qualification uses pristine original WRF, including frozen/snow
columns. The manager-approved water-residual floor is measured from32 original
REAL one-ULP input members per parameter/column; the .1mm WRF ceiling and all
field/energy bounds stay binding.

Frozen/snow joint-default arrays additionally show a2–6ULP flux class
(QFX max2.842e-14 kg/m2/s, relative3.928e-7, two cells). A separate cold
constantness intervention restores48/48 OFF output bytes, localizing the
cause to the traced operands just as in the warm case.

On a full nested root step (Swiss d01, CPU), the eight soil/canopy/snow controls at their default values change
41 fields in up to 244 land cells each, by at most 1.9e-6 of the field range (Q2B/QSFC, 23 ULP). The large ULP
counts sit on near-zero fluxes: QFX reaches 786 ULP at an absolute difference of 4.55e-12 kg m-2 s-1. The same
values captured as constants remove every difference except 1–2 ULP in frozen top-layer soil water. Canopy
temperatures are bitwise unchanged. Manager ruling 2026-10-25: accepted as this round-off class, with the full
table in `docs/surface_params.md`. The GPU table must stay ≤ 4e-6 of the field range (pre-registered).

Refused combinations (loud errors): soil controls with irrigation (`opt_irr` 1–3)
or tile drainage (`opt_tdrn` 1–2), which keep their own initial soil-parameter
copies; and, on a fresh start, changed SMCMAX/BEXP/PSISAT when any land soil
layer starts frozen (WRF's NOAHMP_INIT would split that frozen water with the
edited table). Restarts and unchanged values are accepted. Frozen-soil columns,
including a wet frozen layer where the supercooled-liquid limit binds and warm
rain on frozen ground, match original WRF for all four soil parameters (field,
water and energy bounds unchanged).

Status before release integration: ALL9 CPU changed-value field/water/energy
gates PASS (486field groups,54water,54energy). Shared effective-cap checker
188tests and checkpoint83cases PASS, default reports BYTE; deletion mutants
are killed. Native E41, GPU warm-root cost/memory, final post-topo composed
identity, RW gates and critic remain pending. See `LSMPARAMS.md` and
`docs/surface_params.md`. Update this note before publishing release notes.
