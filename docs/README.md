# Docs

Long-form documentation should stay sparse. Prefer contracts, ADRs, sprint reports, and proof objects over narrative summaries.

- [Runtime parameter families](runtime-parameters.md): v0.4.4 excludes diffusion/damping
  overrides (including `c_s`, refused). They may come in a later release after
  GPU qualification; no date or commitment. MYNN
  is experimental: component/short two-domain GPU root PASS, without three-nest
  D6, station or 24–72-hour qualification.
- [Thompson runtime parameters](thompson-parameters.md): research-candidate
  microphysics coefficients such as `nt_c`; GPU qualification remains pending.
- [MYNN budgets](mynn-budgets.md): experimental TKE budget and EDMF plume histories,
  `bl_mynn_edmf=0`; GPU qualification remains pending.
- [Atmospheric warm start](atmospheric-warm-start.md): experimental `gpuwrf cycle
  --native-atmosphere`; GPU qualification remains pending.

- [v0.4.4 candidate notes](../release_notes/RELEASE_NOTES_v0.4.4.md) and
  [support matrix](release/SUPPORT_MATRIX_v0.4.4.md): actual gates and exclusions.
- [Nested terrain shading](topographic_radiation.md): per-nest height/clock
  behavior and original-WRF operator evidence; coupled candidate status is separate.
