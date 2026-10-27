# o5-f2neg — same-operand amplitude of F2 post-RK negatives vs pristine WRF

Question (lane contract): the GPU default path has post-RK negative moisture/number
visits (qc/qr/qi/qs/qg/Ni/Nr, E213, D27). Does pristine WRF 4.7.1 produce the same
negative on the same operands (faithful), a smaller one (port bug), or is it unresolved?

Prior work this lane does not repeat:
- `proofs/t_f2neg` — independent-trajectory population comparison (Swiss default + WDM6):
  WRF has the same magnitude classes; residual extremum ratios OPEN because the
  trajectories evolve independently.
- `proofs/s2_ntu` — NTU (mp56) 2-step FAST0 capture: pristine `sixth_order_diffusion`
  reproduces the source-class negatives.

## Mechanism (pristine source)

At the last RK stage `solve_em.F:1934-1955` calls `rk_update_scalar_pd`, which adds the
whole RK1-frozen `moist_tend`/`scalar_tend` (physics from `update_phy_ten`, diff_opt=1
`horizontal_diffusion`, `sixth_order_diffusion` formed with dt/3, QV relax/spec) to
`moist_old` (`module_em.F:1877-1911`). `advect_scalar_pd` then limits only outgoing
antidiffusive fluxes against that pre-added field; no clip follows. A negative pre-add
(or a rounding-level cancellation in the update) therefore survives to microphysics.

## Method

`pristine_chain.py` extracts six routines verbatim at build time and wires them as
solve_em does in one REAL4 C-callable driver (`chain`):
`horizontal_diffusion('m')`, `sixth_order_diffusion('m', dt/3)`, `rk_update_scalar_pd`,
`advect_scalar_pd`, `rk_update_scalar(rk_step=3)`, `flow_dep_bdy`. Two modes:
- **chain**: everything recomputed from the operands (only physics sc_tend and the QV
  boundary tendency are taken from the port);
- **isolated**: each operator receives the port's own input (sc_tend, preload,
  advection tendency, update), so a difference is attributed to one operator.
Both are built with `-ffp-contract=off` and `=fast` (`-march=native`); their difference
is the per-cell rounding envelope.

Disclosures added after review (cr-o5-f2neg F1/F2/F6): a post-hoc band split (R0 ≤ 1 eps·S, R1 1–64,
R2 > 64 eps·S) shows the sign is rounding-determined in R0; the 64 eps·S tolerance cannot detect
mass-wiring errors, which are proven by 7 array-equality checks instead; the 7×7×3 footprint is
vertically narrow (all-cell outliers at subnormal cells vanish with FTZ + 7×7×5).

Classification rule (`replay_lib.py`, pre-registered before any data was seen):
`tol = max(8·E, 64·eps32·S)` with E the FMA envelope and S the field maximum over the
7×7×3 stencil footprint; every port-negative cell is FAITHFUL (|port−WRF| ≤ tol),
LARGER_IN_PORT (port more negative beyond tol) or SMALLER_IN_PORT; WRF-only negatives
are counted separately.

Evidence levels:
1. **F2N02 frame probe** (`frame_probe.py`): port chain functions vs pristine chain on
   real states of our own GPU Swiss release-default run (v-run hourly wrfout).
2. **F2N01 in-run capture** (`capture.py` + `replay.py`): the port's own full-grid operands
   at the final RK stage of real CPU steps (release defaults), captured with
   `jax.debug.callback` taps, replayed through the pristine chain.

Results and verdict: `REPORT.md`.
