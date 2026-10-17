**MERGE — lane/t-cfix-mynn `946141e288c11cd84244a62603b5e34f667223fd`.**

Final source/CPU review of the FIX round rebased on main `dca7a4d79`.
The earlier FIX decision on b8fab854ff is preserved unchanged; its sole scope
blocker is closed in this candidate. No remaining source/CPU review blocker.
This admits integration and the manager's default GPU re-qualification; it is
not a release or full-forecast scientific verdict.

[M source] Noah-MP, lake and RUC producers are each BYTE-identical to the
unlimited producers on main dca7a4d79. Limits now live only in
`physics_couplers._mynn_driver_limited_handles`, called through the MYNN surface
view. All references to that view were checked: MYNN adapters and MYNN PBLH
diagnostics. The State handles and physical HFX/QFX are not rewritten by the
view. REAL4 bound tests and the unchanged oracle-proven driver statements form
the limited fluxes only for MYNN; in-range and at-limit columns retain the
producer bytes. The existing PSFC column view is shared rather than rebuilt.

Independent changed-boundary verification on an immutable 946141e288 snapshot:

- [M CPU] **10 PASS**, 19 unchanged tests deselected, 47.21 s. Includes actual
  grid-backed Noah-MP producer→cu3/cu6/cu16 extreme-flux HFX/QFX operands,
  both signs, in-range and at-limit controls, the finite REAL4 MYNN-only view,
  shared producer/caller scope, and the two-step operational caller seam.
  Other schemes receive the unlimited reference operands BYTE; the MYNN view
  receives limited fluxes while preserving the shared State.
- [M CPU] Both changed-boundary mutants killed, zero setup errors:
  producer-side limits give four assertion failures; disabling the MYNN view
  gives one. Independently repeated the producer-leak mutant with its
  keyword-placement assertion omitted: **all three cumulus numerical operand
  comparisons fail by AssertionError**, zero errors. Product and test source
  were restored BYTE after the probes.
- [M source] `REUSE.json` proves 10 relevant files BYTE-identical to b8fab854ff,
  including MYNN kernels, the literal flux helper, public oracle/caller code
  and real-column fixtures. Four coupling functions, including bottom-face
  extraction and MYNN column assembly, have identical ASTs. Carry forward the
  independently verified 166-column taper BYTE and pristine public MAXMF
  gates, including the WRF-derived operand replay worst 3.901030868e-5 m/s.
  Those unchanged numerical arms were not repeated. The existing disclosed
  history-WSPD approximation remains; no exact live-savepoint claim is added.

Independently reproduced official native-facade release jaxpr hashes
([M CPU], FAST1/CENSUS0, one-step trace; no GPU execution):

- Swiss: `7379c7d77e94517f07e7bf37763cc1409d1287a13e50684ed9d624caafaec55e`
- PROD d01: `6487f5b039a074f02f63d140f1e9212b7ede832d04bf5450e44ad2b3f2c34e73`

Performance/memory assessment: [M source] one surface column view is reused;
there are no new persistent carries, host transfers, or synchronization calls.
The trace adds the local limit/re-formation/select block; the worker's
anonymized inventory is +85/-14 lines. [I] the surface algebra is small and
can fuse, while the mass-w average remains removed. **Do not infer allocation
or launch neutrality from that:** formerly dead full-column p/rho/exner layout
values now supply level-zero slices. [U] optimized-HLO elimination, live fp64,
temporary liveness, launches, peak VRAM and actual timing require the manager's
real executable checks. No CPU elapsed time here is a GPU speed claim.

Required next step: merge this exact reviewed candidate, then the manager runs
the full default WN3 0227/0502/0614 ×24 h qualification, PROD, Swiss, and hard
E41/compiled-performance checks. No tolerance or release-gate waiver. All own
snapshots, pytest directories and disposable caches were removed; only small
receipts remain. No GPU work or product edits by this critic.
