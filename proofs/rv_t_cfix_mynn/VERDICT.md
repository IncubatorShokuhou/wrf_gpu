**FIX — do not merge b8fab854ff as-is.**

Reviewed lane/t-cfix-mynn at `b8fab854ff9d7d6f42d41a959c16529aa227422d`,
product `1fb4903d277c47b5d7b31cc081488da1fd2fedb1`, base `46868408b`.
CPU-only independent critic, cores 24,25 / nice19 / 6 GiB RAM slot. No product edits or GPU work.

**One blocking finding: the driver-local limits change other schemes' inputs.**

Pristine `module_bl_mynnedmf_driver.F:381–397` limits only its local `hfx1/qfx1`.
`module_first_rk_step_part1.F:1416` still passes the physical grid HFX/QFX to
cumulus; `module_cumulus_driver.F:1396` passes that QFX to CU_TIEDTKE.
The new `noahmp_coupler.py:458–461` instead writes limited values into shared
`State.theta_flux/qv_flux/fltv`. Those handles are not exclusively MYNN inputs:
`scan_adapters.py:1035–1037` reconstructs Tiedtke QFX from `state.qv_flux*rhosfc`.
GF and NTiedtke also reconstruct HFX/QFX from the same handles (:1164–1167,
:1331–1334). Their physical-flux reconstruction was already approximate; the
additional clamp-induced change is new.

[M CPU] `flux_scope.py` runs the actual Noah-MP producer twice with identical
physical HFX=1500 W/m² and QFX=0.0012 kg/m²/s, toggling only `pbl_limits`.
A sentinel cumulus kernel exposes the actual Tiedtke adapter QFX operand.
Physical State HFX/QFX remain BYTE-equal; the Tiedtke operand falls from
0.0012057740535563846 to 0.0009043305401672885 kg/m²/s, exactly 0.75×
(maximum change 0.0003014435133890961). All six cells reproduce this.
This is a caller-input witness, not a scientific forecast or a GPU measurement.
It fails the contract's **no change outside MYNN** gate. The limits are supposed
to matter precisely at these extreme fluxes, so in-range PROD/WN3 samples do not
close this failure.

Required correction: isolate the limited view to the actual MYNN consumer, or
preserve physical HFX/QFX at every non-MYNN consumer through an equally narrow
change coordinated with their owner. Add an end-to-end producer→cumulus input
regression with fluxes beyond both bounds, healthy in-range controls, and a
genuine mutant. Check all consumers of these shared handles. Preserve physical
State HFX/QFX. Do not encode producer placement as the correctness criterion.

**Passing evidence, to retain.**

- [M CPU] 25/25 registered tests pass, no skips/errors: pristine public driver
  versus logging-only core BYTE in every compared public output; 166-column
  taper BYTE; driver flux literals/bounds BYTE; caller seam at both precisions;
  kte-face exclusion; Noah producer and two-step operational caller seam.
  Whole-column MAXMF worst difference 9.047798812389374e-6 m/s against the
  registered `2e-4 + 1e-3*abs(reference)` bound.
- [M CPU] All five genuine mutants killed by assertions, zero setup errors:
  mass w, upper faces, missing kte bound, missing limits, unlimited Noah producer.
  Snapshot source restored and checked clean afterward.
- [M CPU] Independent original-history replay replaces the worker fixture's
  nonhydrostatic pressure/EOS density/alternative dz expression with WRF-derived
  hydrostatic p/PSFC, hypsometric ALT density, and literal dz8w. All 166 tapers
  remain BYTE; all six MAXMF gates pass, worst difference
  3.901030868291855e-5 m/s. The pressure operand changes by up to 71.8671875 Pa.
  Held WSPD is not published in these histories: its worker approximation is
  retained, so neither replay is claimed as an exact live caller savepoint.
- [M CPU] Official native-facade release probe reproduced independently, with
  only its scratch destination changed: Swiss
  `ec49db53e7f9ac5dbe01cd231802c6adf14e6d52b52605c0d91ef0aa6213eff9`;
  PROD d01 `913eab9a95fef5248cf89725d23fa43cce7dc8f380c02a76705769f45a3eceb9`.
  These are CPU jaxpr identities, not GPU execution receipts.
- [M source] Every product hunk was checked against pristine WRF: bottom faces
  kts..kte, resolved-w loop through kte-1, REAL4 bound literals and statement
  order, and preservation of the physical HFX/QFX slots. No architecture change.
  Six product files changed; source/reference hashes are in `source_scope.json`.

Performance/memory: [I source] removing the mass average eliminates full-field
w addition/scaling; the extra kte mask is a static level predicate and the
limits add four surface compares/selects. No new persistent carries, host
transfers or synchronization calls. These observations do **not** establish
compiled launch count, temporary liveness, peak VRAM, or timing neutrality;
[U] actual GPU cost and E41 remain pending. A fix should avoid separate surface
solves, new carried arrays, or unnecessary launches.

The worker's no-LSM/RUC-water/other-surface residual limit gaps remain disclosed
Class-B limitations; they are separate from the newly reproduced scope leak.
The worker's legacy real-Swiss smoke is retained evidence, not independently
re-executed here; release-native execution and full GPU re-qualification remain
with the manager. PROD/WN2 column replay is intermediate evidence only.

Next: worker fixes the scope issue; the same critic checks the changed seam and
new identities once. Reuse bit-identical face/oracle evidence. After acceptance,
manager runs the required WN3 0227/0502/0614 ×24 h, PROD, Swiss and hard E41 gates.
No review finding permits a numerical tolerance or release-gate waiver.
