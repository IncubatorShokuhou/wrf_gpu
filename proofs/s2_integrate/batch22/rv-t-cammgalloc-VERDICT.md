# MERGE — t-cammgalloc

Sole critic: rv-t-cammgalloc (Codex GPT Sol). Reviewed immutable
`ac6aa7a52bf480e5db4f7dc94137a9f417fd19fd`, product `958ce5266`;
source tree `b65e2e1affd2e6b7b772ccb4b8b16a765285637a`.

The two changed product definitions fix inactive CAM-MG allocation and select
both radiation overrides by the active microphysics producer. No fixes required.

## Independent measured gates

- RV01: **all 12 assigned C123 failures pass**, including strict restart readers
  and the Dudhia radiation handoff. The complete selection produced **65 PASS,
  1 unrelated FAIL, 0 ERROR** in 52.88 s, peak RSS 1,361,608 KiB.
  The extra NSSL-18 test fails with `restart radar carry and selection disagree
  (E78)`. The prior RS03/C123 XML contains the identical failure, and its test
  blob is byte identical to that earlier source. This is the separately owned
  radar-fixture issue, outside these 12 gates; no test or tolerance was changed.
- RV02: **9/9** compiled radiation outputs, constants and canonical traced
  radiation-input programs are byte/exact: active MP11 versus the old function;
  MP6/8/24/28/40/50/55/56 versus pre-CAM `0602830b4`. Common carry values are
  byte identical; exactly 20 inactive CAM leaves disappear, and active MP11
  retains all of them. These are regression proofs, not new WRF forecast claims.
- RV03: independently written allocation, radius and cloud guard deletions each
  produce **1 genuine assertion failure + 1 healthy control, 0 setup errors**.
- RV04/RV05: both official **FULL** default traces remain exact:
  Swiss `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`;
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
  Supervisor wall 37.57/65.83 s, peak RSS 3,008,584/3,873,344 KiB.
- RV00 verifies all **21 producer receipt SHA256s** and byte identity of the
  reviewed source tree to the producer's measured `58c69a762`. The producer's
  263 shared-consumer passes are reused, not counted as independent reruns.

## Source, scientific scope and efficiency

Pristine WRF `module_physics_init.F:1362–1364` defines CAM-MG selection from
MP11; `module_radiation_driver.F:1433–1444` gates the cloud override with that
selection. `Registry.EM_COMMON:3031` owns the CAM-MG package. The patch retains
every other legacy conditional leaf and every active CAM-MG leaf. The entire CAM
science tree, kernel, metadata, strict restart validator and operational entries
are byte unchanged. Inactive populated CAM checkpoints remain refused.

The native CLI loader uses selected-only `_initial_carry_for_run`; it bypasses
the faulty all-tail helper. Existing native composite runs are unexposed to this
allocation bug. Public/daily paths use the faulty helper: cloud/radius overrides
were affected, and presence-dependent PBL work also returns to its inactive
route. The separate c89 precipitation issue is unchanged.

Static Python selection adds no timestep operations, transfers or allocations
[I]. Affected public PROD d01/d02 carry saves 22,478,400/83,595,564 bytes [I;
shape arithmetic]; native CLI carry is unchanged. Both default program hashes
are exact; normal source-key cache invalidation can still require first-use
rebuild. No GPU timing, peak VRAM, E41 or new trajectory-fidelity claim is made.
The producer's broad option matrix OOM at 6G remains disclosed; its subsequent
263-pass selection is not a full-suite result. All critic JAX arms used CPU,
cache OFF, cores 12/13/28/29 nice19, official 6G/no-swap RAM slots, QUIET guards
and timeouts. No GPU arm or subagent was started.

Manager may integrate the reviewed commit. No further review or numerical
repetition is required for these exact bytes.
