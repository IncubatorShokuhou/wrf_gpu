# Native restart identity JSON regression

First bad main merge: `8171990c4`, integrating land options from `55bf5cd90`.
The actual epsilon restart continuations passed on first parent `d1fcbe092`
(2 tests) and failed on the merge (2 identity errors, no setup errors).
`nested_identity` unconditionally added irrigation tuples; optional BATS table
constants were tuples too. JSON readback turns these into lists. Both native
checkpoint verification and history-journal reopening compare identities exactly.

The fix canonicalizes the producer's complete identity through JSON before any
publication. All reader, latest-generation, rotation and journal comparisons
remain exact. The existing wire representation already uses JSON arrays, so the
on-disk format has not changed. The source digest still refuses old-source
checkpoints. Physics, carry schemas and compiled timestep files are unchanged.

Evidence in `receipts/` and `<USER_HOME>/wrf_gpu2_lanes/t-restartfix/`:

- 100 targeted tests pass: all 58 epsilon and 31 TEMF native restart tests,
  four identity regressions and seven land-option restart tests. This closes
  the 46 epsilon and 30 TEMF identity failures reported by t-redsuite.
- Deleting only the final canonicalization produces three genuine assertion
  failures with zero setup errors: default irrigation, active irrigation and BATS.
- Real Swiss CLI, 42×42×44, dt 18 s, CPU, `GPUWRF_FAST_DEFAULTS=0`: uninterrupted
  200 steps versus a verified step-100 checkpoint, self-SIGKILL, and fresh-process
  resume to 200. All 155 terminal carry leaves and all 754 history variables
  across two files, plus global attributes, match byte-for-byte. The only model
  harness adaptation redirects State's device guard to CPU; the product solver,
  stepping, CLI, checkpoint store and journal run unchanged.
- In a fresh process, native storage and each standalone format (operational
  pickle, NetCDF, M7 checkpoint with companion land) restore the same 155
  step-100 leaves byte-for-byte. This is restart/determinism evidence, not WRF
  fidelity or a GPU execution claim.
- Official `proofs/s1_port/release_identity.py`: Swiss
  `f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d`,
  PROD d01 `6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b`.
- All 456 timestep source/key closure files are byte-identical to base
  `1d791420d`; the sole changed product file is outside that closure.
- Shared consumers: 551 unique passes. The one remaining failure is
  `test_identity_excludes_end_time_but_binds_the_stream`, whose incomplete
  `fake_nml` lacks `sst_skin`; t-redsuite owns that test-only repair. The
  manager explicitly declared this lane's gate green and requested READY.
  The first shared arm's 304 setup errors came from my symlink basetemp;
  only those unrun cases were rerun with a real directory (303 passes plus
  the declared fixture failure). Its timing spans hibernation and is invalid.
- Actual 2,544-byte Swiss identity: 1,000 CPU stdlib measurements give median
  27.32 µs, p95 29.43 µs, and 17,119 bytes of additional temporary Python memory.
  Conversion runs once at restart-enabled initialization, outside the timestep.
  No GPU timing or full-forecast speed claim is made.

`cli_probe.py` runs the product CLI with `TRF_ROOT` and `TRF_ACTION=control|cut|resume`.
Use a private DATA-disk root, `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0`, the authorized
CPU affinity/nice/QUIET admission, and `--hours 1`. The cut action requires
`--checkpoint-interval-steps 100` and exits by SIGKILL after all exports. Resume
the generation recorded in `cut_generation.json` in the same stream, then run
`verify_cli.py ROOT`. `mutation_probe.py` initializes the CPU guard before JAX
and expects exactly three assertion failures.

All failed setup/harness attempts are retained and explicitly discarded in the
main experiment ledger: RF01 AST setup, RF06 CPU guard ordering, and RF07's
additional M7 export omitting mandatory companion land. RF07's valid control
was reused after the export fix. No numerical bound or identity guard was relaxed.
Finished ROOT/DATA run data and old snapshots were removed after their consumers;
small receipts, hashes and scripts are retained. Shared-consumer receipts are included.
The final source candidate `a20e70d41` is rebased onto approved main `e45892a4e`.
The complete restart-store file is byte-identical to all successful own gate
trees. RF10/RF11 bind the earlier measured source; approved main independently
confirms the same full frozen hashes (SI82). `FINAL_HANDOFF.json` records that
evidence reuse and the manager's explicit final-composition acceptance.
