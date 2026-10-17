# Retain parent CLAT after initial two-way feedback

Product `30af97a7e`, branch `lane/t-writer-clat`, base main `f737666b6`.

WRF Registry.EM_COMMON:1848 marks CLAT `u=(copy_fcnm)`. Initial feedback
therefore changes the parent's computational latitude while XLAT stays fixed.
The existing `_input_point_overrides` already computes the correct REAL4 plane
(SHA `ce017b03…`), verified against all 25 original c42 parent CPU frames by
v-prep VP162. The nested writer constructor merges `writer_static_overrides`
into its diagnostics. Its output callback keeps those diagnostics at startup
and later output boundaries, including deferred materialization.

The loss occurred in the full compatibility builder: it unconditionally
replaced CLAT with XLAT. The public payload builder now retains supplied
diagnostic CLAT, then state CLAT, after either compatibility builder. Without
a supplied value, existing fallback behavior is unchanged. No feedback,
cadence, stepping, State, restart or kernel source changed.

Measured CPU gates:

- 69 PASS, 3 unavailable retained-reference SKIP, 0 failures/errors, 11.014 s.
  The 22 new tests exercise the real nested constructor and callback at steps
  0/67/1600, synchronous and deferred materialization, full/default/selected
  streams, state aliases, batched device-get and existing fallback. Unrelated
  input metadata loaders and physics are stubbed; deferred processing consumes
  the real captured OutputSnapshot, without a thread-scheduling claim.
- Fresh publication from actual c42 inputs and the existing feedback producer
  matches 25 parent plus 25 child original CPU CLAT frames BYTE, max error 0.
  All 50 original whole-file hashes are reverified. CPU history is used only
  for comparison, never as a writer input. The proof publishes only static
  CLAT/XLAT/XLONG, not a forecast or a repaired full GPU history stream.
- Restoring the actual parent payload function gives 12 numerical CLAT
  assertion failures, 2 healthy fallback controls and 0 setup errors. The
  premature-import setup failure is retained without numerical credit.
- Host 120×70×44 median: 2.703 → 2.814 ms (overlapping sample ranges), traced
  peak 17.14734 → 17.147244 MB; 48 other full-output fields BYTE. Producer CLAT
  is retained as the same NumPy object. No new array/kernel/transfer is added:
  the existing batched output boundary already materializes this diagnostic.

`QUALIFICATION.json` seals raw and tracked receipts; `SOURCE_REUSE.json` binds
unchanged producer/operator/default-program evidence. `prove_c42.py` reproduces
the field publication and cost check. `mutant.py` exits 1 with the expected
12 assertions and 2 controls. Run only with the official RAM slot, CPU platform,
immutable snapshot, QUIET guard and timeout. All owned scratch was cleaned.

The capped c42 GPU run still requires its separately scheduled completion and
full D6 scoring. This CPU writer fix does not relabel that forecast as PASS.
