# v05-aotfp — CPU lane gate PASS

Base: exact v0.4.4 RC `286e2442a9`. Production changes are in `8323e1dccc`;
subsequent changes affect tests/proof tooling only. No merge/tag/GPU work performed.

The internal AOT predicate now requires all eight saved target fields. Unknown
values refuse admission, except the CPU's inapplicable compute capability.
`allow_legacy_fingerprint=True` explicitly restores partial legacy comparisons;
a trusted caller must additionally pass `check_fingerprint=False` to skip the
comparison altogether. The opt-in propagates through domain loaders. Speculative
handles retain the strict policy, and cannot satisfy an unnamed unchecked request.
Ordinary callers continue to fall back to a fresh compile after refusal.

`gpuwrf --version` prints the package version: actual CPU subprocess returned
`gpuwrf 0.4.4`, exit 0. Package version and cache schemas were not changed.

## Gates and evidence

- **20/20 supported-caller host controls:** run, init, fused, prefetch and the
  actual bundle consumer, each with matching/cuda_driver/XLA_FLAGS/empty saved
  fingerprints. Matching controls execute a real tiny CPU-native binding; bad
  controls return loaded=false, a clear error and no binding execution. All four
  bundle views stayed clean with zero new compiled entries. These use model
  stand-ins after admission: no GPU or WRF fidelity claim.
- **257/257 AOT/cheap-key/fastcall/version test nodes covered:** 91 passed before
  the first 600s timeout; the ordered remaining166 produced165 passes and one
  new fixture failure. That positive prefetch fixture omitted required KEY_SCHEMA;
  the test-only correction passed. Production bytes stayed identical throughout.
  `SUITE_NODEIDS.json`, the report ledger and raw logs bind the disjoint prefix,
  suffix and corrected node. This is combined coverage, not one all-green invocation.
- **21/21 existing CLI tests passed.**
- **4/4 genuine deletion kills:** required fields, named opt-in, speculative
  policy, version. Each raw log contains the intended pytest assertion failure;
  setup failures cannot count. An initial mutation harness misclassified a
  nested-AST setup error by rc alone; that aggregate was invalidated, corrected,
  and superseded by these four actual failures.
- **Real compact AOT identity:** v0.4.3 d01 blob103,718,762B, SHA
  `ed5e1030e879b1e47f6b5ed224540dbefa480ed113367377e697329cea61dd3b`.
  Actual format is compiled-thunk with **retained_module=slim, trace_metadata=rich**;
  the initial selector's assumption about compact trace labels was wrong.
  Blob, metadata and stored cheap key are byte-identical after read-only inspection.
  Serializer/target_fingerprint ASTs and cheap-key source/package-version bytes
  are unchanged. None of the three changed source files is in the541-file traced
  source closure. The old/new predicates accept the identical saved target pair.

## Cost and limits

[M CPU paired microprobe] Predicate median1.137→2.059µs, +0.923µs/check.
This is host admission cost only, not GPU readiness or forecast timing. Source
inspection shows no new device buffers, copies, kernels or timestep transfers.
Explicit legacy requests skip speculative reuse and can deserialize again; normal
supported calls retain reuse. Test/probe scopes were CPU12/28 nice19, MemoryMax8G,
no swap; observed peak suite memory3.5G. JAX was CPU-pinned; persistent JIT cache off.

The CPU-only lane contract does not authorize RW1–RW4 GPU runs. Those and independent
review remain the manager's merge/release gates. Nothing merges before the v0.4.4 tag.
Disposable private caches, pytest trees, overlay layers and the detached snapshot
were removed after the final reader; small receipts and source scripts are retained.
The real release bundle and WRF references were read-only and untouched.

## Reproduction

Use a frozen checkout and a fresh output directory, setting `JAX_PLATFORMS=cpu`,
`V05_AOTFP_SOURCE`, `V05_AOTFP_ARTIFACT_DIR` and PYTHONPATH to its `src` and root.
Run `host_controls.py` for the16 direct controls. For each field, use the checkout's
actual `scripts/release/bundle_consumer.py --bundle .../v0.4.3 --layer <private>
--no-verify -- python host_controls.py --worker bundle --field <field>`; its view
checker still runs (`--no-verify` skips only the separate base-P checksum scan).
`identity.py`, `mutations.py` and `host_cost.py` reproduce the remaining probes.
Apply the lane's current quiet/core/memory rules to every command. Primary receipts
are in `receipts/`; original raw artifacts remain under the lane directory.
