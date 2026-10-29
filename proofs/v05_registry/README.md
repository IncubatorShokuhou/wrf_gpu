# Shared ALISIOS registry hooks

Gate: **PASS**, immutable candidate `260e73155a`, real ALISIOS `register.main` +
schema + state checks + flock/append on private atlases. See `GATE.json` and the
three logs here. No model/GPU/JAX execution; no OBS or protected extracts read.

- 28 tests pass; malformed forcing and missing issue return 2; duplicate starts,
  duplicate ends and ends without starts are refused.
- Removing the finally-end gives two AssertionErrors; removing the schema
  precheck gives one AssertionError. Both mutants have zero setup errors.
- Both production registry JSONL hashes are identical before/after. A private
  copy of the 5,434,103-byte WRF ledger retains its exact prefix and receives
  exactly ten private events for five paired hook measurements.
- [M] Median host start/end lifecycle: 0.204739 s; driver peak RSS 50,020 KiB,
  child peak RSS 32,844 KiB. This includes real disk fsync and full-ledger state
  checks. It is host registration cost per invocation, not forecast timing.
- [I] No timestep kernels, device allocations, transfers or compiler programs
  change: this patch touches no `src/` file and the helper imports no model/JAX.

Use `scripts/alisios_registry_launcher_example.py` as the 20-line adoption
template. The sequencer still owns locking, timeouts, caps, and substitution of
the CPU smoke command. Pass JSON start metadata matching real ALISIOS flags
(`run_id`, `by`, `producer`, `purpose`, `family`, `forcing`, `engine`, `nests`,
`lead_h`, `issue`, `run_dir`, `retention`). `lead_h` is maximum lead from the
forcing issue, including any input offset, not necessarily model duration.
Keep descriptions in `purpose`/`engine`, not `forcing` (e.g. `ERA5`, `WN3+AIFS3D`).

`Registry.run(start, end={...})` validates terminal metadata and starts before
yielding. Set `outcome.rc` to the actual child return code. Finally records
FAILED for failures, ABORTED for Python interruption/exit, DONE_CLEAN only for
successful receipted raw deletion, otherwise DONE_HELD. Terminal metadata needs
`corpus` and boolean `raw_deleted`; optional `extract` is a list of
`/absolute/path::SHA256` strings. No extracts are read or automatically hashed.
Child environment must contain both `ALISIOS_WRF_RUN_ID` and
`ALISIOS_ATLAS_DIR`; the template supplies them. A failed end raises for repair.

For `WRF_GPU2_SEQ_SMOKE=1`, explicitly supply a private atlas through
`REGISTRY_PRIVATE_ATLAS` (template) or `Registry(atlas_dir=..., smoke=True)`.
Empty private atlases and copies of the two registry JSONL files are supported.
No production-atlas ancestor/descendant or linked production registry file is
admitted. Smoke executes the same real registry argv and append path as live.

Authorized repair: `Registry.update(metadata, authorization='manager decision reference')`.
It writes an update with the reference in notes; it cannot reopen an ended run.
No duplicate start/end is used for a postprocessing correction.

Receipt replay: `Registry.backfill(receipt)` accepts reviewed JSON shaped as:

```json
{
  "start": {"run_id": "unique:attempt:id", "started_utc": "2026-10-26T13:00:00Z", "...": "all start fields"},
  "end": {"run_id": "unique:attempt:id", "ended_utc": "2026-10-26T14:00:00Z", "...": "all terminal fields"},
  "evidence": {"/absolute/path/MODEL_RECEIPT.json": "64-character SHA256"}
}
```

The receipt adapter owns the factual mapping from model/timer/cleanup receipts;
this helper never infers missing execution, time, rc or cleanup evidence. Both
schemas are checked before writing a new start. Identical replay writes nothing;
an existing matching start can receive its missing end. Conflicts require an
authorized update. Evidence must be small metadata files with matching hashes.
SIGKILL/power loss cannot execute Python finally: reconcile those from reviewed
receipts. The helper does not automatically install hooks into other lanes'
sealed launchers; their owners can adopt this interface after the source freeze.

CLI: `JAX_PLATFORMS=cpu python scripts/alisios_registry.py start|end|update|backfill METADATA.json
--smoke --atlas-dir /private/atlas` (`update` additionally needs `--authorization`).
For a fresh proof, archive the four source paths listed in `GATE.json` from the
commit, then run `check_gate.py --source ARCHIVE --out LANE_ARTIFACT_DIR --commit SHA`
on cores 12,28 under the current CPU/QUIET/RAM policy. The gate removes its own
private atlas copies and mutants; archive cleanup belongs to the caller.
