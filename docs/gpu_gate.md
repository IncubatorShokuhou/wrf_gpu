# Window gate and exact-call sequencer

`scripts/gpu_gate.sh CONFIG.json` admits one frozen plan and calls
`scripts/night_seq_v10.sh PLAN.json`. The shared Python controller uses only
the standard library and never initializes a model backend. Existing manager
scripts are unchanged; adopting this tool requires a newly reviewed config and
launcher smoke receipts. GPU launchers still use `scripts/with_gpu_lock.sh`.

The controller never kills foreign lock holders or GPU processes. It kills and
reaps only child process groups that it started. A failed, missing, duplicate,
unexpected or skipped planned completion gives a nonzero exit and STOPPED.
STEP_ONLY does not override that verdict. Existing numerical/native gates are
owned by each launcher; a CPU smoke reports GPU gates NOT-APPLICABLE.

## Gate configuration

All paths below are explicit; the example is deliberately private/inert.
Times must include a timezone. The blocked interval includes its start and
excludes its end. The latest start is inclusive, but admission is rechecked
after preflight, the CPU lease lead time and the quiet-core sample. Every
matching GO marker may shorten HARD_END_UTC; a marker cannot extend it.
The complete plan must fit before the effective hard end, with two minutes
of margin per step. Disk arithmetic uses integer **bytes**, without rounding.

```json
{
  "plan": "/private/plan.json",
  "run_dir": "/private/new-invocation-directory",
  "cross_dir": "/private/cross",
  "go_glob": "GO_*.md",
  "not_before": "2026-10-27T01:00:00Z",
  "not_after": "2026-10-27T03:30:00Z",
  "hard_end": "2026-10-27T07:30:00Z",
  "blocked_intervals": [["2026-10-27T01:40:00Z", "2026-10-27T02:00:00Z"]],
  "lease": "/private/gpu.lease",
  "lock_path": "/private/gpu.lock",
  "disk_path": "/private",
  "disk_floor_bytes": 391915765760,
  "disk_need_bytes": 46170898432,
  "cpu_lease": {
    "command": ["bash", "/private/cpu_lease.sh"],
    "directory": "/private/manager/leases",
    "cpus": "3,19,13,29"
  },
  "cross_names": {"DONE": "DONE.md", "STOPPED": "STOPPED.md", "NOSTART": "NOSTART.md"},
  "scrub_patterns": ["^LOCAL_STALE_"],
  "notice_hooks": []
}
```

The run directory must not already exist. Result files are written there;
cross names are basenames and terminal crosses are created exclusively.
Use unique cross names for separate invocations. `result.json` records the
exit code, reason, gate ID, notice count and any unreleased CPU ownership.
The sequencer's captured output belongs only to this invocation; the gate
never searches for the newest unrelated log.

`GATE_NOT_BEFORE`, `GATE_NOT_AFTER`, `GATE_HARD_END`, `GATE_GO_GLOB`,
`GATE_CROSS_DIR`, `GATE_RUN_DIR`, `GATE_PLAN`, `GATE_LEASE`, `GATE_LOCK_PATH`,
`GATE_DISK_PATH`, `GATE_DISK_FLOOR_BYTES`, `GATE_DISK_NEED_BYTES` and
`GATE_CPU_LEASE` override their corresponding fields. Numeric/object
overrides use JSON values. `GATE_PYTHON` selects the control-plane interpreter;
otherwise `python3` must be on PATH. The child interpreter `python` and a
nonempty, existing PYTHONPATH are checked separately for every plan entry.
The actual child interpreter must resolve `gpuwrf` from those declared roots,
without importing it; `required_modules` can name additional root packages.

The CPU helper uses the manager's `cpu_lease.sh acquire NAME CPUS EXPIRY JOB`
and `release NAME REASON` interface. Its lease directory and its own
CPU_LEASE_DIR configuration must agree. It must publish the CURRENT table
row and lease before returning success. Gate-generated UUID names avoid
foreign-name collisions. Acquisition occurs only after GO/resource/preflight
admission. The helper's 30-second pin-watcher lead is enforced by checking the
lease's start-utc, followed by a 60-second sample requiring every requested
CPU at most 50% busy. End, abort, signal and deadline paths release ownership;
a release error is retried and can never produce DONE. There are no standing
CPU reservations while the gate waits for a window.
The deadline is checked again after a child exits, so completion between polls
after expiry cannot be accepted. During terminal cleanup, SIGINT/TERM/HUP are
recorded; a signal cancels and reaps the first release helper, then its bounded
retry and result/cross publication finish without asynchronous unwinding.
That path exits 130/STOPPED and reports any retained ownership truthfully.

## Sealed plans and smoke

Each plan has a nonempty `steps` array with distinct lane names. Each step:

```json
{
  "lane": "example-mini",
  "path": "/private/sealed/launcher.sh",
  "sha256": "FULL_SHA256_OF_LAUNCHER",
  "cap_min": 5,
  "args": ["mini", "an argument with spaces"],
  "env": {"PYTHONPATH": "/private/snapshot/src"},
  "phases": ["model", "e41", "check", "registry_end"],
  "registry_smoke": ["python", "/private/snapshot/scripts/alisios_registry.py", "start", "/private/metadata.json", "--atlas-dir", "{smoke_dir}/schema-atlas"]
}
```

`env` is the intentional child environment delta. Inherited model flags,
smoke selectors and secret-looking names are scrubbed; transport variables
such as DBUS_SESSION_BUS_ADDRESS, XDG_RUNTIME_DIR, HOME and PATH survive.
Secrets and smoke selectors cannot be reintroduced by a step's env.
Args are passed directly as an argv array, without shell evaluation.

Standalone preflight: set `SEQ_RUN_DIR` to a private result directory and
`PREFLIGHT_ONLY=1` when invoking `night_seq_v10.sh PLAN.json`. Normal execution
also requires `HARD_END_UTC` (otherwise a two-hour bound is used).

Set `SEQ_SMOKE=1` for the exact sealed launcher, argv, env and caps on CPU.
The sequencer exports `WRF_GPU2_SEQ_SMOKE=1`, `JAX_PLATFORMS=cpu`,
`WRF_GPU2_SMOKE_DIR`, `WRF_GPU2_SMOKE_RECEIPT` and
`WRF_GPU2_REGISTRY_SMOKE=1`. The launcher must substitute the device/private
lock only, execute every declared phase and write this JSON to the supplied
receipt path after success:

```json
{
  "device": "cpu",
  "real_lock_used": false,
  "completed_phases": ["model", "e41", "check", "registry_end"],
  "registry_schema_valid": true,
  "production_registry_writes": 0
}
```

Missing/duplicate phases, a non-CPU device, a real-lock claim or a missing
registry proof fail closed. A schema smoke command is mandatory and runs
before the launcher; it must use the real registry parser with private or
intercepted storage, retaining exactly the production payload/argv. The
v05-registry helper accepts `start METADATA.json --atlas-dir PRIVATE_ATLAS`;
the exported WRF_GPU2_SEQ_SMOKE flag activates its smoke mode. `{smoke_dir}`
in schema argv is replaced with the private sequence result directory (no
shell evaluation). Use a separate schema atlas from the launcher's own
lifecycle atlas, so the precheck cannot create a duplicate lifecycle event.
The shared helper is being developed separately; until integrated,
the tests deliberately use a private schema double. This is a control-plane
test, not evidence about the external registry parser.

Optional `notice_hooks` fields: `min_free_bytes`, `lane`, `entry_glob`,
`entry`, `pattern`, `command` (argv array). Below the byte threshold, the gate
waits for the exact lane RUN, then an ENTRY.json with the requested entry,
mode `gpu` and `env.GA_GATE_ID` matching this invocation. Only its sibling
session.log is searched. The command receives ENTRY path and gate ID and runs
once, before that lane END. Old mtimes are allowed; foreign IDs and unbound
logs are rejected. Hooks are manager-configured notice actions, never enabled
implicitly. No background watcher survives the gate.

## Private verification

`tests/v05/test_gpu_gate.py` preserves the 29 cr-w44-launchers controls by
name and intent, plus RD078 and CPU-lease regression controls. It requires
neither a model backend nor operational data. Run CPU-pinned with a private
pytest basetemp and the repository's QUIET guard. The shell entrypoints must
also pass `shellcheck scripts/gpu_gate.sh scripts/night_seq_v10.sh`.

Performance/memory: no timestep/model changes, device operations, model
intermediates or compile work are introduced. Controller work is one stdlib
process plus short admission checks and current-invocation log reads; the
30-second lead and 60-second quiet sample precede the timed run. CPU time/RSS
are measured with the private controls, not presented as forecast speed.
