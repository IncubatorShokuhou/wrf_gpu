# w44-branch — CPU-qualified, GPU gate pending

CR11 runtime-table delta: the identity now binds WRF/package runtime data and
selected large assets by content, including the actual RRTMG source resolver,
fallback files, wheel resources and linked data directories. See
[RUNTIME_DATA_BINDING.md](RUNTIME_DATA_BINDING.md) and `RUNTIME_DATA_DELTA.json`.
The unchanged critic LANDUSE reproduction now refuses the empty branch while
still observing all 1764 ALBBCK cells change. A reviewed reader registry and
new-reader/deletion controls protect future loader additions. AOT key/writer/
numerical sources remain unchanged; no GPU arm is authorized by this delta.

CR02 output-policy delta: `b02ad28525` adds the resolved output policy to the
restart identity without changing the AOT cheap key or writer. It binds history
variables/subset/compression/FULL_WRFOUT/cadence, regional streams, filename
format and effective writer flags. BR16: 77 PASS including the original critic
reproduction, eleven output-flag regressions and legacy restart/training-writer
consumers. BR17: deleting the binding causes 11 genuine assertion failures and
zero setup errors. See [OUTPUT_POLICY_DELTA.json](OUTPUT_POLICY_DELTA.json).
The prior V2 GPU spec is withdrawn; it predates this fix. A replacement pin/spec
must be used after the same critic accepts this delta. Numerical/default-program
proofs carry forward: AOT key, writer and operational numerical source files are
byte-identical. This change adds one host policy resolution/namelist parse and
small JSON metadata per checkpoint-enabled run, with no step operations, new
carry leaves or device allocations. GPU timing remains unmeasured.

Objective: explicit checkpoint forks with declared changes on all nests, strict
identity/provenance, and forecast windows on whole root steps.

Implementation: host-only branch admission; ordinary restart still compares its
identity exactly. New checkpoints advertise branch schema 1. A branch receives
a new history journal and records its parent chain, manifest and digest. Raw
payload edits precede device restoration; undeclared fields retain their bytes.
The production numerical step and carry structure are unchanged.

Operator documentation: [checkpoint branching](../../docs/checkpoint-branching.md).

Dependencies (manager approved; original authorship retained):

- `src/gpuwrf/io/wrfinput_consistency.py` exact blob from w44-override
  `8bacb7e0c5`, imported as `b3a1e1ef18`.
- `src/gpuwrf/surface_params/controls.py` only from w44-params
  `a8cc80f69f2e214d48e3da1c49dcb14bf6b3df2e` and
  `569c8a222644313cf785b515f7831841edc8807e`, imported as `d18365f5ac` and
  `15c0b6dfec`. The owner remains w44-params; main takes params first, then
  this branch rebases and drops duplicate patches. Exact final file blob matches.

Measured CPU evidence:

- BR04, `cacd4c0cc1`: 102 checks PASS in 14.26 s. Separate ordinary resume,
  null branch and edited branch through actual driver orchestration with stub
  step/writer; child ordinary resume, identity/schema/tamper/shape controls,
  all-nest retained NaN/Inf rejection, existing restart and CLI consumers.
- BR02b/BR03, `66df8398c9`: frozen default canonical programs EXACT:
  Swiss `b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a`;
  PROD `321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b`.
- BR06, `1531501044`: 175 CPU consumer checks PASS in 21.20 s, including
  fractional CLI duration, single-domain restart, prefetch and memory planning.
- BR08, `6198bb7e55`: all 27 original WN3 input-view branch checks PASS;
  202 total checks PASS and one static-token assertion FAIL. Fixed by
  `d78af7727a`; BR09 passes both changed-artifact checks, including fresh
  process identity. This changes only the restart identity view, not AOT keys.
- BR05b: 5/5 in-memory deletion controls killed with 17 assertion failures,
  zero setup errors. BR07 launcher dry-run PASS with eight pinned WN3 input
  hashes and no JAX import; not a GPU execution result.
- BR01: 89 PASS, one assertion on refusal wording. Correct refusal behavior;
  wording corrected and covered by BR04. BR02a: incorrect CASE path; no identity
  evidence, corrected in BR02b. Both failed attempts retained and logged.

Artifacts: `<USER_HOME>/wrf_gpu2_lanes/w44-branch/BR*.{log,xml,json}`.
Caches: disabled for CPU arms. cpu01 snapshot cleaned after final dependency.

Final affected host gates:

- BR10, `b45fc78f86`: 92 PASS in 15.26 s, including actual native CTOPO
  identity separation, declared fork admission and undeclared-value refusal.
- BR11, same source: 6/6 deletion controls killed, 29 genuine assertions and
  zero setup errors. The sixth deletes the shared checker call and causes 12
  failures on original WN3 input views.
- BR12: 27 existing surface-control consumer checks PASS in 22.44 s.
- BR13: original WN3 three-domain input-view host admission (shared CPU): null
  median 0.00568 s; QV first call 5.2833 s, later median 0.24284 s. Traced peak
  allocation: null 1.35 MiB, QV 302.53 MiB; process peak RSS 1.106 GiB including
  imports/input arrays. This is not a full-checkpoint or GPU timing/memory claim.
- BR14: final launcher's actual CPU dry run PASS, eight input hashes and two
  tool hashes exact, no JAX import. Expired/future/missing-GO controls 3/3 PASS
  before lock acquisition. Immutable model pin `b45fc78f868e0bb197853a908e6b97cb266940ab`;
  post-pin source files are exact (later commits touch reports/helpers only).

- BR15: final v2 launcher CPU dry-run and three pre-lock refusal controls PASS.
  Supervisor 1700 s + outer 1740 s/30 s teardown fit a 30 min window.
  V1 artifacts remain retained and are not submitted.

Pending: runtime-family owner composition and params-first rebase on main;
WN3 0227 3-nest GPU null-resume/null-branch/QV finite
gate; measured ON restore time/peak memory; independent critic and manager RW
gates. **READY-FOR-CRITIC for HOST/BRANCH SCOPE ONLY. Full lane GPU gate remains pending; no GPU fidelity/performance claim.**


GPU handoff (not started):

- Spec `<USER_HOME>/wrf_gpu2_lanes/w44-branch/GPU_SPEC_V2.json`, SHA256
  `90b69a6d1ed466f64b6a087aaeee3c1c3019d0d04ac778fb1d22bfc570fc6aec`.
- Launcher `<USER_HOME>/wrf_gpu2_lanes/w44-branch/tools_v2/night_launch.sh`,
  SHA256 `27a4a555bf16e95598c91e5d450ffa564a61b412ef38225778fa2a7819cbfe3b`;
  30 min GPU cap, manager-provided GA_START_UTC/GA_END_UTC and explicit
  `manager -> w44-branch: GO GPU <spec SHA256>` required. Final snapshot is
  retained only as this queued arm's live dependency.
- WN3 0227: six-hour control, checkpoint at hour three; separate fresh-process
  ordinary resume, empty branch and all-nest QV×0.99 branch to hour six.
  Null arms require exact carry/history bytes; edited seam permits exactly QV
  on each nest and must remain finite. Override explicitly records unbalanced
  fixed-MU and source-supersaturation waivers; it is a research smoke.
- Official fd passed only after device/inode validation; parent has no CUDA
  pool. Child group is terminated and reaped before timeout return. Rich cold
  cache is scanned against the actual control program inventory after release
  of the GPU lock. Successful small receipts are preserved before exact own
  cache/output/snapshot cleanup. Failed gates retain diagnostic data.
