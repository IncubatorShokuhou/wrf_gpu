# `gpuwrf da cycle` — ensemble forecast/analysis cycles in one process (experimental, d01)

`gpuwrf da cycle` advances an ensemble of d01 members through forecast windows with an analysis between
windows, and `gpuwrf da members` runs the members of one `ENSEMBLE_MANIFEST.json` in one process. Both are
opt-in and experimental; the default `gpuwrf run` program is unchanged. The analysis itself is
[`gpuwrf da analyse`](da.md) (LETKF); this page covers the member driver and the cycle loop.

## Why one process

A separate `gpuwrf run` per member pays a fixed cost every time (Python/JAX imports, the GPU backend, loading
the step executable, initialisation). On the case of the first data-assimilation study, that fixed part was about 84 % of a warm
6-hour d01 member (that study's cost model, not in this repository: 21.4 s per member, of which ~17 s fixed). The member driver
runs every member as the *ordinary* CLI forecast (`gpuwrf.cli.main(["run", ...])`) inside one process:

* each member writes exactly the history a separate `gpuwrf run` of its case writes (byte-identical, see
  *Evidence*);
* imports, backend and initialisation caches are paid once;
* with `share_executables` (default) every later member reuses the step executable the first member loaded
  or compiled (`runtime.domain_tree.set_process_aot_reuse`: the memo is keyed by the program address —
  domain + cheap key / HLO hash — so a shared entry is the executable a fresh load would return).

Limits that keep this exact: all members of one process must share the namelist outside `&time_control`
(the cloud-overlap flag and other switches are pinned at import) and the `GPUWRF_*`/`XLA_*`/`JAX_*`
environment may not change between members. Both are refused, never ignored.

## Configuration

Either a `gpuwrf.da.v1` analysis file (see [da.md](da.md)) with a `cycle` section — LETKF arms then use that
file's observations, parameters, moisture policy and governance — or a cycle-only file with
`"schema": "gpuwrf.da.cycle.v1"` and the same keys at top level (zero-update arms only).

```json
"cycle": {
  "source_case": "/abs/case",                 // namelist.input + wrfbdy_d01 (+ wrflowinp) covering every window
  "t0": "2023-01-15T00:00:00Z",               // first window start (a wrfbdy record time)
  "window_hours": 3, "cycles": 2,             // window: multiple of 3 h and of the wrfbdy interval
  "arms": {
    "zero":   {"kind": "zero_update"},                     // control: analysis == background
    "letkf":  {"kind": "letkf", "experiment": "A"},        // experiment of the enclosing gpuwrf.da.v1 file
    "letkf_r":{"kind": "letkf", "seam": "restart"}         // same analysis, restart seam
  },
  "initial_members": {"m00": "/abs/m00/wrfinput_d01", "m01": "/abs/m01/wrfinput_d01"},
  "forcing_hours": 6,                         // optional, see below
  "share_executables": true, "require_platform": "gpu"     // optional
}
```

`initial_members` are wrfinputs valid at `t0` with the source case's land state (published as first-window
cases with `land='cold'`); `initial_cases` takes ready case directories instead. Paths are absolute. Member and
arm labels name directories, so each must be one path component (`[A-Za-z0-9][A-Za-z0-9_.-]*`, at most 64
characters); `ctrl` and the arm name `shared` are reserved.

The loop and the member driver read cases, wrfinputs and manifests themselves, outside the guarded analysis
API, so anything under the ALISIOS data roots is refused before it is opened: the inputs themselves, every
entry below a case directory (links resolved), every file name in its namelist resolved the way the run
resolves it (`auxinput4_inname`, `gfdda_inname` or any other key, absolute or relative, templates expanded),
paths named in the process environment, every file below the runtime's table, asset, data and compile-cache
roots (defaults and overrides), and the output roots. While a member forecast or a case writer runs, any
Python or netCDF open of such a path is refused as well. Stage such data outside those roots through the
governed tools first.

Run: `gpuwrf da cycle --config da.json --out RUN [--dry-run]`. `--dry-run` prints the windows, arms and the
number of member forecasts without running anything.

### Arms and seams

* **zero_update** — every member restarts from its own background frame (zero increment). It is the control
  that isolates what the restart seam alone does to a forecast.
* **letkf** — `api.analyse_case` on the member frames at the window end (`ensemble.kind = members`), writing
  per-member analysis wrfinputs and next-window cases (`member_cycle.write_member_cycle`). The analysis names
  its members m00.. in input order; the loop matches every analysed member to its own loop member through the
  background frame it was analysed from (`background_wrfout`), never by label or position.

`seam` decides how a member crosses the analysis:

* `file` (default): the next window starts from the analysis wrfinput, like WRF from a wrfinput — the port
  re-initialises its held physics (radiation tendency until the next radiation call, KF trigger/NCA,
  h_diabatic, accumulators). Noah-MP land is carried through the land-cycle marker.
* `restart`: each member keeps one continuing stream. The member's end-of-window restart generation (restart
  v2) receives only the analysis increment (`analysis wrfinput − background frame`) on u, v, qv, θm (from T
  and QVAPOR), p′/p_total and ph′/ph_total — MU must be unchanged — and the stream resumes with
  `--resume-checkpoint --extend-run`. Held physics, land, accumulators and step scratch stay as the model
  left them (a WRF restart likewise re-reads only state; `rk_step_prep` recomputes ww/alt/php before use).
  Cells with a zero increment keep their stored bits, so a zero-update restart arm *is* the continuous
  forecast, bit for bit. Totals are stored in REAL: an increment below half an ulp of p_total (~0.004 Pa at
  1000 hPa) or ph_total leaves the total unchanged while the perturbation moves, so the base the dycore
  recovers as total − perturbation shifts by at most half an ulp of the total, once per window (the model
  rounds its totals the same way at every step end).

Window 0 is identical for every arm and runs once per member (checkpointed in the first restart arm's stream
when there is one). Analyses and case writing are host-only and run in a CPU child process
(`JAX_PLATFORMS=cpu`), never on the GPU.

### One step program for the whole loop

The wrfbdy record count is currently a program shape (until the wrfbdy record-count fix lands). The loop therefore
writes every case with the same `forcing_hours` of forcing records — one window by default, the whole horizon
(`cycles × window_hours`) when a restart arm exists — so every window, arm and seam shares one compiled step
program. The source case must cover `t_last + forcing_hours`.

## Output layout

```
RUN/c00/shared/cases/<m>      first-window cases (all arms)
RUN/c00/shared/fc/<m>         window-0 forecasts (no restart arm)      | RUN/streams/<arm>/<m>  restart streams
RUN/cKK/<arm>/fc/<m>          file-seam forecasts of window KK         | RUN/checkpoints/<arm>/<m>
RUN/cKK/<arm>/analysis_work/  analysis receipts, members_<exp>/ manifest + cases (letkf) or prep/ (zero)
RUN/CYCLE_RECEIPT.json        per window/stream/member: wall, executable source, history SHA-256, steps
```

Every member directory holds `member_receipt_NNNN.json` (argv, rc, verdict, wall, compile events, executable
source per domain, history SHA-256) and the CLI payload.

## Evidence

* `tests/a1_da/test_member_driver.py` (CPU, no model run): the driver passes each member to the CLI unchanged,
  pins namelist/environment, records receipts and failures; the process registry hands a later member's tree
  the first tree's loaded executable (three member trees, one load) and never shares an in-process `Compiled`;
  restart-seam increments are exact (zero increment keeps the stored bits, including -0.0; θm uses the
  moisture factor; a moved MU is refused); the loop runs window 0 once, keeps one record count, resumes
  restart streams with `--extend-run`; a process whose `gpuwrf` import took the offline path (`gpuwrf da check`)
  is refused. 15 single-file mutants (`proofs/w41_da_cycle/mutants.py`, passing baseline first) each make a
  named test fail.
* Existing advance-factory consumers (`tests/test_aot_executable.py` Step-C/alias/verify tests,
  `tests/test_v024_nested_runtime_reuse.py`) pass unchanged: the registry is off by default.
* CPU end-to-end, Swiss d01, 2 members x 2 windows x 3 h (`proofs/w41_da_cycle/cpu_proof2.sh`): zero-update
  restart arm, LETKF file-seam arm and LETKF restart-seam arm (synthetic sonde, `gpuwrf.da.v1` file). The
  zero-update restart stream equals a continuous separate 6 h `gpuwrf run` (7/7 history files byte-identical),
  window 1 of the LETKF file seam equals a separate run of its case (4/4), two in-process runs of the same
  window-0 cases agree bit for bit (8/8), and the LETKF injection changes u/v/θ/qv while the zero injection
  changes no cell. On CPU, prime the cache first (one separate run): a cold first member plus a later member
  that loads the exported blob keeps two step executables in memory (12.5 GB for Swiss d01; the primed loop
  peaked at 8.3 GB).
* GPU arm, 9 km Canary Islands domain (120×70, release defaults, RTX 5090), N=10 x 2 windows x 6 h
  (`proofs/w41_da_cycle/gpu_arm.sh`). First session (cold cache): all 10 window-0 members `PIPELINE_GREEN` in
  one process; member 1 took 415 s (step compile), members 2–10 took 11.0–14.1 s each for 6 h (median 11.7 s)
  on the shared executable, with no recompile; host memory stayed at 5.7 GiB over all 28 member runs. A
  separate warm `gpuwrf run` process (same cache) took 38 s for a 6 h member. The first lock session hit its
  900 s bound before the separate runs, which ran in a second 2-minute session on the same cache
  (`proofs/w41_da_cycle/gpu_separate.sh`); 16.9 GPU-minutes in total. Every history file was
  byte-identical to separate fresh processes: window 0 of member m01 (7 frames), the window-1 file-seam case
  (7 frames), and the zero-update restart seam against a continuous 12 h run (6 frames after the seam).

Disk: a member window writes its history (FULL stream, ~90 MB per hourly frame of that domain) and, on the restart
seam, one restart generation per window (~108 MB for that domain); N=10 × 2 windows ≈ 12–18 GB. Delete or prune
old generations when they are no longer needed.

Why a cold in-process `Compiled` is not shared: it checks its full input treedef, and the namelist treedef
carries per-case metadata the cache key deliberately ignores (the case's terrain source path, the date
clock). On the first CPU run the second member therefore recompiled the step (1330 s vs 1287 s). Blob-loaded
executables check the leaf contract (count, shapes, dtypes, HLO digest) and are shared.

## Resident batched members (design, not implemented)

The next step after one-process sequential members is to keep all N members resident on the device and
advance them together:

* **Batched stepping.** `GPUWRF_BATCH_ENSEMBLE=B` (v0.23) already runs B same-geometry forecasts under one
  outer `jax.vmap`; it is not qualified under the v0.4 release defaults (REAL carry, Pallas kernels), has no
  restart (`B=1` only) and is a separate program (its own compile). Qualification = byte-identity of each
  lane vs its single-member run on CPU and GPU, then a throughput A/B vs the sequential driver (d01 120×70 is
  launch-bound on the RTX 5090, so batching is the lever; [U] until measured).
* **Resident analysis seam.** With members resident, the restart seam does not need checkpoint files: pull the
  six analysed fields per member at the window end (the history writer already pulls them), run the LETKF on
  the host, push back the increments only, and continue each carry in place — no case files, no re-init.
* **Memory.** Per-member resident state is the carry (~0.2–0.5 GB for d01 at REAL carry [I]); N=20–40 fits a
  32 GB GPU before the step temporaries; the C-auto pool must be sized for B lanes.
