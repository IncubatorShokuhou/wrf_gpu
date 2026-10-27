# Precompile namelist variants

**The v0.4.4 variant queue and cache checks are tested on CPU. GPU priming
cost and full release validation are still pending. Deviceless compilation followed by GPU loading is not available in v0.4.4; this command does not install those artifacts.**

`gpuwrf precompile` prepares the shared executable cache for complete WRF
namelists, such as several supported microphysics/PBL combinations on one grid.
It runs a short native forecast for each uncached variant, so initialization,
nesting, stepping, radiation and output programs can enter the ordinary cache.
Use input coverage and a forecast length that exercise the programs you need.
Changing the grid or a scheme can require more compilation later.

CPU-only compilation followed by GPU loading is shelved for v0.4.4. The public
command currently queues normal GPU builds. Preparing a queue needs no GPU;
executing it needs an authorized GPU window. No default forecast option changes.

## Runnable Swiss queue example (CPU only)

From the repository root, prepare a single complete namelist variant:

```bash
mkdir -p runs
cat > runs/swiss-variants.json <<'EOF'
{"variants": [{"name": "swiss", "namelist": "../examples/switzerland_d01/namelist.input"}]}
EOF
JAX_PLATFORMS=cpu gpuwrf precompile --namelist-variants runs/swiss-variants.json \
  --input-dir examples/switzerland_d01 --cache-dir runs/swiss-compile-cache \
  --work-dir runs/swiss-precompile --hours 3
```

This writes a plan, not GPU executables. Inspect `runs/swiss-precompile/report.json`
and its `execute_argv` before scheduling execution. The relative namelist path
resolves against `runs/swiss-variants.json`, not against the input case.

Create `variants.json`; paths are relative to this JSON file:

```json
{
  "variants": [
    {"name": "mynn", "namelist": "namelist-mynn.input"},
    {"name": "ysu", "namelist": "namelist-ysu.input"}
  ]
}
```

Each entry requires exactly `name` and `namelist`. Supply complete supported
namelists, with the grid matching the input files. Names must be unique and use
letters, digits, `_` or `-`. The command validates scheme support and root
timestep alignment before running anything. For a 54 s root timestep,
use a multiple of three forecast hours.

Prepare the queue:

```bash
JAX_PLATFORMS=cpu gpuwrf precompile \
  --namelist-variants variants.json --input-dir /data/case \
  --cache-dir /data/cache --work-dir /data/precompile --hours 3
```

`/data/precompile/report.json` lists each variant as `QUEUED` and includes an
`execute_argv` command array for an operator's GPU-window scheduler. It does not start
a GPU process or populate the cache. The work, cache and input directories must
be separate. Choose disk-backed locations with enough space for transient
forecast output. Sealed release caches need a writable overlay.

In an approved window, execute the same queue through the repository wrapper:

```bash
scripts/with_gpu_lock.sh --label YOUR_LANE -- \
  env JAX_PLATFORMS=gpu gpuwrf precompile \
    --namelist-variants variants.json --input-dir /data/case \
    --cache-dir /data/cache --work-dir /data/precompile --hours 3 --execute
```

In this repository's shared development setup, the wrapper also requires a
valid GPU-use lease issued by the project coordinator. Each variant
runs in a fresh child process that inherits only the validated official lock
descriptor. The parent holds no CUDA pool. The usual CLI configures the child's
GPU allocator before backend initialization. `--timeout-seconds` defaults to
1800 per variant; a timeout or failed run is reported as `FAILED` and never
marked complete. Variant-specific logs remain in the work directory. Temporary
input views, wrfouts and scratch files are removed after each attempt.

The selected namelist is copied into a private input view as
`namelist.input`; meteorological inputs stay read-only. CPU history files are
excluded from the view so priming uses native initialization. Input files and
the packaged code/data must remain stable during a build.

Successful variants report `BUILT`; reruns with the same work directory report
`CACHED` when the completion record and every recorded executable/cache artifact
still match. The record binds the namelist, inputs (including auxiliary
streams), package code/data, dependency versions, environment and GPU target.
Missing or modified artifacts require another priming attempt. Identical
namelists under different variant names share a completion record. A first invocation
without a record runs the forecast even if some cache entries already exist;
the normal runtime reuses those entries.

`CACHED` attests this tool's completed priming work. It does not qualify a new
scheme combination against CPU-WRF or guarantee that a longer forecast never
needs a program absent from the priming segment.

## Cost, refusals and CPU-WRF comparison

This is a gpuwrf workflow extension, not a WRF namelist feature. Queue
preparation for the two-nest Canary Islands and three-nest Tenerife cases took
1.178 / 0.925 s and 359,372 KiB
peak process RSS, with zero backend initialization attempts and no executable
cache created [M]. The queue tool and its existing callers passed 41 CPU checks;
additional exact-source and cache-integrity evidence is in the
[PRECOMPILE report](../proofs/w44_precompile/PRECOMPILE.md) and
[CPU packet](../proofs/w44_precompile/receipts/W5_CPU_PACKET.json).

Execution incurs ordinary forecast initialization, compilation, stepping,
radiation and output work; a short priming run may not exercise every future
program. Per-variant GPU wall time, readiness and peak VRAM have not been
measured in these CPU checks [U]. The unchanged source leaves default model
operators/arrays unchanged; isolated CLI registration overhead is unmeasured.
Producing a native executable on CPU and checking its compiled resource use
does not establish safe GPU loading or correct execution. Do not substitute
such experimental artifacts for a qualified ordinary GPU cache.

Invalid/unsupported namelists, misaligned root durations, duplicate/invalid
variant names and overlapping input/cache/work directories refuse. Changed
inputs, tables, code or cache artifacts invalidate a completion record;
timeouts/failures never become successful cached builds. No `--execute` is
needed for a CPU queue preview.

Precompilation does not require CPU-WRF comparison output. To qualify a new
physics combination, run original CPU-WRF with the same namelist/inputs and
compare the subsequent forecast, independently of a `BUILT`/`CACHED` tool
status. Keep the normal full history; process investigations may add fields
through CPU-WRF `iofields_filename`, as listed in each physics tool page.
Return to [Tune WRF for your region](regional_optimization.md).
