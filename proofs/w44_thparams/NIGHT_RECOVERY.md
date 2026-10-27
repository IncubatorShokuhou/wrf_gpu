# Night launcher recovery, 2026-10-24

The native Thompson component passed on the actual GPU. The root arm failed
because the harness selected different work from the product and put OFF and
ON cold preparation inside one 700-second child timeout.

Measured evidence is under `<USER_HOME>/wrf_gpu2_lanes/w44-thparams/GPU_FAIL/`.
The original immutable source `2e66785c50`, v2 manifest, cache and receipts stay
untouched. No GPU run was launched during this recovery.

| Phase | Measured result |
|---|---|
| Native component first calls | OFF 9.214 s; ON 9.090 s |
| Component science | ON default 20/20 leaves BYTE; 5/5 changed-value original-WRF gates pass; all five omissions killed |
| Component event probes | All seven arms: input nonfinite, inadmissible and pre-mask nonfinite counts zero |
| OFF fused-root compile | Backend alarm 201.789 s; JAX cache records 249 whole seconds for compile/load |
| After cache publication | 150.795 s until AOT file publication, before fused execution by source order |
| Root timeout | 700 s; OFF captured, ON reached lower-HLO cache miss, no completed ON executable |
| Whole sequencer step | 771 s, including component and CPU E41; exit 7 |

The 150.795-second interval contains export, metadata compaction, compiled-thunk
conversion and publication. Their separate contributions were not recorded.
It is not a measurement of Thompson compilation or forecast execution.
New durable phase journals distinguish loading, lowering, compilation and
export, preserving the current phase even if TERM interrupts it.

The missing output observer changed the compiled program. Product PROD uses
history cadences d01=67 and d02=200, with d01 exact alarms 67,134,200 over three
hours, and asynchronous domain execution with a root synchronization cadence
of one. The child's 200-step output cadence is not divisible by its ratio of
three. With an output observer the product scheduler therefore uses `d01` and
`d02`; `output=None` selected `fused_d01` instead. The harness now calls the
original product scheduling helpers and supplies a non-None observer. It does
not measure output-file IO.

The retained E41 failure is real: one OFF `fused_d01` kInput fusion has 664
instructions (>500), with maximum stack 32 B and no oversized nonleaf callees.
Its attributed source frames are surface-layer/Noah-MP and coupling, with no
Thompson frames. The four native component/event executables pass E41. The
v0.4.3 release inventory contains d01, d02 and fused_d02, not this fused_d01.
Identical relevant source files do not establish identical optimized programs.
The manager accepted this as a harness-specific retained finding: normal
E201/E41 on the actual product programs decides the release gate. There is no
physics patch or E41 waiver in this lane.

Cold preparation is split into these independently scheduled entry points:

- `night_prepare_off_d01.sh`
- `night_prepare_off_d02.sh`
- `night_prepare_on_d01.sh`
- `night_prepare_on_d02.sh`

All four write one shared `gpu_v3/cache`. Each prepares one original operational
program without executing a timestep, with a 900-second GPU cap and 930-second
outer cap, also bounded by the manager's authorized window. It uses the
existing domain-spec builder and child force-down producer, then prepares
record windows exactly as the runtime does. Completed programs survive a later
arm's timeout. The old combined `prime` entry point refuses to run.

The 900-second reservation exceeds the failed arm's measured roughly
572 seconds to first OFF fused export, including initialization/lowering.
This is a conservative reservation for one program, **not a measured completion
guarantee for the corrected product programs**. Their compile/export phase
times must be reported from the next authorized run. Four maximum reservations
total 60 GPU minutes; the manager may schedule them separately.

After all four preparations, `night_check.sh` binds every OFF/ON domain receipt
to both its cheap-key and HLO files and metadata, verifies native hashes, then
runs the unchanged E201/E41 scanner after GPU release. Missing one value's
executable cannot be covered by the other value's domain name. Only then may
`night_warm.sh` run root A/B and profiling. Those separate fresh-process arms
have 300-second caps each inside a 900-second window; they must load all root
programs and create no new native programs.

CPU smoke is explicit: `THPARAMS_LAUNCH_SMOKE=1` runs the same launcher and
product child environment with the CPU device/Pallas interpretation substituted.
The mini model uses real mixed-phase columns, loads/compiles/exports native CPU
AOT, executes it, and checks byte-exact replay. Warm smoke loads the prepared
OFF/ON programs and proves Nt=50e6 changes output. It is launcher/model smoke,
not GPU timing or root fidelity. The separate scheduler smoke exercises the
actual scheduler for first/ordinary steps, both hourly child split patterns,
and three full hours; removing only the observer restores the bad fused path.

Final receipts, tool hashes, immutable source pin and remaining gates are in
the external `NIGHT_LAUNCHER_v3.json` and `GPU_FAIL/LAUNCH_SMOKE_FINAL.json`.
CPU smoke caches are disposable after the last smoke consumer. Actual v2 GPU
evidence remains retained by the manager's instruction.
