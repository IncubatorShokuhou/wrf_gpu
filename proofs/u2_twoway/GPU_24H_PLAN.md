# Pending two-way GPU qualification

Status: PREPARED, NOT RUN. This lane's wave contract prohibits GPU execution.
Original CPU-WRF is the sole numerical reference. The real WN3 short CPU gate
passes two parent/six child steps over ten frames (312 registered bounds,
zero nonfinite values/guards). This intermediate crop does not qualify the
full 24-hour GPU forecast; that KI-11 gate remains open. The CPU receipt is
`proofs/u2_twoway/QUALIFICATION.json`.

1. Merge the reviewed candidate and freeze its commit. Use a detached snapshot
   with `data/wrf_pristine` pointing to the pristine source. Reuse the operator
   receipts only with exact source identity. Record actual backend, source,
   executable, input, namelist, cadence and output-field hashes per arm.
2. Stage an owner-selected WN3 two-domain real case for **both** original CPU-WRF
   and GPU: `feedback=1`, odd ratio 3, full 24 hours, identical timestep,
   `spec_zone`, physics, input inheritance and output cadence. Current admission
   requires B=1, Thompson8/MYNN5/MYNN-SL5/Noah4/RRTMG4-4, cu0/1,
   hypsometric_opt2, sst_update0 and inactive urban/lake/nudging/advected TKE.
   Use the full original WN3 two-domain geometry for this 24-hour arm; the
   short Tenerife child crop is intermediate CPU evidence only. Use each of
   `smooth_option=0/1/2` in its own matched reference/forecast arm; label the
   designated release configuration explicitly. Do not substitute WN2 PROD for
   release fidelity. The short CPU tool demonstrates how to preserve inputs and
   copy a dump-free original executable. Full references need 24-hour namelists.
3. Wait for the manager's GPU window and valid lease. Source
   `<USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env`, then run every GPU
   command through `scripts/with_gpu_lock.sh --label u2-twoway -- ...` with a
   manager-approved measured-cost timeout. Assert the measured process actually
   uses `jax.devices()[0].platform == 'gpu'`. Declare and clean lane-owned caches
   after the last dependent arm. Respect QUIET and day/night CPU grants.
4. Run `gpuwrf run --input-dir CASE --output-dir OUT --proof-dir PROOFS --hours 24
   --max-dom 2 --feedback` using the frozen CLI. All paths must be in the lane's
   artifact directory. This command is a recipe, not authorization to run GPU.
   Scan the actual compiled GPU executable with the full `scripts/e41` tools and
   retain immutable program/cubin identity. Defaults keep hard E41 limits.
5. Apply the unchanged D6 manifest to every hour/domain, and separately score
   feedback overlap, overlap edges and parent exterior for U/V/W/T, moisture,
   active number/scalar fields, PH/MU, PHB/MUB and diagnostic pressure. Reject
   non-finite values before statistics. Report missing or unsupported auxiliary
   fields explicitly. Audit pre-mask F2 counts and the complete same-work manifest.
6. Measure whole-run wall, first/warm segments, feedback cost, device operations,
   transfers, peak VRAM/live memory and cached readiness. In particular, feedback
   currently disables fused cascades. Leaf-local filtering and packed auxiliary
   families fuse their own work; actual device launch/traffic costs are pending.
   No GPU speed claim is supported by CPU operator timing. Compare
   additional cost against the same real forecast with feedback off as a
   performance/regression control, never as scientific truth. Preserve restart
   compatibility checks for updated resident PB/PHB/MUB/h_diabatic/precipitation.

The final reviewer must see complete receipts for each declared arm. A finite
run, passing operator test, or matching default jaxpr cannot substitute for the
24-hour original-WRF comparison.
