# v45-graupel GPU preregistration

Scientific question: does WRF mp8's retained entry-graupel sedimentation gate
remove trace erasure on the actual GPU, and does it resolve the retained WN3
0227 d03 hour-72 event? The CPU operator witness establishes a real omission;
it does not establish the cause of that coupled event.

1. **Component, cap 300 s, one official lock, own 8 GiB/no-swap scope.**
   Immutable candidate source; 188-column fixture SHA
   `10663ff5e1c79fda1860bfbd0009f72d3beef904d1bd7df7b57b48fe102dfcc7`.
   Native full-column GPU kernel, dt=6 s, 44 levels, release Thompson flags;
   assert `jax.devices()[0].platform == 'gpu'` inside the process.
   Candidate: zero QGRAUP zero/nonzero mismatches against pristine WRF,
   existing MIX increment bounds and finite checks unchanged.
   Gate-deletion baseline must reproduce at least one trace erasure.
   Original CPU-WRF outputs are the fidelity reference; deletion is a
   causal/regression control only.
2. **Gate cost in the same component process.** Tile the explicit real-structured
   fixture to 31,239 columns, compile candidate/deletion separately, warm both,
   measure ABBA, 20 synchronized device calls per block, no host array reads
   inside timing. Record samples/medians, compile readiness, executable memory
   analysis and device memory stats. >10% median regression is HOLD pending
   attribution. This is kernel gate cost, not forecast speed/S1 evidence.
   Private candidate/deletion JAX caches plus private Triton/CUDA/bootstrap dirs
   below `<USER_HOME>/wrf_gpu2_lanes/v45-graupel/GPU_component_gpu`; caches <=1 GiB,
   total lane usage including one measured 1.1 GiB source archive <=2.5 GiB.
   E41 runs on CPU after lock release on both real JAX caches, pattern
   `jit_graupel_*`; inspect verdict JSON (rc 2 can mean INCOMPLETE).
   Keep compact receipts and remove exact owned caches after the last consumer.
3. **Coupled attribution, manager long-window arm.** Use the final reviewed
   composed candidate, frozen source manifest and manager's release launcher,
   original WN3 0227 3-nest 72 h inputs (start f006, dt54/18/6, Thompson/RRTMG/
   MYNN/Noah-MP/KF d01). Cold setup separate, <=60 min, E41 after cold compile;
   warm N72 cap 1200 s, expected ~8 min, one official lease/lock. No borrowing an
   older executable under the new source identity. Compare all 73 frames/domain
   against the original CPU-WRF through admitted facade copies. Frozen D6 and
   integrity detector unchanged; QGRAUP must pass without the old exception.
   Compare same-session release control for s/fc-h, VRAM and RSS (>10% STOP).
   Preserve h60–72 d03 until attribution is complete. RW1/RW2/RW4 on the composed
   release remain manager gates. Required run disk (~23 GiB output + reference
   staging + caches) needs a manager DISK-PLAN; the lane will not allocate it on
   root or start the coupled arm independently.

The component launcher is `component.sh`; it requires `GRAUPEL_TREE` pointing
to an immutable archive, manager window timestamps and the lane lease. In
`night_seq_v9 SEQ_SMOKE=1` it uses a private flock and CPU interpreter with the
same fixture and semantic checker; GPU gates are NOT-APPLICABLE. Smoke does
not authorize GPU execution. Manager must provide the timing core lease and
quiet-core admission (E251) before any measured GPU arm.
