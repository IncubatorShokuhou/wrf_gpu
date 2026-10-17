MP52 cold-compile attribution (CPU deviceless; qualification open)

The retained c23 GPU setup exceeded 1,800 seconds without a stepping
executable. Its captured CUDA target is
`<USER_HOME>/wrf_gpu2_lanes/v-prep/WN3-smoke/attempts/disk_relocation_1433/target_config.pbtxt`,
SHA256 `f46c185780fdbd4aab62ef20293385bb8de31a71bd260e9ce62cbfc21f196a23`.
All probes below disable GPU visibility and use `JAX_PLATFORMS=cpu`, CUDA
SM120/warp32 compilation, one private cache and a bounded CPU RAM scope.
These are compiler diagnostics, not GPU numerical or speed qualification.

The native P3 wrapper uses 44 physical levels padded to 64 and two ice
categories for mp52. The generated process block expands Python category
loops while tracing; its retained process-only IR has 49,339 lines,
1,646 `tt.reduce` operations and no `scf.while` or `scf.for`. The complete
raw IR has 64,665 lines / 3,826,882 bytes / 2,006 reductions. This describes
the expanded dependency graph; it does not establish a complexity law or
prove that category count alone causes the stall.

| Probe | What changed | Measured result |
|---|---|---|
| TP03/TP05 | Raw native mp52, original source | Lower 2.874 s; compiler dump stops before `TritonGPUCoalesce`; no cubin within 300 s |
| TP06 | Static category extraction with `tt.split` | IR 3.109 MB; still exceeds 300 s; candidate reverted (`4f6e394a9`) |
| TP07b | Normal TTIR simplification passes | 0.224 s; IR 2.467 MB / 884 reductions; coalescing still exceeds 300 s |
| TP08 | Keep only INITIAL phase, same output selection | Cubin in 1.669 s; STACK 0 B, REG 40, LOCAL 0, no non-leaf callee |
| TP09 | Keep INITIAL + PROCESSES, before sedimentation | IR 2.919 MB; coalescing exceeds 175 s; own process terminated |
| TP10 | Diagnostic zero ice-ice lookup | IR 2.200 MB; coalescing exceeds 160 s; invalid physics, discarded |
| TP11 | Diagnostic constant category destination | IR 2.898 MB; exceeds 180 s; invalid physics, discarded |

Thus the slow isolated compilation is already present in the process phase;
sedimentation is not necessary. Neither the category-extraction rewrite nor
normal simplification removed it. No production optimization is retained.
E157 records the same pass stalling on other expanded Pallas column graphs.

TP12 separately lowered the actual whole Swiss mp52 program for the captured
CUDA target on CPU: initialization 23.580 s, lowering 31.620 s, serialized
MLIR 56,711,415 bytes, with actual compiler options preserved. TP13 armed a
900-second whole-program CompileOnly bound and produced no executable. The
reboot interrupted receipt recovery; its completed exit/time is unknown.
It does not prove that the isolated pass is the only whole-program blocker.

A bounded GDB stack sample of the exact original native TTIR is prepared as
TP24. No stack sample has completed yet. A final compiler-mechanism finding
will be appended after that probe; no successful mp52 executable, E41 pass,
D6 score or native numerical result is claimed by this report.

Retained artifacts under `<USER_HOME>/wrf_gpu2_lanes/t-p3fix/`:
TP03_mp52, TP07b_mp52, TP08_initial, TP09_processes, TP10_no_ice_ice,
TP11_no_destination, TP12_whole and TP37_ir_inventory.json. Disposable
compile caches are deleted after their final dependent arm; original raw
IR, compact reports and failed-probe evidence are retained.

TP24b completed the bounded sample on the actual whole-program embedded
kernel. JAX's matching dialect context decoded the sealed 937,406-byte
bytecode (SHA a4bea638...) to 3,827,016 bytes of TTIR (SHA 3a20a1a0...).
The standalone Triton bytecode parser initially rejected an attribute type;
printing through JAX's matching context fixed only this diagnostic decoding.
No scientific pass or IR edit was applied.

At the 45-second interrupt, 35 sampled frames are recursive
`getForwardSliceImpl`, below `mlir::getForwardSlice` and `mlir::getSlice`.
The caller is `buildCoalescedEncoding` at
`lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:44`, called by
`CoalescePass::runOnOperation` at `Coalesce.cpp:101/86`. This locates the
slow native compiler work in dependency-slice construction during coalescing.
The expanded category/process graph is the measured input. A precise
asymptotic cost or a category-count-only cause is not established.

The actual imported compiler is Triton 3.6.0 from
`<USER_HOME>/.local/lib/python3.13/site-packages/triton`, libtriton SHA256
`23dd9f5e3808ba6e244278640f61c48bd46a2b14eb5ed3e33f8af6bad5bba72c`.
An earlier inspection of the separate miniconda installation's package
metadata was not the loaded compiler; it is not used as provenance here.
The real target hash remains f46c1857..., and no cubin was produced.
Full machine-readable scope, decoded-input hashes and the raw stack are in
reference_control/mp52_compiler_stack.json and mp52_gdb_stack.log.

Disposition: unresolved compile blocker for mp52, with no production change
retained. A compiler slice/graph construction fix or a measured smaller
kernel decomposition needs fresh compile/resources and GPU validation;
no such unmeasured redesign is included in this reference repair.
