# PORT_PLAN v1 critique — w5-portplan-astra

**Verdict: CRITIQUE_SHARPEN_NATIVE_RUNTIME_AND_EVIDENCE.** Retain Pallas–Triton plus a narrow backend semantic layer. Revise the execution gates, support claims and schedule before implementation dispatch. The census supports trying reuse; it does not yet establish a cheap, fast, production-qualified port.

Reviewed the v1 announced at 2026-10-17T11:46:20Z, after the manager's 11:47Z instruction. Frozen review copies: plan, SHA256 `f7582fedf087ec1ff340f6136b5b3ca95ce1faef0171a5bbc77ac08121c92799`; inventory, SHA256 `00c53fc1a6d7ad63228f791b96064d8f35bb559e6016efc10b33fc823a5afe34`. Line references below refer to that plan copy. Source inspection tree: `57974759229d52405f2b2a6f4d39ffa25117d3ad`.

Independent view was written before reading the plan; static inventory independently found 72 factory sites in 50 files outside the facade. This review used CPU source/receipt inspection and primary-source web checks, with no GPU, model run, implementation or repeat numerical validation. [M] denotes an identified source observation or existing measurement; [I] inference/estimate; [U] unmeasured. Pending author compiles are acknowledged and can be folded into v2 without delaying this critique.

## What to retain

- The common Pallas algorithms, retained JAX/State/IO interfaces, explicit vendor arithmetic, and separate native-capability versus NVIDIA-instruction predicates are the right starting point.
- W5P01 is useful compiler-feasibility evidence, especially the diagnosed `reduce_precision` failures and exposed register/compile risks. W5P02 is useful evidence that the chosen AMD instruction spelling can prevent contraction in the tested stock compiler.
- Prefer A1 where it passes; keep A2 as a measured per-kernel escape. Preserve WRF DOUBLE islands, integer RNG exactness and original-WRF scientific references.
- The newly identified Pallas–Triton deprecation is real and deserves a pinned-stack policy. It is confirmed by the [JAX Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html).

## Required changes to v2, in priority order

### 1. Make device control flow and readiness an early acceptance gate

**Plan:** lines 109, 347 and risk R3/R4 acknowledge command buffers/readiness, but M1 passes on a finite three-hour PROD run. That permits a path with host-driven predicates/transfers and unacceptable readiness, contrary to the kernel's operational requirements.

**Evidence:** [M, upstream source] XLA commit `65ad34a5303043439047292f11d0e0d78c364f47` still returns `UnimplementedError` from ROCm conditional-node and while-condition-node methods. [Pinned source](https://github.com/openxla/xla/blob/65ad34a5303043439047292f11d0e0d78c364f47/xla/stream_executor/rocm/rocm_command_buffer.cc#L104), retrieval/hash receipt. This is not an exact-wheel execution result, but HIP graph availability does not establish CUDA CONDITIONAL/WHILE parity. The project currently skips its command-buffer policy off CUDA: xla_autotune.py.

**Change:** before a broad geometry sweep, execute one composed ordinary/radiation/KF cadence sequence on an exact ROCm tuple. Assert native dispatch and actual device; count transfers, synchronizations, launches and allocations within the timestep; test fixed-trip acoustic loops and branch predicates; measure fresh-process cached readiness and memory. Any required static scheduling, graph segmentation/unrolling or backend runtime work must have an owner and budget. Preserve cadence, buffer lifetime and numerical semantics. Profile first; do not assume a whole-step rewrite is necessary. A finite forecast alone cannot pass this gate.

### 2. Bound the census claims to the source and compiler actually inspected

**Plan:** lines 16–23 call the census evidence that the port is cheap; line 178 concludes no A19-class cliff; line 187 asserts register use is shape-independent; M0/§3.1 treat the census as an E41 analogue.

**Evidence:** [M source] the author's capture tool forces `interpret=True`, traces with CPU-selected backend branches, then lowers the resulting jaxpr for both target names. Lowering platform does not retroactively select the CUDA-native Python branches (the existing E115 trap). Thus 46/79 identical TTIR pairs are a valid observation about these captured bodies, not a comparison of the actual production CUDA bodies with the proposed post-M2 ROCm bodies. Compile tool uses **stock Triton**; A1 uses **XLA-bundled Triton**. V1 correctly acknowledges this at line 199, but its stronger acceptance claims do not respect that distinction. CAM-MG is direct evidence that the two pipelines can differ materially: custom compiler route.

There are also actual open cases: Noah canopy times out in v1, and the census includes a legacy implicit-W kernel. A metadata report of zero scratch in completed isolated kernels cannot exclude whole-program fusion/layout cliffs, the uncompiled kernel, or runtime latency. SGPR spills into VGPR lanes are not a measured zero-cost event. Register use is not generally shape-independent: the MYNN solve unrolls `nz`, and padding, constant bounds and layout choices can change generated code.

**Change:** keep a diagnostic census with complete denominators: compiled / failed / timed out / not attempted, including the two lowering failures. Separately inventory the final **executing** A1/A2 binaries at real WN3 shapes, flags, layouts and compiler identities; bind each binary to its gate result. Carry separate SGPR-spill, VGPR-spill, scratch and occupancy fields. Replace the ambiguous “spills = 0” requirement at line 291 with an explicitly approved backend-specific criterion; preserve the existing scientific/resource gates. Do not call an offline stock artifact the actual executable's E41 result. Retest only changed artifacts and the final composed candidate.

### 3. Supply the missing pre-Ampere execution route and fix milestone dependencies

**Plan:** §2(f)/M5 propose the same kernels plus a CUDA-12 wheel; M5 is two lane-days. M0 requires a green census before M2 fixes known lowering failures, and M1 requires native PROD before those boundary fixes.

**Evidence:** [M source/receipt] gpu_preflight.py explains XLA's sm_80 minimum. The actual GTX1080Ti receipt used **JAX CUDA-12** and still hit that Triton gate. `GPUWRF_ALLOW_UNSUPPORTED_GPU` changes admission, not kernel lowering. Stock Triton/ptxas success below sm_80 is promising, but it bypasses the default XLA compiler that refuses the product program.

**Change:** make M5 conditional on a real stock/custom-call route below sm_80, with target selection, buffer aliases, stream/capture behavior, cache/reload and original-WRF qualification. Reuse the existing 1080Ti box for the first authorised probe; the plan need not assume another rental. Treat two days as a feasibility budget until this path works. Separate M0's discovery census from its later acceptance gate. The minimal capability/RN implementation precedes native composed-step M1; M3 and final M4 follow that gate. Show the dependency graph instead of deriving elapsed time from total lane-days divided by agent count.

### 4. Remove precision conclusions that the CPU probes do not establish

**Plan:** §2.1 says OCML/libdevice differ by about 1–2 ULP and are “D6-irrelevant,” AMD division is WRF-closer, and similar advertised FP64 rates imply no new cost. The MYNN inventory says no vendor-specific constructs/reuse as-is.

**Evidence:** [M source] W5P02 compiles `a*b+c` and `a/b+c` and counts instructions; it executes no numerical inputs. This establishes code-generation structure, not complete IEEE edge-case behavior or forecast fidelity. MYNN tridiag imports `_rn` from `dyn_rk_fp32` and uses it throughout the recurrence: its apparent lack of local asm does not make it independent of vendor arithmetic. The source's non-CUDA `_rn` is ordinary arithmetic today.

**Change:** retain W5P02 with its CPU/compiler scope. Require the final primitive/compiler path to meet frozen WRF contracts, including cancellation, ties, signed zero, normals/subnormals and active wet/cold cases; keep exact integer streams and existing exact REAL component checks. Keep end-to-end cross-vendor byte equality out of the release requirement. Resolve the inconsistency between D2's “bitwise only” rule and §3's correct literal-WRF tridiag2 bitwise gate. Remove the unproved 1–2 ULP/D6 and FP64 cost assurances. A better local division approximation does not establish a closer coupled forecast. Map imported arithmetic helpers to every consuming family.

### 5. Separate historical performance, projections and current acceptance numbers

**Plan:** §0 uses 3.6 s/fc-h PROD and 7.4 s/sim-h WN3 as today's baselines, then projects 18–31× CPU speedups; §1/§3 use BP83 shares and 0.6–0.9 AMD efficiency. The new explanation assumes a fixed host gap and says concurrent cases hide it.

**Evidence:** [M/I historical] `git show 925b22dda2:tests/v025/b_phys/BP83/ROOFLINE_RC3.md` explicitly describes **BP82 RC2 measured roots mapped to RC3 HLO, ranking only**. It samples three roots per class and says its modeled byte floor is not a DRAM roofline certificate. Main INBOX's CP147 row (2026-10-05T10:28Z) supports the historical 3.604458 whole-run number. The inventory's WN3 value is **hour-2 7.686**, different from the prose's 7.4 and not whole-run S1. The v0.4 release notes explicitly make no new S1/roofline/energy claim. Family labels also do not by themselves prove the asserted 75% Pallas share: several families contain both kernels and XLA glue.

**Change:** retain BP83 as historical ranking with those qualifications. Resolve and cite the WN3 receipt and units. Call device ratios **planning scenarios**, give the named hardware denominator and assumptions, and leave current end-to-end AMD speed [U]. Do not multiply an unmatched historical GPU baseline into a release CPU speedup claim. Use a fresh or reusable exact-source cadence-weighted profile to set actual targets. The ROCm host gap can grow; concurrent cases only hide it if CPU scheduling, memory, streams and runtime contention permit. Measure both B=1 wall and maximum-resident-case throughput, with an equally declared CUDA comparison.

The proposed ranges are not disproved by this review; they are too weakly constrained to serve as M3 acceptance thresholds. Static occupancy is not an empirical AMD efficiency coefficient.

### 6. Make hardware/scheme support and the rental dependency explicit

**Plan:** D1 lists gfx942/gfx950/gfx1100/1101/1200/1201 as supported, while M3/M4 provide gfx942/gfx1100 hardware only. RDNA4 access is absent and purchase is optional. §3.2 claims RX7900XTX rental from DataCrunch at $0.39/h.

**Evidence:** [W] the current [DataCrunch redirect/Verda price list](https://verda.com/pricing) lists NVIDIA instances and gives no RX7900XTX offer. I found no primary source for that quoted consumer-AMD rental. This is **unverified availability**, not proof that no private offer could exist. [DigitalOcean's MI300X $2.59/h](https://docs.digitalocean.com/products/droplets/details/pricing/) and [Runpod's advertised from-$2.39/h](https://www.runpod.io/gpu-models/mi300x) are supported by primary sources. Neither establishes available capacity or profiling privileges.

**Change:** obtain an actual RDNA quote, borrowed machine or approved purchase before making that schedule/budget concrete. The $250–450 figure can remain a conditional compute allowance, with RDNA access and other charges separate [U]. Verify exact image existence and pin its digest plus wheel/compiler hashes; the proposed container tag alone is not a reproduced environment. Require hardware counter/profiler access and include cache storage and teardown in the rental plan. No rental is requested or made by this critique.

Publish a backend × architecture × scheme matrix with native/codegen-tested/hardware-qualified/unqualified states. MI300X does not qualify RDNA4, and gfx1100 does not qualify gfx1101 automatically. Unsupported schemes should fail clearly or be explicitly experimental, with an honest execution-path disclosure. The 19-module “options” bucket needs individual evidence: several have existing 24 h CUDA D6 qualifications, with separate E41 disclosures in the v0.4 matrix. A generic new-backend tier must not imply those CUDA receipts transfer.

### 7. Treat M6 as a compatibility project, not demonstrated future-proofing

The deprecation is a good finding. Pin the **exact validated JAX/jaxlib/plugin/compiler tuple** now, rather than treating all 0.10.x combinations as equivalent. Vendoring `lowering.py` and `primitives.py` may be useful, but the installed lowering imports private `state`, `indexing`, `literals`, `GridMapping`, tracing and MLIR APIs throughout its first 60 lines. Generic-form MLIR does not preserve those frontend contracts or the launch ABI.

**Change:** give M6 a bounded compatibility spike on a named next JAX version: one real column, one stencil, one custom primitive, with aliasing, compile/cache reload and device execution. Keep the 6–10 LD estimate as a conditional prototype estimate until that succeeds. Do not switch all AMD kernels to A2 merely for uniformity; qualify its capture, compile/readiness and memory consequences first. A pinned compiler/runtime bridge is substantial engineering even when the science bodies are reused.

Intel likewise needs more than an Intel-XPU Triton emitter: [oneAPI JAX 0.11.2 alpha](https://github.com/Intel-tensorflow/jax/releases/tag/oneapi-jax-v0.11.2-alpha) excludes Pallas and reports executable serialization failures. Native launch integration and cached readiness belong in its assessment. This is compatible with Intel remaining assessment-only for v0.5.0.

## Better execution plan and estimate treatment

Use the proposed semantic layer, but select implementations by a **qualified capability tuple**, not just `vendor in {cuda, rocm}`. Keep kernel-native capability, arithmetic semantics, control flow, AOT support and scheme eligibility distinct. Existing `cuda_native()` adapters and direct Pallas imports must be included; the author's inventory already identifies much of this correctly.

Recommended order:

1. Pin source, tuple and hardware access; keep the census as discovery. Bind every failure/pending case to an owner without pretending CPU compilation is hardware qualification.
2. Implement the minimum shared semantic/dispatch changes and prove a composed native timestep/cadence sequence on MI300X. This is the architecture feasibility gate, including transfer/control-flow and cached-load evidence.
3. Validate the same design on a real RDNA target early. Then tune only measured bottlenecks, with real layouts and final source specializations.
4. Run original-WRF component and final composed WN3 qualification for every advertised scope, followed by S1/memory/scaling/readiness/restart evidence. Carry forward proven unchanged CUDA validation; do not rerun 120 numerical tests and every compiler target on every unrelated commit.
5. Pursue older NVIDIA only through the qualified alternative compiler/runtime route. Keep M6/Intel as separately bounded work until feasibility is demonstrated.

**Effort [I]:** 18.5 LD can remain an optimistic kernel-reuse base, not a commitment to all listed support tiers in 7–9 days. I would show an explicit **3–8 LD contingency for control-flow/readiness integration** and, if M5 is retained, **3–6 LD for a pre-Ampere launch/cache/capture route beyond the wheel/smoke task**. That yields a conditional **24.5–32.5 LD planning envelope**, excluding reviews, M6, hardware waiting and substantial optional-scheme/compiler repairs. These are risk reserves, not measured effort or a hard upper bound. Re-estimate after step 2; a failed runtime feasibility gate can change the architecture, not merely add a day. M6's 6–10 LD remains separate and conditional. Do not add the family table's LD a second time; v1 correctly says those are included.

**Missing alternatives:** keep selective fused JAX/XLA for pointwise/stencil families where it preserves equations and passes measured performance/fidelity gates. Legacy column-loop blowups do not rule out every tuned XLA family implementation, nor prove every alternative requires a second scientific model. Keep bounded HIP/FFI escapes for demonstrated blockers. The arbitrary two-family cap and “50% of CUDA roofline fraction” rule should not replace actual measured critical-path evidence; BP83's modeled floor is explicitly unsuitable as that certificate. A new whole-model DSL remains unjustified.

## Validation closure checklist for the revised plan

- Original CPU-WRF/pristine oracles remain truth. Preserve frozen component contracts, including exact REAL contracts where present, integer RNG, all required fields/leads, and pre-mask F2 accounting. “Interpreter route is correct” at line 27 should become “component/short-smoke qualified on its documented scope”; its existing two CPU steps are not a trajectory certificate.
- Use the manager's pinned **full WN3 release case set** and manifests. PROD is iteration/debug evidence under current AGENTS; do not label the WN2 path a release fidelity claim. Avoid hard-coding an old field denominator without binding it to the actual manifest and scheme set.
- Pre-register actual-binary coverage and backend resource criteria. Preserve the original E41 obligations and explicitly define approved AMD counterparts; a zero-scratch test kernel is insufficient. Include boundaries, padded lanes, alias ownership and whole-step layout/copy costs.
- Add the currently omitted R2 memory-scaling gate (α ≤ 1.10 on three sizes), R3 cold/cached readiness bars (including cached readiness ≤ 60 s), crash-safe restart/AOT reload, S3/transfer evidence, measured S4 roofline/residual evidence and the required station/long-run scope. Report true peak memory, not just an allocator pool or compiler temporary estimate.
- Preserve the CPU-only pytest default and official GPU lock/lease requirements. Prefer the existing dedicated device-runner pattern over weakening the global `tests/v025` CPU guard; assert the actual vendor/device inside every measured process.
- Numerical validation occurs once on the final non-bit-identical composed candidate; unchanged exact-source/native artifacts reuse their receipts. CPU interpretation, offline codegen, native component fidelity and full forecast gates remain distinct.
- Correct smaller source claims: the [Mosaic API](https://docs.jax.dev/en/latest/jax.experimental.pallas.mosaic_gpu.html) explicitly names Ampere/Hopper/Blackwell, so generic sm_120 backend support is not simply unknown (our WRF kernels there remain unqualified); [Apple's JAX page](https://developer.apple.com/metal/jax/) documents a plugin but unsupported float64. The float64 limitation is enough for the Apple assessment decision; “no maintained JAX GPU” is not established. Add direct primary-source links/dates for consequential [W] claims.

This is one critique of the original plan. It authorises no implementation, GPU run, cloud spending, gate waiver or support announcement. The author should incorporate these corrections into v2; this lane stops after the CRITIQUE READY handoff.
