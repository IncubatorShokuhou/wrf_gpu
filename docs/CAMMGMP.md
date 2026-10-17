# CAM Morrison–Gettelman (`mp_physics=11`) — experimental GPU

**v0.4.0: stock-Triton native route merged (batch45).** GPU mp11
requires `triton==3.6.*` and a visible supported CUDA device; compiler/device
preflight fails explicitly. The old c89 GPU attempt remains failed RAINNC D6
with15/25frames and DLF2 missing. A later9a404 source completes25frames with
zero bounded misses and NF0, but raw D6 is INCOMPLETE for DLF2. Its narrowly
qualified WRF-zero writer is now merged; c89
now has a registered-source24h D6/inventory/NF0 PASS on2ed8c04c26, with
native E41 UNKNOWN. The separate offline STACK1240B FAIL remains disclosed;
XLA glue-only PASS is not whole-native scan coverage. See the [support matrix](release/SUPPORT_MATRIX_v0.4.0.md),
[route/artifact/CPU proofs](../proofs/t_cammgws/README.md) and
[batch45 integration](../proofs/s2_integrate/batch45/QUALIFICATION.json).

The experimental nonchemistry port includes the original WRF CAMMGMP driver,
macrophysics, prescribed modal aerosol activation and the complete MG core.
Public inputs, outputs and held histories are REAL4. CAM's internal `r8`
procedures remain the explicit WRF DOUBLE island.

The pristine WRF configuration checker imposes **no mandatory mp11/bl9/ra3
pairing**. The microphysics driver warns and zeros deep convection operands
when `cu_physics != 7`, and shallow operands when `shcu_physics != 2`.
The port preserves that behavior. Standalone shallow convection is not yet
operationally ported; requesting its nonzero selector remains fail-closed.
Existing qualification limits of other selected schemes still apply.

`accum_mode`, `aitken_mode` and `coarse_mode` are bound by both loaders, with
WRF defaults of 1e9, 3e8 and 2e5. The initialized modal namespaces and all
selected CAM constants participate in the cheap/AOT executable key.

The Registry activates vapor, cloud liquid/ice, rain/snow and four number
scalars (`QNCLOUD`, `QNICE`, `QNRAIN`, `QNSNOW`). Old RH/cloud histories and
turbulent work fields are restart state; the normal CAM-MG history additions
are `CLDFRA_MP`, `CLDFRA_MP_ALL`, `CLDFRA_CONV`, `CLDFRAI`, `CLDFRAL`, `LRADIUS`
and `IRADIUS`. `QNDROP` is a driver work argument, not an mp11 history scalar.
Restart readers refuse missing, mismatched or nonfinite held state.

CPU qualification: all 52 public fields are byte-identical to the original
no-FMA WRF caller over 48 calls spanning warm/cold/mixed phase, prescribed
aerosols and convection/zeroing regimes. All 27 modified native outputs are
inside an independent enclosure from 116,256 original WRF calls. The control
plan was frozen before candidate evaluation; it perturbs individual REAL4
inputs by one ULP and measures the original off/on-FMA spread with identical
initialized namespaces. See `proofs/s2_cammg/CONTROL_PLAN.json` and fixtures.

The GPU path uses one Pallas program per column, including vertical and
adaptive loops. CPU interpretation measured 241,600 temporary bytes for a
24-level column and 46.7–65.4 s compilation on one lane core. Those are CPU
artifact measurements, not GPU speed or VRAM claims. The original GPU realization produced wrong precipitation despite finite
outputs. The merged route uses stock Triton3.6.x through jaxlib's kernel call. Its exact
PTX offline artifact has STACK1240B (E41 FAIL),255registers and4 leaf callees;
an XLA glue-only PASS does not scan this native kernel. First launch assembles
PTX per fresh process (~26s per shape [M]); CPU artifact timings do not
establish whole-forecast GPU performance. The registered-source c89 full24h scientific gate now passes; native
resource coverage remains UNKNOWN. [Current source/receipts](../proofs/r_docs040/RELEASE_CHECKPOINT.json). [Exact artifact and reload scope](../proofs/t_cammgws/README.md).

The real Swiss CPU continuation (42×42×44, dt18 s, BL5/RRTMG4/Noah-MP4)
completed two corrected steps with zero nonfinite values, original fatal events or guard
events, and a byte-identical full restart. Both current PBL-to-microphysics handoffs were byte-identical. Peak CPU RSS was4.39 GiB under a
6 GiB/no-swap scope; CPU-only RRTMG tiles128 and phase cache clearing kept the
proof bounded. Both frozen pre-F20 default baseline hashes remain unchanged at that source.
The evidence packet is `proofs/s2_cammg/QUALIFICATION.json`.

The PBL handoff uses WRF's shared `EXCH_H` directly as `KVH3D`. The current
PBL call runs at physics entry; its coefficients and shared `TKE_PBL` survive
RK unchanged and reach microphysics on that same own-step. No coefficient is
recomputed from the transported state. The pristine call audit is recorded in
`proofs/s2_cammg/PBL_CALLER_AUDIT.json` for every admitted PBL. In particular,
MYNN writes `EXCH_H` but is not passed shared `TKE_PBL`; YSU also leaves that
TKE field held. A blanket conversion from `QKE` is incorrect. Unwritten top
memory and fields without a producer retain their input/restart values.

Shin-Hong (`bl_pbl_physics=11`) binds `shinhong_tke_diag` for all microphysics
selections and both surface paths. Registry default 0 holds shared TKE;
cold initialization writes original REAL4 `EPSQ2L/2=0.005` on mass levels.
Mode 1 forwards the diagnosed TKE. Lane t-shinhong corrected the diagnostic
(interface indices, TTEND Exner factor, mixlen cloud water, CORF) and the
caller (QC3D/QI3D/RQCBLTEN/RQIBLTEN, pblflg-gated scale factors); the
op-by-op REAL4 kernel is byte-identical to pristine WRF on 108 column-mode
cases (`proofs/t_shinhong/README.md`), so the former v0.4.0 caveat is closed. CAM-MG
receives the held/diagnosed shared profile across RK, with no QKE substitution
or adapter input floor. Caller modes, cold initialization, binding and restart
retention are covered; the old mode-0 caller fails.

The correction reuses the 20 already allocated CAM fields and the existing PBL
outputs. It adds no PBL solve, persistent array, host transfer or synchronization.
At the Swiss shape each existing REAL interface field is317,520 bytes; MYNN
now retains one current diffusivity profile in that carry. Legacy PBL precision
wrappers preserve CAM REAL carry without widening it. GPU output-store, layout,
launch and traffic costs remain unmeasured; the changed whole-step lowering
and CPU continuation are checked before resubmission.
