# CAM Morrison-Gettelman microphysics (mp_physics=11)

**Status in v0.4.0: Validated 24 h.** Passed a 24-hour comparison with the original CPU WRF in the v0.4.0 release check, in one tested single-domain configuration. Other combinations and nested domains are not validated. It needs the Triton 3.6 compiler (`pip install "triton==3.6.*"`); the code-quality scan of its compiled GPU code could not be completed.

GPU runs with `mp_physics=11` need `triton==3.6.*` and a visible supported
CUDA device; the compiler and device checks stop with a clear error otherwise.
An earlier GPU version failed the precipitation (RAINNC) comparison and did not
write the `DLF2` history field; both are fixed in the release. See the
[support matrix](release/SUPPORT_MATRIX_v0.4.0.md),
[route, artifact and CPU proofs](../proofs/t_cammgws/README.md) and the
[integration evidence](../proofs/s2_integrate/batch45/QUALIFICATION.json).

The non-chemistry port includes the original WRF CAMMGMP driver,
macrophysics, prescribed modal aerosol activation and the complete MG core.
Public inputs, outputs and held histories are REAL4. CAM's internal `r8`
procedures remain the explicit WRF DOUBLE island.

The pristine WRF configuration checker imposes **no mandatory mp11/bl9/ra3
pairing**. The microphysics driver warns and zeros deep convection operands
when `cu_physics != 7`, and shallow operands when `shcu_physics != 2`.
The port preserves that behavior. Standalone shallow convection is not yet
operationally ported; requesting its nonzero selector remains fail-closed.
The validation status of the other selected schemes still applies.

`accum_mode`, `aitken_mode` and `coarse_mode` are bound by both loaders, with
WRF defaults of 1e9, 3e8 and 2e5. The initialized modal namespaces and all
selected CAM constants are part of the compiled-program cache key.

The Registry activates vapor, cloud liquid/ice, rain/snow and four number
scalars (`QNCLOUD`, `QNICE`, `QNRAIN`, `QNSNOW`). Old RH/cloud histories and
turbulent work fields are restart state; the normal CAM-MG history additions
are `CLDFRA_MP`, `CLDFRA_MP_ALL`, `CLDFRA_CONV`, `CLDFRAI`, `CLDFRAL`, `LRADIUS`
and `IRADIUS`. `QNDROP` is a driver work argument, not an mp11 history scalar.
Restart readers refuse missing, mismatched or nonfinite held state.

CPU checks: all 52 public fields are byte-identical to the original
no-FMA WRF caller over 48 calls spanning warm/cold/mixed phase, prescribed
aerosols and convection/zeroing regimes. All 27 modified native outputs are
inside an independent spread from 116,256 original WRF calls. The reference
plan was fixed before the port was evaluated; it changes individual REAL4
inputs by one rounding unit and measures the original spread with fused
multiply-add off and on, with identical initialized namespaces. See `proofs/s2_cammg/CONTROL_PLAN.json` and fixtures.

The GPU path uses one Pallas program per column, including vertical and
adaptive loops. CPU interpretation measured 241,600 temporary bytes for a
24-level column and 46.7–65.4 s compilation on one CPU core. Those are CPU
measurements, not GPU speed or memory claims. An earlier GPU version produced
wrong precipitation despite finite outputs. The released route uses stock
Triton 3.6.x through jaxlib's kernel call. An offline scan of its exact GPU code
(PTX) shows a 1,240-byte stack, above the project's 256-byte limit, plus 255
registers and 4 leaf callees; the scan of the surrounding compiled program does
not cover this kernel. The first launch in each new process assembles the GPU
code (about 26 s per shape, measured); the CPU timings above do not describe
whole-forecast GPU speed. The 24-hour comparison with the original CPU WRF
passed. [Current source and records](../proofs/r_docs040/RELEASE_CHECKPOINT.json). [Exact artifact and reload scope](../proofs/t_cammgws/README.md).

A CPU continuation on a real Swiss case (42×42×44, dt 18 s, BL5/RRTMG4/Noah-MP4)
completed two steps with zero nonfinite values, WRF fatal events or guard
events, and a byte-identical full restart. Both PBL-to-microphysics handoffs
were byte-identical. Peak CPU memory was 4.39 GiB; a CPU-only RRTMG tile size of
128 and clearing caches between phases kept memory bounded. The default
forecast program is unchanged.
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
Mode 1 forwards the diagnosed TKE. The Shin-Hong TKE diagnostic (interface
indices, TTEND Exner factor, mixlen cloud water, CORF) and its caller
(QC3D/QI3D/RQCBLTEN/RQIBLTEN, pblflg-gated scale factors) were corrected; the
op-by-op REAL4 kernel is byte-identical to the original WRF code on 108
column-mode cases (`proofs/t_shinhong/README.md`). CAM-MG
receives the held/diagnosed shared profile across RK, with no QKE substitution
or adapter input floor. Caller modes, cold initialization, binding and restart
retention are tested; the tests reject the old mode-0 caller.

The correction reuses the 20 already allocated CAM fields and the existing PBL
outputs. It adds no PBL solve, persistent array, host transfer or synchronization.
At the Swiss shape each existing REAL interface field is 317,520 bytes; MYNN
now retains one current diffusivity profile in that carry. Legacy PBL precision
wrappers preserve CAM REAL carry without widening it. GPU output-store, layout,
launch and traffic costs have not been measured.
