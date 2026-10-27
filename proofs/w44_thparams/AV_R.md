# AV_R — rain-law coefficient and its dependent initialization

CPU-qualified isolated candidate; the next composed program and GPU/E41/cost
remain open. The frozen Nt four-parameter GPU packet is unchanged.
The public runtime binder refuses this experimental control until its own
GPU/E41/cost gates pass; the low-level CPU qualification path is retained.

Original WRF declares `av_r=4854.0` as a private REAL PARAMETER at
`module_mp_thompson.F:143`, units s⁻¹ (`bv_r=1`). Active uses include both
sedimentation moments (:3618/:3626), cloud/ice collection initialization
(:786–788), and rain-evaporation initialization (:801). The rain velocity
expressions in the collection-table builders (:4187/:4403) are commented out;
their active rain velocities use the independent Gunn–Kinzer polynomial.

Changing this is a gpuwrf extension requiring a recompiled WRF reference. The
implementation passes the scalar AV_R and three derived REAL values into the
existing Pallas kernel. The latter cover all four dependent initialization
statements (`t1_qr_qc == t1_qr_qi`, `t2_qr_qi`, `t2_qr_ev`). Scalar preparation
runs before the column kernel. No table or prognostic storage is added.

- AR01 [M CPU]: original `wrf_constants.F90`, FCOPTIM, recompiled at
  2427/4854/9708. All nine eager coefficient values are BYTE. Compiled t1
  differs by one REAL ulp through association; the other two coefficients
  are BYTE. One JIT handles all three values.
- AR02 [M CPU compiled]: real 24-column, 44-level, dt=18 s corpus; changed
  9708 alters 15 output leaves with only OFF/ON traces. All values finite.
- AR03 [M original WRF vs CPU]: recompiled `av_r=9708` passes all ten fields
  under the unchanged mixed-phase increment tolerances; ignoring the new
  value fails. The original table hashes and caller are retained.
- AR04 [M CPU causal intervention]: freezing ONLY the ON coefficient and
  its dependent values on the same native application restores **20/20
  output leaves BYTE to OFF**. No equation, table, mask, state or fixture was
  changed. This isolates the observed default differences to constantness.
- AR05 [M CPU CUDA-target lowering]: OFF raw HLO is BYTE to original
  `12a9494b…` (628924 bytes). ON4854/9708 raw HLO and canonical graph are BYTE,
  SHA `686fca29…`, 630776 bytes, one native call. This is not GPU compilation.
- AR06 [M CPU]: 94 admission, original-initializer-bit, catalog, restart and
  parameter-key tests pass in 2.26 s. Different values retain the static/aval
  signature; different selected names do not collide.

| Default output | Changed cells | Maximum absolute | Maximum REAL ulps |
|---|---:|---:|---:|
| qc | 1 | 1.4551915228e-11 kg/kg | 1 |
| qr | 10 | 1.1641532183e-10 kg/kg | 3 |
| Nr | 20 | 0.00146484375 #/kg | 6 |
| Ng | 2 | 9.5367431641e-7 #/kg | 1 |

Maximum relative difference is 4.5417409841e-7. The existing generic checker
failure is preserved; no frozen WRF tolerance has changed. The manager's
measured few-ulp rule admits this localized isolated class; the combined
default class remains to be measured. No GPU rounding or performance claim.

Product implementation: `a49ce24e0c` scalar initializer plus `8b2c9b9aa3`
admission/operand wiring. Artifacts: `<USER_HOME>/wrf_gpu2_lanes/w44-thparams/AR01`
through `AR04`; the original Fortran binaries were removed after scoring.
