# Coupled F2 disclosure — accepted for CPU lane graduation

Manager decision2026-10-14 accepts the measured km_opt=5 versus km_opt=2
negative-visit difference as a different turbulence trajectory in the same
F2 round-off class (E213/D27). The terminal guard policy passes. No full F2
waiver, masking clamp or numerical-tolerance change is claimed.
F2_DISPOSITION.json records the authority and scope. The historical filename
is retained for links; this disclosure no longer blocks CPU lane graduation.

## Measured paired case

K5-63 SMS uses real Swiss42x42x44 inputs, dt18, three own steps, MP8/CU0,
SF5/Noah-MP4/RRTMG4, PBL0/diff2/km5. Carry and metrics are REAL32. Product
kernel bodies run on CPU with explicit REAL-rounding emulation; native GPU
execution and coupled forecast fidelity remain pending. All189 carry leaves
are finite; all actual guard-water changes are zero. Guards are disabled.

K5-78 changes only km_opt5 to2, retaining the same input files, timestep,
steps, precision, census, guard policy and other resolved namelist options.
All183 leaves are finite, with zero actual guard-water changes. Both arms
have three terminal QC candidates in ring0, with the same minimum index
(k11,j41,i4). The three legacy repaired labels are would-repair candidates.
The original zero-candidate assertion remains recorded as failed; the standing
matched-event policy supplies the accepted terminal classification.

| Recorded QC quantity | SMS km5 | Paired km2 |
|---|---:|---:|
| Post-RK negative visits |15|14|
| Post-RK interior visits |12|11|
| Post-RK minimum (kg/kg) |-1.0379390923132004e-11|-2.830937307490844e-12|
| Terminal ring0 visits |3|3|
| Terminal minimum (kg/kg) |-7.88769548808188e-14|-2.625401151091101e-14|
| Actual added/removed guard water (kg) |0|0|

The post-RK minimum index is(k13,j38,i6) in both arms. Counts and extrema
are preserved in COUPLED_F2.json and PAIRED_KM2_CONTROL.json, together with
raw artifact paths and hashes. The accepted difference is round-off scale
in this observed case; this is not a general epsilon threshold or a waiver
for other amplitudes/populations. Port-vs-port classification is attribution
evidence, not independent WRF fidelity. The independent pristine operator
and nine-RK-stage receipts remain the physics evidence.

Pristine WRF excludes the specified ring from microphysics, matching the
port. Initial QCLOUD and all eight QCLOUD boundary values/tendencies are zero.
No MP mask or QC floor was changed. The earlier MYNN default control had a
different guard count; MATCHED_CONTROL.json retains that history. The closer
single-option TKE control is the relevant terminal-policy witness.

GPU launches/VRAM/E41, native coupled D6 and long-run release F2 qualification
remain pending. Any GPU arm requires a manager window and the official lock,
lease, bridge, timeout and immutable snapshot. The current wave uses CPU only.
