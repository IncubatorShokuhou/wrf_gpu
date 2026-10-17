# Original WRF initialization and source limits

The oracle compiles untouched numerical bodies with compiler-defined zero
initialization (`-finit-real=zero -finit-integer=0 -finit-logical=false`) and
production-style disabled FP traps. A second build with alternate real
initialization produced the same complete public output stream: 36 calls BYTE.
No material output dependence on that initialization was observed. The port
initializes local work arrays to zero; it does not add physical clamps.

The original driver header warns about unused/incomplete CAM-history
diagnostics and QME3D. This port follows the actual nonchemistry public driver,
including its macrophysics call, and makes no claim to supply chemistry or
omitted CAM history diagnostics. The full original public inventory is retained
in the scientific column gate. Only Registry-selected model fields enter the
operational carry/history stream.

`module_mp_cammgmp_driver.F` explicitly documents that TURBTYPE/SMAW-derived
`smm` does not affect microp_aero_ts outputs. They remain the actual CAM-UW
producer fields in the operational held state. The original CAM-UW
`id_sedfact=.false.` makes WSEDL's entrainment feedback inactive; no substituted
sedimentation feedback is introduced.

Generated numerical statements carry original `.F` locations. CPP line-marker
mapping was checked against identical nonempty preprocessed streams; the
relabelling preserves executable ASTs (`SOURCE_LINE_IDENTITY.json`). The
saturation/error-function helper units were translated directly from `.F`,
so their line mapping is the identity mapping.
