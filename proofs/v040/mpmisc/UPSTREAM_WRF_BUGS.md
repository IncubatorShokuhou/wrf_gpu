# Ferrier initialization sensitivity

The unmodified `module_mp_etanew.F` and `module_mp_fer_hires.F` bodies were
built with compiler-defined zero initialization and with alternate NaN REAL
initialization (integer zero/logical false, no floating-point traps). In the
12 fresh stress columns, all recorded input and output fields were identical
between the two builds, with no nonfinite output (MP29). These columns cover
3/2/4 consecutive calls, GID 1/2, 18/54/1 s timesteps, high cloud/rain and
rime factor 20. No material initialization dependency was observed in this
sample. This does not prove all upstream paths initialize every local.

Original source hashes, compiler flags and driver hashes are retained in
`<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/MP29/receipt.json` and `MP29_nan/receipt.json`;
the field comparison is `MP29/init_sensitivity.json`. Build directories were
removed immediately. No pristine routine body was edited.
