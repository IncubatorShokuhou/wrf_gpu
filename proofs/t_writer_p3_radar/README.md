# P3 REFL_10CM: existing producer retention

Class A. P3 already computes `diag_ze = 10*log10((ze_rain+ze_ice)*1e18)`
and returns it as `ze`, in dBZ. WRF passes `REFL_10CM` to the wrapper's
`diag_zdbz_3d` argument, which receives this P3_MAIN output directly.
The native port adapter previously omitted `ze` from its selected outputs.
This change retains that output at the actual post-RK microphysics exit,
using the existing radar carry, interior mask, restart and writer path.
There is no second logarithm, reconstructed radar, new scattering equation
or new State leaf. Initial c23 radar is zero in both original CPU series;
subsequent values are the producer's dBZ. P3 has no `REFD_MAX` variable.

CPU evidence: 157 focused checks plus 14 ice-restart consumers pass. The
same-input original WRF P3 fixture passes three steps under the existing
P348/P360 compiled policy. Its strict 0.0003 dBZ residuals remain disclosed:
36/288 exceed that strict band, maximum 0.0485477 dBZ, covered by the already
registered independent WRF contraction spread. No frozen bound was changed.
All eight P3 mathematical source files remain byte-identical to the earlier
qualified model; the original P360 scores are retained in this packet.

All 25 raw and 25 ZERO-init c23 CPU frames pass byte-exact held-field and
NetCDF publication checks. These are transcription checks, not a claim that
the GPU trajectory matches those frames. A real adapter-retention mutant
causes one value assertion and zero setup errors. Both fresh full default
program hashes remain exact: Swiss `f672a573`, PROD `6fc715a2`.

The new native MP52 output ABI has 27 fields instead of 26. The existing
reflectivity arithmetic may become live after compiler dead-code removal.
One resident c23 REAL radar volume adds 310464 bytes; temporary output/mask
storage and register, stack, compile, launch and runtime costs require actual
native qualification. The P3 native compiler repair is a separate lane.
No GPU/E41/full-c23 fidelity or forecast-performance PASS is claimed here;
the old c23 GPU result stays INCOMPLETE until new output is measured.

P3 checkpoints require the full runtime reader to preserve radar. The daily
P3 caller uses the existing full-carry stepper for the same lifetime. The
existing particle/ice restart assertions were retained and now also verify
radar bytes. `QUALIFICATION.json` seals sources, references and raw receipts.
