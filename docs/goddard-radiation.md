# Goddard radiation

**Status in v0.4.0: Validated 6 h.** Passed a six-hour comparison with the original CPU WRF; the accepted check covers six forecast hours, so treat longer forecasts as experimental.

Select `ra_lw_physics=5`, `ra_sw_physics=5`, `mp_physics=8` and `use_mp_re=1`
for the paired WRF 4.7.1 Goddard scheme. The REAL4 wrapper uses the original
cloud optical tables and ozone climatology; the SW/LW transfer routines retain
WRF's explicit DOUBLE arithmetic. Heating and surface/TOA fluxes refresh at
`radt` and are held between calls, including land forcing and history output.
The held MP8 radii, fluxes and pre-correction SW heating counter survive restart.

The port is also checked on the CPU against the original WRF code. The original
cloudy-SW code reads an uninitialized optical depth. The port mirrors a reference
in which the compiler sets it to zero; results are sensitive to this choice. See
[upstream WRF bugs](UPSTREAM_WRF_BUGS.md).

Mixed radiation pairs, other microphysics radius mediation, exact-equator
latitude, GOCART/aerosol feedback, slope/shadow radiation and `swint_opt=1`
are refused for this configuration. This prevents silently substituting missing
caller inputs. Original rain/graupel/hail radii are zero for the MP8 caller;
their optical-depth behavior follows WRF's table interpolation literally.

Surface radiation consumes the held Noah-MP ALBEDO/EMISS, including SLUCM's
merged surface, or the existing RUC surface view.
RUC initialization and cadence refresh therefore use the same selected surface
operands. Slab1, Noah-classic2 and Pleim-Xiu7 are refused for Goddard because
their current adapters do not expose a validated held ALBEDO/EMISS view here.
BEP/BEM2/3 requires RRTMG4/4 and is also refused before execution. Goddard+lake
is refused because the shared surface-property land mask drops the lake albedo
overlay on water-marked lake cells. The no-LSM0
control retains the existing common surface-table path. See the
[surface caller audit](../proofs/s2_goddardrad/CRITIC_FIXES.md).

Qualification assets and reproduction commands are in
[`proofs/s2_goddardrad`](../proofs/s2_goddardrad/). The scheme keeps a cumulative
on-device `radiation_diagnostics.sw_heating_negative_count` for WRF's original
negative-SW-heating correction; it is available from the operational carry and
is checked by the short CPU coupled run.

The public wrapper's four pristine cases have 148/148 REAL4 outputs byte-identical
with CPU FMA and algebraic simplification disabled. On the normal CPU compiled
path, surface/TOA fluxes, native cloud/solar history and LW rates remain byte
identical; SW rates differ by at most 1.72804e-10 K/s (95 ULP near cancellation).
An independent pristine-WRF native-ISA/FMA build differs from its generic build
by at most 9.09495e-12 K/s. The compiled SW excess is a documented deviation.
The fixed reference tolerances were not changed.

The paired solver uses bounded 128-column tiles and one cadence refresh, and
history consumes held fields. A CPU component probe on 128 repeated original
12-level columns measured 3,320,328 bytes of compiled temporary storage, 132,616
argument bytes and 19,584 output bytes; compile took 4.41 s. Nine warm samples
gave medians of 34.02 ms mixed day/night and 23.89 ms all-night, with substantial
shared-host noise. These are component observations, not a speedup claim.
The five additional held history planes cost 20 bytes per horizontal cell and
the cumulative counter costs four bytes. GPU launches, peak GPU memory and full
forecast timing have not been measured for this option; the default forecast
program is unchanged.
