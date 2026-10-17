# SSiB upstream undefined local

Pristine WRF 4.7.1 `phys/module_sf_ssib.F`, SHA256
`2b83b39640fd5d53859f1f281be59a9866baf97924e5d03d9964ec5d69bd5c73`.

`UPDAT1` skips `SNOWW(2)`'s assignment on the multilayer `ISNOW=0` path
(lines 5744–5749 versus assignment 5765), then reads it in the conductivity
calculation at line 5967. It is a local REAL array, not a missing caller argument.
All five poisoned-local snow runs trap there on their second step. The original
non-poisoned runs remain finite; reversing column call order changed none of
9,675 values in the tested three-step matrix.

The manager's 2026-10-13 standing wave policy supplies deterministic compiler
initialization: `-finit-real=zero -finit-integer=0 -finit-logical=false`, with FP
traps off as production WRF. The pristine source stays byte-identical. The port
initializes that local scratch to zero; it does not carry previous-column stack
memory or change WRF's equations.

A separately labeled diagnostic copy sets only `SNOWW(2)=1.0`. It is not the
oracle. Against the zero-initialized original, 9,674/9,675 values are identical;
the remaining ROFF value changes by 7.8444e-12 m. This is not material in the
cold three-step coverage; it is not a general proof for every forecast.
Receipts: `SS04_snow_poison`, `SS09_snow_zero`, `SS11b_init_sensitivity.json`
under `<USER_HOME>/wrf_gpu2_lanes/s2-ssib/`.

# Rounding and caller qualifications

The literal, no-FMA REAL4 transcription is BYTE against the original on 400
column calls, including daylight/night/wet soil, multilayer snow, melting,
snowfall and rain on snow. The compiled path is scored against independently
measured original-WRF FMA/no-FMA spreads. The transfer-law bounds propagate
the pre-registered 0.0005 K canopy-temperature allowance and an eight-REAL-op
pressure-factor ULP envelope through the original RASIT5 routine. These bounds
are not chosen from the port's observed output errors.

One failing compiled column was instrumented without changing any output
bytes: TRIB differs by 11 ULP (0.000336 K), BPS by one ULP. Original RASIT5 on
those compiled operands reproduces the compiled RIB within one ULP. The
experimental GPU/coupled-forecast qualification remains pending; D6 is unchanged.

The oracle caller preserves all 48 snow leaves and WRF's actual-argument
write-back `T2=TSK` after SSIB. The older v0.17 five-regime caller had no positive
snow inputs and discarded the snow outputs, so it supplies only the historical
snow-free control, not snow-trajectory qualification.

## Disclosed compiled REAL4 residuals

With frozen tolerances and original-WRF FMA/input bounds unchanged, ordinary
and cold-snow runs pass all 8,775 field checks. The 300 melt, snowfall and
rain-on-snow calls have 56 field comparisons beyond those bounds (30,000
comparisons total): 2 melt, 38 snowfall, 16 rain-on-snow. All values are finite.
Maximum residual absolute errors: flux 0.05615234 W/m2, temperature
0.00067139 K, Q2 3.2084063e-7 kg/kg, UST 0.00013295 m/s. The literal path
remains BYTE for all 400 calls; compiled checks remain recorded as failures,
not relabeled as within the independently derived bounds. This is experimental
admission under the manager's standing small-excess policy, not full coupled
fidelity qualification. D6 against CPU-WRF remains required.

Complete retained-value scoring: `compiled_residuals.json` (product source
identity and field/location/error records); original artifacts SS26/SS36 in
the lane directory. No output-error-derived tolerance is used.

## Other original-caller edge behavior

`surface_driver.F:4403-4404` forms U10/V10 by dividing by the unmodified
horizontal wind magnitude. At exactly U=V=0 this is undefined (0/0). The port
preserves the expression and reports nonfinite results through its normal
finiteness gate; it does not add an unregistered calm-wind floor. The column
corpus has positive wind magnitudes; exact-calm caller behavior is unqualified.

SSiB also passes its ROFF directly to SFCRUNOFF; INTERCS/INTERC reset ROFF and
compute it in metres while Registry labels SFROFF in mm. The port retains the
original value and Registry metadata, including this upstream discrepancy,
rather than silently scaling an original-WRF diagnostic. Accumulated evaporation
SFCEVP is Registry r-only and remains restart-only; it is not a default history
field.

The final units audit tightens QFX (kg/m2/s) from the broad W/m2 class to
0.002/HLAT, with original WRF HLAT=2.52e6 J/kg (SSIB:1353-1354). This finds
five additional residuals at cells already failing the latent-heat comparison,
for 56 total field excesses. Maximum QFX error is 1.6716513e-8 kg/m2/s. The
original 51 scores are retained in history; no tolerance was widened.
