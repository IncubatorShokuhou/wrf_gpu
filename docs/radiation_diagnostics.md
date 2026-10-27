# Radiation heating and cloud-input diagnostics

Available in the merged v0.4.4 development build, not v0.4.3. The heating
exports have pristine-WRF CPU-oracle evidence and a three-nest GPU smoke;
the full release gates and paired kernel/copy accounting remain open.
`rad_diag` is a gpuwrf extension, not an original WRF namelist key:

```fortran
&gpuwrf
 rad_diag = 1,
/
```

The option accepts this only with RRTMG longwave and shortwave
(`ra_lw_physics=4`, `ra_sw_physics=4`). A scalar selects all domains; a list
selects them individually. Each domain holds values from its own radiation
call until its next call; the history writer does not solve radiation again.

## Swiss example

The shipped Swiss case already selects RRTMG4/4. Make a private copy, append
the new group, then run it on the GPU:

```bash
mkdir -p runs
cp -a examples/switzerland_d01 runs/swiss-raddiag-case
cat >> runs/swiss-raddiag-case/namelist.input <<'EOF'
&gpuwrf
 rad_diag = 1,
/
EOF
gpuwrf run --input-dir runs/swiss-raddiag-case --domain d01 --hours 3 \
  --output-dir runs/swiss-raddiag
```

Use a fresh copy/output directory for another attempt. This is a setup example;
the measured GPU case below is Tenerife, not this Swiss command. Use a longer
day-and-night forecast to investigate cloud heating.

## Fields and WRF meaning

| History field | Meaning | Units |
| --- | --- | --- |
| `RTHRATLW`, `RTHRATSW` | Uncoupled all-sky theta heating, longwave/shortwave | K s-1 |
| `RTHRATLWC`, `RTHRATSWC` | Uncoupled clear-sky theta heating, longwave/shortwave | K s-1 |
| `QC_RAD`, `QI_RAD`, `QS_RAD`, `QG_RAD` | Condensate mixing ratios actually passed to RRTMG after cloud mediation | kg kg-1 |

The four heating names follow the WRF Registry. The four `*_RAD` mixing-ratio
fields are gpuwrf extensions with no equivalent CPU-WRF history variables;
comparing them to unmediated QCLOUD/QICE/QSNOW/QGRAUP would compare different
quantities. Existing CLDFRA holds the fraction sent to the solvers.

`QI_BL` is added beside existing `QC_BL` in default history, independently of
`rad_diag`. An explicitly selected stream requesting QC_BL also includes QI_BL.
It exposes existing MYNN PBL cloud ice; it is not an extra prognostic quantity.
Lead-zero history precedes the first solve, so initialized diagnostic zeros
are not evidence of a zero daytime heating rate.

## Limits and cost

Enabled diagnostics require restart schema `rad_diag_schema_version=1`
(NetCDF attribute `GPUWRF_RAD_DIAG_SCHEMA_VERSION=1`). Older/missing-schema
checkpoints, missing diagnostic fields, invalid REAL dtype/shape, nonfinite
values and changed per-domain selection refuse.
Non-RRTMG selections refuse. Clear heating reuses the clear interface fluxes
already computed, with no second transfer solve. Retaining eight REAL32
volumes costs 32 bytes per mass cell [I], before allocator/workspace effects.
Disabled diagnostic storage is absent. QI_BL reads an existing leaf and adds
history output, rather than a prognostic array. Its isolated output cost has
not been measured.

Measured on the Tenerife case with three nests, over three forecast hours
starting at 00 UTC on 28 February 2026,
with allocator fraction 0.30 [M]:

| Selection | Second segment, s/forecast-hour | JAX live peak, MiB | Per-process GPU reservation peak, MiB |
| --- | ---: | ---: | ---: |
| Diagnostics off | 5.725 | 2746.09 | 10592 |
| `rad_diag=1` | 6.642 | 2979.86 | 10580 |
| `rad_diag=1`, `icloud_bl=0` | 5.571 | 3053.73 | 10580 |

This single sequential comparison has no A/A noise estimate. The observed
ON second-segment increase is 0.917 s/forecast-hour (16.0%); live memory rises
by 245,131,360 bytes (8.5%). Fixed pool reservation hides the live-memory
increase. These are short nighttime observations, not whole-run 24-hour
timings, a daytime bound or a speedup claim for `icloud_bl=0`. The reported
last-call proxy omits the tail and is not used here. Extra kernels/copies
remain unknown because the paired profiler capture did not complete.

## Qualification scope

The pristine-WRF solver/merge checks include 186 real day/night columns with
liquid and ice. GPU OFF, ON and ON+`icloud_bl=0` each complete the three-nest
3-hour case with 12 history frames and finite fields. ON versus OFF preserves
378 shared emitted variables across 12 frames byte for byte (4,536 comparisons).
That is regression evidence for emitted history, not every hidden carry leaf
or an independent fidelity reference. The actual GPU executable resource gate
passes for all 12 scanned entries. A separate four-nest 333 m run and the
composed release gates are not covered.

For the separate WRF switch that excludes MYNN subgrid clouds from radiation,
see [icloud_bl](radiation_pbl_clouds.md).

## CPU-WRF comparison history

Use original WRF with identical radiation, cloud options and output times.
Its `iofields_filename` can request existing Registry fields:

```text
+:h:0:RTHRATLW,RTHRATSW,RTHRATLWC,RTHRATSWC,QC_BL,QI_BL,CLDFRA,QCLOUD,QICE,QSNOW,QGRAUP
```

For exact `*_RAD` comparison, an instrumented original WRF caller is needed
after cloud mediation. Standard condensate history is context only. Source,
field metadata, oracle scope and cost limits are in the merged
[RADDIAG report](../proofs/w44_raddiag/RADDIAG.md),
[CPU receipt](../proofs/w44_raddiag/CPU_RECEIPT.json) and
[GPU receipt](../proofs/w44_raddiag/GPU_RECEIPT.json).

Return to [Tune WRF for your region](regional_optimization.md).
