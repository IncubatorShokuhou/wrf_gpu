# MYNN subgrid clouds in radiation: icloud_bl

WRF's `&physics icloud_bl` controls whether radiation merges MYNN subgrid
cloud fraction, water and ice into its resolved-cloud inputs. It does not
disable MYNN cloud production or transport. In the merged v0.4.4 development
build, set this in the existing `&physics` group:

```fortran
&physics
 icloud_bl = 0,
/
```

`icloud_bl` is a single WRF value shared by all domains, unlike the optional
per-domain `rad_diag` selection. Do not use a list to request different cloud
coupling on individual nests.

Default 1 retains the existing path; 0 skips the radiation-only merge. WRF
cloud modes `icloud=0`, 2 and 3 bypass that merge independently. In particular,
`icloud=0` removes radiation cloud effects; it is a different experiment from
excluding only the MYNN subgrid contribution.

## Status, limits and cost

v0.4.3 admits the default value only. The new 0 branch is CPU-oracle tested
against the literal pristine WRF merge. A three-nest Tenerife run over three
nighttime hours starting at 00 UTC on 28 February 2026
GPU run with `rad_diag=1, icloud_bl=0` completes with finite output and passes
the executable resource gate. This does not establish better regional cloud
or temperature skill, or complete the release gates.

The selection changes the compiled program, so changing it may require
compilation. No new prognostic fields are needed. The measured combined
selection uses 3053.73 MiB JAX live peak and 5.571 s/forecast-hour for its second
segment [M]. See the [radiation cost table](radiation_diagnostics.md#limits-and-cost)
for the other selections and the fixed-pool reservation. There is no repeated
A/A or paired kernel/copy count, so these observations do not establish an
isolated EDMF/cloud-merge speed benefit. MYNN itself remains active.

## CPU-WRF comparison history

Use the same `icloud` and `icloud_bl` settings on both arms. Request:

```text
+:h:0:QC_BL,QI_BL,CLDFRA,QCLOUD,QICE,SWDOWN,GLW,RTHRATLW,RTHRATSW
```

This is CPU-WRF `iofields_filename` content. Compare PBL condensates
separately from radiation inputs and compare day/night heating and surface
fluxes. The optional [radiation diagnostics](radiation_diagnostics.md) expose
the gpuwrf mediated condensates. See the original-WRF oracle and GPU scope in
the merged [RADDIAG report](../proofs/w44_raddiag/RADDIAG.md).

Return to [Tune WRF for your region](regional_optimization.md).
