# MYNN budgets and EDMF control

Experimental, off by default. Two different CPU checks pass: on selected real
columns the compiled MYNN step stays byte-identical with the switches on, and
every history term matches the original WRF driver; for whole domains, the
model was only lowered (types and data flow checked, not executed).
Qualification of the composed GPU forecast is still pending.
These options require native fp32 MYNN (`bl_pbl_physics=5`) and the standard
physics source coupling. Existing forecasts keep their default program.

Set the WRF keys per domain in `&physics`:

```fortran
 bl_pbl_physics = 5, 5, 5,
 tke_budget = 1, 1, 1,
 bl_mynn_output = 1, 1, 1,
 bl_mynn_edmf = 1, 1, 1,
```

The actual WRF key is `tke_budget`; `bl_mynn_tkebudget` is not a WRF key.
The ordinary history alarm controls output frequency, including hourly output
on every nest. Values are held from the last MYNN call, rather than recomputed
at output time or integrated over the history interval.

| Switch | Additional history fields |
| --- | --- |
| `tke_budget=1` | `qSHEAR`, `qBUOY`, `qDISS`, `qWT`, `Dtke` |
| `bl_mynn_output=1` | `edmf_a`, `edmf_w`, `edmf_qt`, `edmf_thl`, `edmf_ent`, `edmf_qc`, `sub_thl3D`, `sub_sqv3D`, `det_thl3D`, `det_sqv3D`, plus raw PBL rates `RTHBLTEN`, `RQVBLTEN`, `RQCBLTEN`, `RQIBLTEN`, `RUBLTEN`, `RVBLTEN` |

Descriptions, units and staggering come from the pristine WRF Registry.
The four production/transport profiles use `bottom_top_stag`: the original
mass prefix is copied and its unused top is zero. `Dtke` and the other fields
use `bottom_top`. The TKE rate metadata preserves the Registry's `m2 s-2`,
although the underlying budget equations describe rates in `m2 s-3`.

`Dtke` is the local MYNN TKE change divided by the MYNN timestep. It excludes
dynamical transport and other forecast sources. Its closure is compared with
the pristine WRF residual; a zero residual is not imposed. The standard WRF
build has `env_subs=.false.`, so its four sub/detrainment histories are zero.

Set `bl_mynn_edmf=0` to disable all plume mass-flux sources. All ten EDMF
histories then remain zero. As in WRF (Registry package `mynn_dmp_edmf`),
history files of such a domain omit `MAXMF`, `MAXWIDTH` and `ZTOP_PLUME`. The momentum sub-switch may be zero only with
this master switch off; a momentum-only disable with EDMF on is refused.
The unqualified TKE mass-flux and downdraft sub-options remain refused.
Selecting EDMF off is a physics change: compare it with an EDMF-off CPU-WRF run,
not with the default forecast. Turbulent mixing itself stays active.

The production MYNN solver supplies every physical field, including carried
EL and QSQ. When either history switch is on, a separate diagnostic twin
receives the same prepared operands and supplies history only. This costs
another MYNN pass and resident history storage; both switches off incur no
twin computation or history allocation. Logical storage is four bytes times
the number of mass cells times the selected five and/or sixteen fields.
Actual GPU kernels, peak memory and forecast cost remain to be measured.
On the CPU component (32 columns, 44 levels, two MYNN calls) the median call
time grows from 3.61 to 4.73 ms by day with EDMF, or from 2.12 to 2.18 ms
without EDMF, and the retained output from 237,968 to 474,848 bytes. These are
CPU component figures, not GPU forecast estimates.

Budgets persist through native checkpoints, pickle restarts and NetCDF
`wrfrst` extensions under schema 1. An ON restart requires the same history
selection, valid held arrays and marker. Enabling history from an older OFF
checkpoint is refused before device restoration.

## Compare with original CPU-WRF

Use the same `tke_budget`, `bl_mynn_output` and `bl_mynn_edmf` values. WRF then
writes the budget and EDMF profiles to history by default. The raw PBL rates
are restart-only in the WRF Registry; request them in `iofields_filename`:

```text
+:h:0:RTHBLTEN,RQVBLTEN,RQCBLTEN,RQIBLTEN,RUBLTEN,RVBLTEN
```

Compare at matching call and output times, and include daytime plumes as well
as cloudy or stable nights. The proof table, per-column Dtke and PBL-rate
conditioning floors, ENT initializer deletion test and measured CPU costs are
in the [evidence report](../proofs/w44_budget/BUDGET.md). These floors do not
relax any physical-field byte-identity requirement.
