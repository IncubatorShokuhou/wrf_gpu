<!-- Draft for w44-docs: append as a section of docs/UPSTREAM_WRF_BUGS.md (v0.4.4). Evidence: proofs/w44_wrfdebug/WRFDEBUG.md -->

## CPU-WRF crashes at start-up with `cycling = .true.` (Noah-MP, no `prec_acc_dt`)

**Who is affected:** anyone running original CPU-WRF with `cycling = .true.` in
`&time_control`, `sf_surface_physics` other than 3 (RUC) — for example Noah-MP — and
`prec_acc_dt` unset or 0. This is a bug in WRF itself, not in wrf_gpu and not in
your input files. WRF 4.7.1 and the newest release we checked, v4.8.0 (`master` and
`develop`, October 2026), carry the same code; it has been there since at least v4.3.

**What you see:** `wrf.exe` aborts within seconds of start-up. glibc reports a heap
error (`corrupted double-linked list`, `corrupted size vs. prev_size`) or there is a
segmentation fault. Where it happens depends on the build and the number of MPI ranks:
the first history write, the Noah-MP table read, and so on. The first `wrfout` file can
be left truncated (48 bytes, unreadable). The same inputs run normally with
`cycling = .false.`.

**Cause:** `dyn_em/start_em.F` (lines 299–310 in 4.7.1) clears the diagnostic
buckets when `cycling` is on:

```fortran
grid%prec_acc_nc(i,j) = 0.   ! allocated only when prec_acc_dt > 0
grid%snow_acc_nc(i,j) = 0.   ! allocated only when prec_acc_dt > 0
grid%acrunoff   (i,j) = 0.   ! allocated only for RUC (sf_surface_physics = 3)
```

The Registry allocates these three arrays only for their packages (`prec_acc`,
`ruclsmscheme`). Otherwise each one is a 1×1 placeholder, and the loop writes a whole
tile into it. A build with bounds checking stops at the first bad write:
`Index '81' of dimension 1 of array 'grid%prec_acc_nc' above upper bound of 1`.

**Workarounds:**
- A namelist change is **not enough**. `prec_acc_dt > 0` allocates `prec_acc_nc` and
  `snow_acc_nc`, but the `acrunoff` write still overruns under Noah-MP. That run
  crashed with a segmentation fault.
- **Fix:** guard the two writes with the package conditions and rebuild:
  `IF (config_flags%prec_acc_opt == 1)` around `prec_acc_nc`/`snow_acc_nc`, and
  `IF (config_flags%sf_surface_physics == RUCLSMSCHEME)` before `acrunoff`.
  The patch is `proofs/w44_wrfdebug/start_em_cycling_bucket_guard.patch`. It changes
  only code that runs when `cycling` is on. On a normal start the patched build wrote
  `wrfout` files byte-identical to the unpatched build.
- Or run without `cycling`. This changes what WRF does at start-up (no cycling
  rebalance, no bucket reset, different MYNN/Thompson initialisation), so it is not
  the same experiment.

The bug is in WRF's Fortran memory handling, so the GPU model does not share it. We use the patched CPU-WRF build only to produce CPU reference
("twin") runs from cycled starts. It is a separate build, not the default CPU-WRF.
