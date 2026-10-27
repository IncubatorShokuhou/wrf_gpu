# w44-wrfdebug — why original CPU-WRF aborts on cycling starts

Lane w44-wrfdebug (Opus 5.5), 2026-10-23. CPU only (cores 8, 9, 12, nice 19, E219 CPU-only
OpenMPI env, 12 GiB RAM slot). No GPU used. The w44-warmstart input sets were used read-only.

## Root cause [M]

This is a genuine bug in the WRF 4.7.1 source. It is not caused by the input content, the
build, or OpenMPI.

`dyn_em/start_em.F:299-310` (subroutine `start_domain_em`):

```fortran
    if(config_flags%cycling) then
! Clear the buckets for diagnostics at initial time
       DO j = jts,min(jte,jde-1)
       DO i = its, min(ite,ide-1)
            grid%prec_acc_nc(i,j) = 0.      ! line 303
            grid%snow_acc_nc(i,j)  = 0.
            grid%sfcrunoff  (i,j) = 0.
            grid%udrunoff   (i,j) = 0.
            grid%acrunoff   (i,j) = 0.      ! line 308
            grid%acsnow     (i,j) = 0.
```

- `prec_acc_nc` and `snow_acc_nc` belong to package `prec_acc` (`prec_acc_opt==1`,
  Registry.EM_COMMON:3269). That option is set only when `prec_acc_dt > 0`
  (share/module_check_a_mundo.F:1142).
- `acrunoff` belongs only to package `ruclsmscheme` (`sf_surface_physics==3`,
  Registry.EM_COMMON:3157; generated `in_use_for_config_a.inc:863`).
- When a package is off, WRF allocates the array as `(1,1)`
  (generated `inc/allocs_9.F:4301`). The cycling loop then writes a whole tile into it:
  an out-of-bounds heap write.
- The glibc heap check fires later, at whichever `malloc`/`free` comes next. That is why the
  w44-warmstart arms aborted at different places:
  - B at the first history write;
  - C/D in Noah-MP `read_mp_global_parameters` / allocation.
- The operational WN3 namelist (Noah-MP 4, `prec_acc_dt` unset) hits both overruns on every
  domain.

**Version scope [M]:**
- The canairy gen2 `start_em.F` is byte-identical to pristine 4.7.1 (sha256 `4b6c98a0…`).
- The loop is unchanged in upstream WRF `v4.8.0` (released 2026-06-08) and on the current
  `master` and `develop` (fetched 2026-10-23; `start_em.F` sha256 `76c6ab37…`, loop at lines 314–327).
- The loop is present in v4.3, v4.4, v4.5, v4.5.2 and v4.6.0.
- The v4.8.0 Registry still puts the three arrays in packages (lines 3211, 3324).
- A search of the GitHub issues and PRs for "cycling prec_acc_nc", "cycling acrunoff" and
  "start_em cycling buckets" found nothing.
- So the bug is **not fixed upstream**. A later WRF release would not help; this patch is needed.

## Evidence matrix

Tool: `tools/wrfdebug_arm.py`. It uses the same env, cores, 3×1 decomposition and stop rule
(3 initial history files + 10 root steps, 240 s) as the warmstart bisect. Inputs are
read-only symlinks with SHA checks before and after each arm. Runs are under
`<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/runs/<arm>/`.

| arm | inputs | namelist change | exe | result |
|---|---|---|---|---|
| warmstart A | fresh real.exe May04 00 | — | orig | PASS (reference) |
| **E1** | fresh real.exe (same as A) | **+ `cycling=.true.` only** | orig | **FAIL** 8.2 s, `corrupted double-linked list`, 48 B wrfout_d01 |
| E2 | fresh | + cycling, + `prec_acc_dt=60,60,60` | orig | FAIL 7.2 s, SIGSEGV (rc 139): the namelist workaround is NOT enough |
| **E6** | fresh | + cycling | orig + `-fcheck=bounds` (start_em only) | `start_em.F:303 Index '81' of dimension 1 of array 'grid%prec_acc_nc' above upper bound of 1` (all 3 ranks: 81/41/2) |
| **E7** | fresh | + cycling + prec_acc_dt | orig + bounds | `start_em.F:308 … array 'grid%acrunoff' above upper bound of 1` |
| E3 | fresh | + cycling | **patched** | PASS 63 s |
| **E4** | fresh | — (cycling off) | patched | PASS; t0 wrfout d01/d02/d03 **BYTE-identical** to warmstart A (2e6901d1 / a8005c43 / a5b8233e) |
| **E9/E10** | fresh | — (cycling off) | orig vs patched | all 6 wrfout (t0 + **+1 h forecast** d01–d03) **BYTE-identical** |
| E5B | WRF's own wrfout renamed wrfinput (B) | native GPU nml (cycling) | patched | PASS 76 s, 3×375 vars |
| E5C | our native cycled files (C) | native | patched | PASS 86 s |
| E5D | real-schema files (D) | native | patched | PASS 80 s |
| E8 | D | native | patched + bounds | PASS: no other out-of-bounds write in start_em under cycling |

E1 alone settles the question: the very inputs that pass in arm A abort once `cycling=.true.`
is added.

## Fix and private build (approved by manager + ALISIOS for cycling twins only)

Patch (`start_em_cycling_bucket_guard.patch`): guard each write with the predicate that owns
the array, as the Registry package defines it. Nothing else changes.

```fortran
            IF ( config_flags%prec_acc_opt .EQ. 1 ) THEN
            grid%prec_acc_nc(i,j) = 0.
            grid%snow_acc_nc(i,j)  = 0.
            ENDIF
            ...
            IF ( config_flags%sf_surface_physics .EQ. RUCLSMSCHEME ) grid%acrunoff   (i,j) = 0.
```

This keeps WRF's intended semantics: every bucket that exists is still zeroed.

Build method (`tools/build_private_wrf.sh`):
- Recompile the single object `dyn_em/start_em.F` with the exact CMake `flags.make` flags.
- Replace it in a private copy of `libWRF_Core.a`.
- Relink with the original `main/CMakeFiles/wrf.dir/link.txt` command.
- The script never writes into the original build.

| item | path | sha256 |
|---|---|---|
| original wrf.exe (canairy gen2 dmpar) | `<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/run/wrf.exe` | `1d0f5779fff143361e470e37f89c734497ed5a7bb55d0c53b7e3fccc569dd49b` |
| **control relink, unmodified source** | `<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/orig/wrf.exe` | `1d0f5779…` **BYTE-identical** (object `a30adeb7…` == archive member) |
| **patched wrf.exe (cycling-twin build)** | `<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/patched/wrf.exe` | `1c1269fb10b7aa135bf42417f1aa7f7a36f7137ea40cd82769316e8ac5bb4918` |
| patched start_em.F | `…/build/patched/src/start_em.F` | `2531f466c8c76e0ea1eababce7d64105a464ead6157007992e4d3aa940eea032` |
| patched start_em.F.o | `…/build/patched/start_em.F.o` | `b69693331bfe65adcad68eb96c4aebc34e67c95acf46392256140f1641e70a11` |
| diagnostic bounds builds (not for use; exes deleted after E6–E8) | `…/build/orig_bounds`, `…/build/patched_bounds` | `eabdf57e…`, `d21f19c6…` |

The patched build is the same toolchain, flags and every other object as `1d0f5779`. Use it
ONLY for cycling twins: a separate build family in the ALISIOS DATA_MAP. The patched exe is
staged at `<DATA_ROOT>/wrf_gpu2/scratch/w44-wrfdebug/build/patched/wrf.exe`. The manager may
move it to a permanent place; keep the sha.

## Normal-start identity (patch inert outside cycling)

- E4: the t0 history of all three domains is BYTE-identical to the original build.
- E9/E10 (fresh inputs, cycling off, original vs patched exe, to the +1 h history of d01–d03):
  **all 6 files BYTE-identical [M]**, each 375 variables. Wall times: orig 496 s, patched 402 s (stop at d01 step ≥ 70).

| file | sha256 (orig == patched) |
|---|---|
| wrfout_d01_2026-05-04_00:00:00 | 2e6901d180b2888b1068fcb7eddfef3b78f02420f09aebf6054090eaa7a7c6d9 |
| wrfout_d01_2026-05-04_01:00:18 | 4cc926ab7ca640f8fa640b2a720f9f92639b0a5710a7ceff851dde02cb1dbc5c |
| wrfout_d02_2026-05-04_00:00:00 | a8005c4325697a400fc370af395988f03138b6f42db18020690a6229c40c4cbf |
| wrfout_d02_2026-05-04_01:00:00 | 0f98c0a6273ad112ae85b0d2941fe9308fb429500ddd23ce56eb722dd8c631cb |
| wrfout_d03_2026-05-04_00:00:00 | a5b8233ebb7b84ecc4c33eac5e5dda7cad435ba19608390e6e6ec6fc41fa531a |
| wrfout_d03_2026-05-04_01:00:00 | 6e9f6e0daf0b52be3324203e97b8bc791ce2b9a8f697eec095d1cd489c867131 |

  The t0 files are also identical to the w44-warmstart original arm A.

## Build facts (manager addendum)

- WRF `V4.7.1` (`inc/version_decl`). CMake "gen2" build: `EM_REAL`, `ARW`, nesting BASIC,
  `USE_MPI=ON`, `USE_OPENMP=OFF`, `USE_DOUBLE=OFF` (RWORDSIZE=4), `MAX_DOMAINS_F=21`,
  `BUILD_RRTMG_FAST=OFF`, `BUILD_SBM_FAST=ON`, chem/hydro/KPP off.
- Toolchain: conda-forge gfortran 14.3.0. Flags `-O2 -march=nocona -mtune=haswell -ftree-vectorize
  -funroll-loops -fPIC -fstack-protector-strong`, big-endian. No bounds checking, so the
  overrun is silent until glibc notices.
- Libraries: netCDF-C 4.10.0, netCDF-Fortran 4.6.2, Open MPI 5.0.10.
- The build is not "old" in WRF terms; it is the same release as pristine. The bug does not
  depend on the build (pristine serial has the identical loop) [I for serial; not run].
- **Side finding:** the gen2 source tree is NOT pristine 4.7.1. It differs in 9 files of
  dyn_em/phys.
  - env-gated diagnostic instrumentation (`wrf_gpu_diag_state`, `WRITE(0,*) SFCLAY1D_*`) in
    solve_em, first_rk_step_part1/2, module_sf_mynn and surface_driver;
  - `module_sf_mynn.F`: land QSFC validity test `QSFC ≤ TINY(1.0)` instead of `≤ 0.0`.
    This edit is IN the binary: recompiling the current source gives a byte-identical object
    (d639d71d). It differs from upstream only for subnormal QSFC;
  - `module_ra_rrtmg_sw.F`: `kmodts=1` (Eddington) instead of upstream `kmodts=2`. The edit
    was made 2026-05-21, after the 2026-04-28 build. The binary's object disassembles
    identically to upstream v4.7.1 (`kmodts=2`), so `1d0f5779` is NOT affected. Any REBUILD
    from that tree would change shortwave radiation → tell ALISIOS before anyone rebuilds.

## Reproduce

```
JAX_PLATFORMS=cpu scripts/with_ram_slot.sh --max 12G -- taskset -c 8,9,12 \
  env JAX_PLATFORMS=cpu python3 proofs/w44_wrfdebug/tools/wrfdebug_arm.py <NAME> \
  --inputs {A,B,C,D} --nml {fresh,native} --exe {orig,patched,orig_bounds,patched_bounds} [--cycling true] [--prec-acc-dt 60] [--hour1]
```
