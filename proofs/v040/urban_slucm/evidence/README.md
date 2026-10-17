# SLUCM wiring evidence (lane o2-urban, CPU only)

- `probe_release_*`: proofs/o1_nlbind/probe.py (copy with lane tmp dir), Swiss RD11 release namelist, real CLI loader,
  location-stripped jaxpr of `_advance_chunk_fori` (1 step). Base c8d1d9f98 and wired 09deef1e6 both
  `2bf12cb8374d3a84…` (= published release identity).
- `probe_urbanized_swiss_*`: same probe on RD11 with a 4x4 land block (j,i 18..21) relabelled IVGTYP/LU_INDEX=13
  (RD11 has no urban cell). release arm `2bf12cb8…` (unchanged), `sf_urban_physics=1` traces (`38faf68b…`).
- `smoke_urbanized_swiss_fd0_3steps_*`: swiss_urban_smoke.py, GPUWRF_FAST_DEFAULTS=0 (release Triton kernels
  cannot execute on CPU, E156), 3 production steps (dt 18 s) per arm, 0 non-finite values in either arm; with SLUCM
  the block runs Noah-MP as NATURAL (14), the urban carry evolves and the block TSK/HFX/UST/QSFC/ALBEDO show the urban
  blend; TSK outside the block differs by <= 1.5e-5 K. The +8 K urban skin jump at step 1 is WRF's own urban_var_init
  behaviour (skins seeded from TSK, layers from TSLB; see oracle step 0->1, e.g. stress_cold_stable +2.3 K).
- Rebased on main c10ce7a08 (`9c3bc03ae`): `probe_release_rebased_*` Swiss release `2bf12cb8…` (unchanged);
  `probe_urbanized_swiss_rebased_*` release `2bf12cb8…`, `sf_urban_physics=1` `38faf68b…` (same as pre-rebase).
- CLI `gpuwrf run --dry-run` (urbanized Swiss without wrfout files -> standalone_native_init): sf_urban_physics=1 with
  Noah-MP passes the pre-JAX honour check (rc 0); with sf_surface_physics=2, or in the CPU-history replay driver
  (input dir with wrfout files), it is refused pre-JAX (rc 2) with a named reason.
- `smoke_urbanized_swiss_release_real32_interp_2steps_*`: same smoke under the RELEASE defaults (REAL32 carry) with
  O1_PALLAS_INTERPRET=1 (Pallas interpret + the o2-km2 CPU emulation of the PTX REAL-rounding asm; not GPU binary
  qualification), GPUWRF_CENSUS=1, production entry `_advance_chunk`, 2 steps per arm: 0 non-finite and 0 census
  guards in both arms; SLUCM block effects as in the legacy smoke; TSK outside the block bit-identical (0.0).
