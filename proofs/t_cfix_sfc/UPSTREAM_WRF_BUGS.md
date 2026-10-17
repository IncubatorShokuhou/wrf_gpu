# Upstream WRF 4.7.1 issues met by lane t-cfix-sfc

## CHKLOWQ is never set without a land-surface model (MYJPBL, QNSEPBL)
- `CHKLOWQ` is an `i1` (solver-local, uninitialised automatic) field (Registry.EM_COMMON:1081).
  `module_surface_driver.F` declares it `INTENT(OUT)` (:817) and assigns 1.0 only inside the LSM
  branches (:3429 Noah-MP, :3792, :4033, :4248); with `sf_surface_physics=0` no statement sets it.
- MYJPBL reads it (`module_bl_myjpbl.F:548` `QFC1=XLV*CHKLOWQ*AKHS_DENS`, :576 `CLOW(2)=CHKLOWQ`):
  the land surface-moisture lower boundary of the MYJ diffusion and the land QSFC diagnosis.
- Policy (manager 2026-10-13, uninitialised reads): mirror compiler zero-initialisation. The port
  passes CHKLOWQ = 0 for MYJ without an LSM (no land surface moisture flux into MYJPBL, QSFC kept),
  and 1 after Noah-MP. MYJ+no-LSM is outside every composite; disclosed, not qualified.
