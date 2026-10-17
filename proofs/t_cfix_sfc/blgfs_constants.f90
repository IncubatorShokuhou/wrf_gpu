! Bit patterns of the constants WRF's BL_GFS call actually uses (lane t-cfix-sfc F13, rv-t-cfix SFC2).
! REAL wrapper constants: module_model_constants (module_pbl_driver.F:1413,1418 pass them as REAL).
! kind_phys values: MODULE_GFS_PHYSCONS + MONINP PARAMETERs/inline literals (default-REAL literals
! assigned to REAL(kind_phys)), declared exactly as module_bl_gfs.F:620-672.
! build: gfortran -O2 -I<oracle/pbl3/zero> blgfs_constants.f90  (output: fixtures/blgfs_constants.txt)
program blgfs_constants
  use module_model_constants, only: r_d, cp, g, rcp, rovg, ep_1, karman
  use module_gfs_machine, only: kind_phys
  use module_gfs_physcons, grav => con_g, RD => con_RD, CPG => con_CP, FV => con_FVIRT, RV => con_rv
  implicit none
  real(kind=kind_phys) gg, gor, gocp, vk, dw2min, cfac, sfcfrac, qmin, zfmin, one
  parameter(gg=grav)
  parameter(GOR=GG/RD, GOCP=GG/CPG)
  parameter(VK=0.4, DW2MIN=0.0001, CFAC=7.8, SFCFRAC=0.1, QMIN=1.E-8, ZFMIN=1.E-8)
  one = 1.0_kind_phys
  print '(a,7(1x,z8.8))', 'REAL r_d cp g rcp rovg ep_1 karman', r_d, cp, g, rcp, rovg, ep_1, karman
  print '(a,5(1x,z16.16))', 'PHYS con_g con_rd con_rv con_cp con_fvirt', grav, RD, RV, CPG, FV
  print '(a,8(1x,z16.16))', 'MONINP gor gocp vk dw2min cfac sfcfrac qmin zfmin', gor, gocp, vk, dw2min, cfac, &
       sfcfrac, qmin, zfmin
  print '(a,4(1x,z16.16))', 'INLINE .1 1.746 1.286 2.1', one*.1, one*1.746, one*1.286, one*2.1
end program
