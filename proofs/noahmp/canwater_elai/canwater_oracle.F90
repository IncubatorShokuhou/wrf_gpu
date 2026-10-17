! Pristine-WRF CANWATER oracle with the NOAHMP_SFLX caller wiring (V033-W2, GPUWRF_NOAH_CANWATER_ELAI).
!
! cwe_routines.inc = VERBATIM slices of <USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F
! (PHENOLOGY :1255-1358, CANWATER :6265-6394), written by make_oracle.py, which checks the slice SHA256 values.
! Both routines are PRIVATE in MODULE_SF_NOAHMPLSM, so they are compiled here inside a module that takes the
! parameter type, DVEG and the physical constants from the compiled pristine module. Parameters come from
! the real MPTABLE (MODIFIED_IGBP_MODIS_NOAH) through the pristine TRANSFER_MP_PARAMETERS.
! Caller wiring reproduced from NOAHMP_SFLX: PHENOLOGY -> ELAI/ESAI; FVEG for dveg=4 (:864-865, :874-875);
! BDFALL from ATM (:1216, OPT_SNF=1); FROZEN_CANOPY from the TV entering ENERGY (:2210-2216);
! then WATER -> CANWATER (:6116). Stream I/O, big-endian (WRF -fconvert=big-endian), REAL = 4 bytes.
module cwe_oracle
  use module_sf_noahmplsm, only : noahmp_parameters, DVEG, TFRZ, HVAP, HSUB, HFUS, CWAT, CICE, DENH2O, DENICE
  implicit none
contains
  include 'cwe_routines.inc'
end module cwe_oracle

program canwater_oracle
  use module_sf_noahmplsm, only : noahmp_parameters, noahmp_options, TFRZ
  use noahmp_tables, only : read_mp_veg_parameters, read_mp_soil_parameters, &
       read_mp_rad_parameters, read_mp_global_parameters
  use module_sf_noahmpdrv, only : transfer_mp_parameters
  use cwe_oracle, only : phenology, canwater
  implicit none
  type(noahmp_parameters) :: parameters
  integer :: ncol, ic, vegtyp, yearlen, pgs, frz, soiltype(4)
  real :: dt, lat, julian, snowh, tv0, troot, shdmax, sfctmp, fcev, fctr, tg
  real :: lai, sai, elai, esai, igs, fb, fveg, bdfall
  real :: canliq, canice, tv, cmc, ecan, etran, fwet, qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc
  logical :: frozen_canopy

  call read_mp_veg_parameters('MODIFIED_IGBP_MODIS_NOAH')
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, &
                      1, 1, 1, 0, 0, 0, 1, 0)
  soiltype = 3; pgs = 0

  open(10, file='cwe_in.bin', access='stream', form='unformatted', status='old', action='read')
  open(11, file='cwe_out.bin', access='stream', form='unformatted', status='replace', action='write')
  read(10) ncol, dt, yearlen
  do ic = 1, ncol
    read(10) vegtyp, lat, julian, snowh, tv0, troot, shdmax, sfctmp, canliq, canice, tv, fcev, fctr, tg, frz
    call transfer_mp_parameters(4, vegtyp, soiltype, 1, 4, 0, parameters)
    lai = 0.0; sai = 0.0
    call phenology(parameters, vegtyp, 0, snowh, tv0, lat, yearlen, julian, &
                   lai, sai, troot, elai, esai, igs, pgs, fb)
    ! NOAHMP_SFLX :864-865 (dveg=4) and :874-875, verbatim
    FVEG = SHDMAX
    IF(FVEG <= 0.05) FVEG = 0.05
    IF(parameters%urban_flag .OR. VEGTYP == parameters%ISBARREN) FVEG = 0.0
    IF(ELAI+ESAI == 0.0) FVEG = 0.0
    ! ATM :1216 (OPT_SNF=1), verbatim
    BDFALL = MIN(120.0,67.92+51.25*EXP((SFCTMP-TFRZ)/2.59))
    frozen_canopy = (frz == 1)
    call canwater(parameters, vegtyp, dt, fcev, fctr, elai, esai, tg, fveg, ic, 1, &
                  bdfall, frozen_canopy, canliq, canice, tv, cmc, ecan, etran, &
                  fwet, qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc)
    write(11) lai, sai, elai, esai, fb, fveg, bdfall, canliq, canice, tv, fwet, cmc, ecan, etran, qfrzc, qmeltc
  end do
  close(10); close(11)
end program canwater_oracle
