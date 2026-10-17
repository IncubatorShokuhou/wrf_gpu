module table_output
  implicit none
  integer :: unit
  interface emit
    module procedure emit0,emit1,emit2,emit3
  end interface
contains
  subroutine header(name,rank,dimensions)
    character(len=*),intent(in) :: name
    integer,intent(in) :: rank,dimensions(3)
    character(len=64) :: padded
    padded=name
    write(unit) padded,rank,dimensions
  end subroutine
  subroutine emit0(name,a)
    character(len=*),intent(in) :: name
    real,intent(in) :: a
    call header(name,0,[1,1,1]); write(unit) a
  end subroutine
  subroutine emit1(name,a)
    character(len=*),intent(in) :: name
    real,intent(in) :: a(:)
    call header(name,1,[size(a),1,1]); write(unit) a
  end subroutine
  subroutine emit2(name,a)
    character(len=*),intent(in) :: name
    real,intent(in) :: a(:,:)
    call header(name,2,[size(a,1),size(a,2),1]); write(unit) a
  end subroutine
  subroutine emit3(name,a)
    character(len=*),intent(in) :: name
    real,intent(in) :: a(:,:,:)
    call header(name,3,shape(a)); write(unit) a
  end subroutine
end module
program flg_tables
  use table_output
  use band
  use module_ozone
  use ice2
  use water1
  use rain1
  use graup1
  use rayle1
  implicit none
  character(len=1024) :: path
  call get_command_argument(1,path)
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='replace')
  call emit('hk_1',hk_1); call emit('fko3_1',fko3_1)
  call emit('hk_2',hk_2); call emit('coeh2o_2',coeh2o_2)
  call emit('hk_3',hk_3); call emit('coeh2o_3',coeh2o_3)
  call emit('hk_4',hk_4); call emit('coeh2o_4',coeh2o_4)
  call emit('hk_5',hk_5); call emit('coeh2o_5',coeh2o_5)
  call emit('hk_6',hk_6); call emit('coeh2o_6',coeh2o_6)
  call emit('hk_7',hk_7); call emit('coeh2o_7',coeh2o_7)
  call emit('hk_8',hk_8); call emit('coeh2o_8',coeh2o_8)
  call emit('hk_9',hk_9); call emit('coeh2o_9',coeh2o_9)
  call emit('hk_10',hk_10); call emit('coeh2o_10',coeh2o_10)
  call emit('coech4_10',coech4_10); call emit('coen2o_10',coen2o_10)
  call emit('hk_11',hk_11); call emit('coeh2o_11',coeh2o_11)
  call emit('coech4_11',coech4_11); call emit('coen2o_11',coen2o_11)
  call emit('hk_12',hk_12); call emit('coeo3_12',coeo3_12); call emit('coeh2o_12',coeh2o_12)
  call emit('hk_13',hk_13); call emit('coeh2o_13',coeh2o_13)
  call emit('hk_14',hk_14); call emit('coehca_14',coehca_14); call emit('coehcb_14',coehcb_14)
  call emit('hk_15',hk_15); call emit('coehca_15',coehca_15); call emit('coehcb_15',coehcb_15)
  call emit('hk_16',hk_16); call emit('coeh2o_16',coeh2o_16)
  call emit('hk_17',hk_17); call emit('coeh2o_17',coeh2o_17)
  call emit('hk_18',hk_18); call emit('coeh2o_18',coeh2o_18)
  call emit('ozone_pres',pres); call emit('ozone',ozone)
  call emit('ice_ap',ap); call emit('ice_bp',bp); call emit('ice_cps',cps); call emit('ice_cpir',cpir)
  call emit('water_re',re); call emit('water_fl',fl)
  call emit('water_bz',bz); call emit('water_wz',wz); call emit('water_gz',gz)
  call emit('rain_rwc',rwc); call emit('rain_brn',brn); call emit('rain_wrnf',wrnf); call emit('rain_grn',grn)
  call emit('graup_gwc',gwc); call emit('graup_bg',bg); call emit('graup_wgf',wgf); call emit('graup_gg',gg)
  call emit('rayle_ri',ri)
  close(unit)
end program
