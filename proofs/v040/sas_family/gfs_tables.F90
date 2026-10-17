! Visibility-only exports are added to a lane copy of module_gfs_funcphys.
! No routine or table-generating expression is changed.
program dump_gfs_tables
  use module_gfs_funcphys
  implicit none
  call gfuncphys()
  open(20,file='gfs_tables.bin',access='stream',form='unformatted',status='replace')
  write(20) c1xtdp,c2xtdp,c1xthe,c2xthe,c1ythe,c2ythe, &
            c1xtlcl,c2xtlcl,c1ytlcl,c2ytlcl,c1xma,c2xma,c1yma,c2yma
  write(20) tbtdp,tbthe,tbtlcl,tbtma,tbqma
  close(20)
end program
