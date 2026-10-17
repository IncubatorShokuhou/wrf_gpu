program c
implicit none
   real,parameter :: blckdr=0.0063,cn=0.75
   real,parameter :: eps1=1.e-12,epsl=0.32,epsru=1.e-7,epsrs=1.e-7
   real,parameter :: el0max=1000.,el0min=1.,elfc=0.23*0.5
   real,parameter :: alph=0.30,beta=1./273.,g=9.81,btg=beta*g
   real,parameter :: a1=0.659888514560862645,a2x=0.6574209922667784586
   real,parameter :: b1=11.87799326209552761,b2=7.226971804046074028
   real,parameter :: c1=0.000830955950095854396
   real,parameter :: adnh= 9.*a1*a2x*a2x*(12.*a1+3.*b2)*btg*btg
   real,parameter :: adnm=18.*a1*a1*a2x*(b2-3.*a2x)*btg
   real,parameter :: bdnh= 3.*a2x*(7.*a1+b2)*btg,bdnm= 6.*a1*a1
   real,parameter :: aeqh=9.*a1*a2x*a2x*b1*btg*btg &
                         +9.*a1*a2x*a2x*(12.*a1+3.*b2)*btg*btg
   real,parameter :: aeqm=3.*a1*a2x*b1*(3.*a2x+3.*b2*c1+18.*a1*c1-b2) &
                         *btg+18.*a1*a1*a2x*(b2-3.*a2x)*btg
   real,parameter :: requ=-aeqh/aeqm
   real,parameter :: epsgh=1.e-9,epsgm=requ*epsgh
   real,parameter :: ubryl=(18.*requ*a1*a1*a2x*b2*c1*btg &
                            +9.*a1*a2x*a2x*b2*btg*btg)   &
                           /(requ*adnm+adnh)
   real,parameter :: ubry=(1.+epsrs)*ubryl,ubry3=3.*ubry
   real,parameter :: aubh=27.*a1*a2x*a2x*b2*btg*btg-adnh*ubry3
   real,parameter :: aubm=54.*a1*a1*a2x*b2*c1*btg  -adnm*ubry3
   real,parameter :: bubh=(9.*a1*a2x+3.*a2x*b2)*btg-bdnh*ubry3
   real,parameter :: bubm=18.*a1*a1*c1             -bdnm*ubry3
   real,parameter :: cubr=1.-ubry3,rcubr=1./cubr
   real,parameter :: r_d=287., r_v=461.6, cp=7.*r_d/2., ep_1=r_v/r_d-1.
   real :: cpv, elocp, rc02, c0
   cpv=cp; c0=0.55
   elocp=2.72e6/cpv; rc02=2.0/(c0*c0)
   write(*,'(A,1X,Z8.8)') 'elfc',elfc,'beta',beta,'btg',btg,'a1',a1,'a2x',a2x,'b1',b1,'b2',b2,'c1',c1, &
     'adnh',adnh,'adnm',adnm,'bdnh',bdnh,'bdnm',bdnm,'aeqh',aeqh,'aeqm',aeqm,'requ',requ,'epsgm',epsgm, &
     'ubryl',ubryl,'ubry',ubry,'ubry3',ubry3,'aubh',aubh,'aubm',aubm,'bubh',bubh,'bubm',bubm,'cubr',cubr,'rcubr',rcubr, &
     'ep_1',ep_1,'cp',cp,'elocp',elocp,'rc02',rc02
end program
