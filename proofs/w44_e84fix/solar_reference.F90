! CPU quotient intervention on the released REAL solar-clock expression graph.
! q(mode=0) is IEEE RN. mode=1 is RN reciprocal then RN multiply (exact for
! constant-reciprocal HLO; a disclosed div.full surrogate for runtime division).
! mode=2/3 moves an isolated RN quotient down/up two representable values.
module e84_clock_literal
use iso_c_binding
implicit none
contains
real(c_float) function q(a,b,mode,target,id)
 real(c_float),intent(in)::a,b
 integer,intent(in)::mode,target,id
 real(c_float)::r
 q=a/b
 ! Candidate: only the audited 9 MUST solar quotient families are repaired.
 ! The two D families (GMT and source scale) keep the rank's reciprocal model.
 if (mode==4) then
  if (id/=7.and.id/=11) return
  r=1./b
  q=a*r
  return
 endif
 if (mode==0.or.(target/=0.and.target/=id)) return
 if (mode==1) then
  r=1./b
  q=a*r
 else if (q/=0.) then
  if (mode==2) then
   q=nearest(nearest(q,-1.),-1.)
  else if (mode==3) then
   q=nearest(nearest(q,1.),1.)
  endif
 endif
end function
subroutine replay(n,lat,lon,jbase,utc,lead,solar,mode,target,csza,hrang,aux,aa,bb) bind(C)
 integer(c_int),value::n,mode,target
 real(c_float),intent(in)::lat(n),lon(n)
 real(c_float),value::jbase,utc,lead,solar
 real(c_float),intent(out)::csza(n),hrang(n),aux(6),aa(11),bb(11)
 real(c_float)::jul,minute,absolute,frac,deg,dpd,sxday,sxlong,declin,da,twice,eot,sm,xt24,gmt,hours,local,lr,c,rjul,ecc,solcon
 integer::i
 aa=0.;bb=[60.,1440.,180.,365.,365.,60.,60.,60.,15.,365.,1368.22]
 aa(1)=lead;minute=q(lead,60.,mode,target,1)
 absolute=utc+minute;aa(2)=absolute;frac=q(absolute,1440.,mode,target,2)
 jul=(jbase-1.)+frac
 aa(3)=3.141592653589793;deg=q(aa(3),180.,mode,target,3)
 aa(4)=360.;dpd=q(360.,365.,mode,target,4)
 if (jul>=80.) then
  sxday=jul-80.
 else
  sxday=jul+285.
 endif
 sxlong=(dpd*sxday)*deg
 declin=asin(sin(23.5*deg)*sin(sxlong))
 aa(5)=6.2831853071795862*(jul-1.);da=q(aa(5),365.,mode,target,5);twice=2.*da
 eot=((((.000075+.001868*cos(da))-.032077*sin(da))-.014615*cos(twice))-.04089*sin(twice))*229.18
 aa(6)=solar;sm=q(solar,60.,mode,target,6)
 xt24=mod(sm,1440.)+eot
 aa(7)=utc;gmt=q(utc,60.,mode,target,7)
 aa(8)=xt24;hours=q(xt24,60.,mode,target,8)
 aa(10)=jul*360.;rjul=q(aa(10),365.,mode,target,10)*deg
 ecc=(((1.000110+.034221*cos(rjul))+.001280*sin(rjul))+.000719*cos(2.*rjul))+.000077*sin(2.*rjul)
 solcon=1370.*ecc;aa(11)=solcon
 aux=[jul,declin,solcon,deg,eot,q(solcon,1368.22,mode,target,11)]
 do i=1,n
  local=(gmt+hours)+q(lon(i),15.,mode,target,9)
  hrang(i)=(15.*(local-12.))*deg
  lr=lat(i)*deg
  c=sin(lr)*sin(declin)+(cos(lr)*cos(declin))*cos(hrang(i))
  csza(i)=min(max(c,-1.),1.)
 enddo
end subroutine
end module
