! Diagnostic transcription of SFDDAGD:1057-1058, unmodified model constants.
! Dump every REAL operand after two failed precision hypotheses; no model body patch.
program saturation_operands
 use module_model_constants
 implicit none
 integer :: n,i
 character(len=512) :: input,output
 real,allocatable :: t(:),p(:),a(:),b(:),c(:),d(:),e(:),f(:),es(:),h(:),den(:),num(:),q(:),full(:)
 call get_command_argument(1,input);call get_command_argument(2,output)
 open(10,file=trim(input),access='stream',form='unformatted',status='old')
 read(10)n
 allocate(t(n),p(n),a(n),b(n),c(n),d(n),e(n),f(n),es(n),h(n),den(n),num(n),q(n),full(n))
 read(10)t,p
 do i=1,n
  a(i)=t(i)-SVPT0
  b(i)=t(i)-SVP3
  c(i)=SVP2*a(i)
  d(i)=c(i)/b(i)
  e(i)=EXP(d(i))
  f(i)=SVP1*e(i)
  es(i)=f(i)*10.0
  h(i)=p(i)/100.0
  den(i)=h(i)-es(i)
  num(i)=EP_2*es(i)
  q(i)=num(i)/den(i)
  full(i)=EP_2*(SVP1*EXP(SVP2*(t(i)-SVPT0)/(t(i)-SVP3))*10.0) / &
               (p(i)/100.0-(SVP1*EXP(SVP2*(t(i)-SVPT0)/(t(i)-SVP3))*10.0))
 enddo
 open(11,file=trim(output),access='stream',form='unformatted',status='replace')
 write(11)SVP1,SVP2,SVP3,SVPT0,EP_2,R_d,R_v
 write(11)a,b,c,d,e,f,es,h,den,num,q,full
 close(10);close(11)
end program saturation_operands
