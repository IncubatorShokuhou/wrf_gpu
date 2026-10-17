! WRF 4.7.1 share/module_model_constants.F values the surface driver passes (E83).
real, parameter :: r_d=287., r_v=461.6, cp=7.*r_d/2., g=9.81, rcp=r_d/cp, xlv=2.5e6
real, parameter :: ep_1=r_v/r_d-1., ep_2=r_d/r_v, karman=0.4, eomeg=7.2921e-5, stbolt=5.67051e-8
real, parameter :: svp1=0.6112, svp2=17.67, svp3=29.65, svpt0=273.15, p1000mb=100000.
