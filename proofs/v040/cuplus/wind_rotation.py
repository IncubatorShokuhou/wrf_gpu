"""Controlled nonzero transverse-wind inputs for pristine CMT callers."""
def bind(text,scheme,degrees):
    if degrees==0:return text
    decl=f'''  real,parameter :: wind_angle={float(degrees):.9e}
  real,parameter :: wind_cos=cos(wind_angle*acos(-1.)/180.),wind_sin=sin(wind_angle*acos(-1.)/180.)
  real :: wind_u,wind_v
'''
    anchor='  ! --- parse case id ---' if scheme=='mskf' else '  character(len=32)::arg\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,decl+anchor if scheme=='mskf' else anchor+decl)
    if scheme=='mskf':
        anchor='  U(:,kts:kte,:)=U(:,kts:kte,:)+BACKGROUND_WIND\n'
        rotation='''  do jj=jms,jme
    do ii=ims,ime
      do k=1,KX
        wind_u=U(ii,k,jj);wind_v=V(ii,k,jj)
        U(ii,k,jj)=wind_cos*wind_u-wind_sin*wind_v
        V(ii,k,jj)=wind_sin*wind_u+wind_cos*wind_v
      enddo
    enddo
  enddo
'''
    else:
        anchor='  u_phy(:,1:KX,:)=u_phy(:,1:KX,:)+background_wind\n'
        rotation='''  do k=1,KX
    wind_u=u_phy(1,k,1);wind_v=v_phy(1,k,1)
    u_phy(1,k,1)=wind_cos*wind_u-wind_sin*wind_v
    v_phy(1,k,1)=wind_sin*wind_u+wind_cos*wind_v
  enddo
'''
    assert text.count(anchor)==1
    return text.replace(anchor,anchor+rotation)
