"""Original3487..3550 interpolation, in original i/k/ii/jj operation order."""
import jax.numpy as jnp


def interpolate_corners(corners,di,dk,dii,djj):
    """Corners are ordered (jj,ii,i,k), each index0/1; no stack allocation."""
    def at(jj,ii,i,k):return corners[8*jj+4*ii+2*i+k]
    dproc1=at(0,0,0,0)+di*(at(0,0,1,0)-at(0,0,0,0))
    dproc2=at(0,0,0,1)+di*(at(0,0,1,1)-at(0,0,0,1))
    iproc1=dproc1+dk*(dproc2-dproc1)
    dproc1=at(0,1,0,0)+di*(at(0,1,1,0)-at(0,1,0,0))
    dproc2=at(0,1,0,1)+di*(at(0,1,1,1)-at(0,1,0,1))
    gproc1=dproc1+dk*(dproc2-dproc1)
    tmp1=iproc1+dii*(gproc1-iproc1)
    dproc1=at(1,0,0,0)+di*(at(1,0,1,0)-at(1,0,0,0))
    dproc2=at(1,0,0,1)+di*(at(1,0,1,1)-at(1,0,0,1))
    iproc1=dproc1+dk*(dproc2-dproc1)
    dproc1=at(1,1,0,0)+di*(at(1,1,1,0)-at(1,1,0,0))
    dproc2=at(1,1,0,1)+di*(at(1,1,1,1)-at(1,1,0,1))
    gproc1=dproc1+dk*(dproc2-dproc1)
    tmp2=iproc1+dii*(gproc1-iproc1)
    return tmp1+djj*(tmp2-tmp1)


def access_lookup_table(table,jj,ii,i,k,index,dum1,dum2,dum4,dum5,*,xp=jnp):
    r=lambda a:xp.asarray(a,dtype=table.dtype)
    jj0,ii0,i0,k0,index0=jj-1,ii-1,i-1,k-1,index-1
    corners=tuple(table[jj0+dj,ii0+dii,i0+di,k0+dk,index0]
                  for dj in range(2) for dii in range(2) for di in range(2) for dk in range(2))
    return interpolate_corners(corners,dum1-r(i),dum2-r(k),dum4-r(ii),dum5-r(jj))
