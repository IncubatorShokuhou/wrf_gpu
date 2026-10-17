"""Extract unchanged phy_prep mass-Z/T8W statements for explicit caller inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def main(source,output):
    original=source.read_text()
    start=original.index('!  compute z at p points or half levels')
    end=original.index('! calculate hydrostatic pressure',start)
    # End precisely after the top extrapolation loops, before the next section.
    body=original[start:end]
    assert 't8w(i,kde,j)' in body and 'enddo' in body.lower()
    text='''program native_phy_prep
implicit none
integer :: nc,nz,i,j,k,u,row,i_start,i_end,j_start,j_end,k_start,k_end,kte,kde
real :: z0,z1,z2,w1,w2
real,allocatable :: t_phy(:,:,:),p_phy(:,:,:),z_at_w(:,:,:),z(:,:,:),p8w(:,:,:),t8w(:,:,:),fzm(:),fzp(:)
character(len=1024) :: input,output
call get_command_argument(1,input);call get_command_argument(2,output)
open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
read(u)nc,nz
allocate(t_phy(nc,nz,1),p_phy(nc,nz,1),z_at_w(nc,nz+1,1),z(nc,nz,1),p8w(nc,nz+1,1),t8w(nc,nz+1,1),fzm(nz+1),fzp(nz+1))
read(u)t_phy,p_phy,z_at_w,fzm,fzp
close(u)
i_start=1;i_end=nc;j_start=1;j_end=1;k_start=1;k_end=nz;kte=nz+1;kde=kte
'''+body+'''
open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
write(u)t8w
close(u)
end program
'''
    output.write_text(text)
    output.with_suffix('.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        unchanged_region_sha256=hashlib.sha256(body.encode()).hexdigest(),
        scope='original phy_prep mass-Z/interface-temperature statements; explicit supplied heights/weights; not a forecast replay'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.source,a.output)
