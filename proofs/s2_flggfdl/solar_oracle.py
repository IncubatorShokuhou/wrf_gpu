"""Extract unchanged SOLARD body and emit a small independent native driver."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main(source,output):
    original=source.read_text()
    start=re.search(r'(?im)^\s*SUBROUTINE SOLARD\(',original).start()
    end=re.search(r'(?im)^\s*END SUBROUTINE SOLARD[^\n]*\n',original).end()
    body=original[start:end]
    # SOLARD only reads/writes the module scalar R1; all physical statements,
    # declarations and DATA remain byte-identical to the pristine body.
    text='module solar_native\nreal :: R1\ncontains\n'+body+'end module\n'
    text+='''program solar_oracle
use solar_native
implicit none
integer :: y,m,d,h,u,i,j
integer :: years(8)=[1999,2001,2009,2025,2026,2000,2008,2024]
character(len=1024) :: output
call get_command_argument(1,output)
open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
do i=1,8
 y=years(i)
 do m=1,12
  d=1+mod(m*3,27)
  do j=1,4
   h=[0,6,12,23](j)
   call SOLARD(h,d,m,y)
   write(u)h,d,m,y,R1
  enddo
 enddo
enddo
close(u)
end program
'''
    # Fortran does not allow indexing an array constructor directly.
    text=text.replace('integer :: y,m,d,h,u,i,j','integer :: y,m,d,h,u,i,j\ninteger :: hours(4)=[0,6,12,23]').replace('h=[0,6,12,23](j)','h=hours(j)')
    output.write_text(text)
    output.with_suffix('.json').write_text(json.dumps(dict(
        native_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
        source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        extracted_body_BYTE=True,records=384),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.source,a.output)
