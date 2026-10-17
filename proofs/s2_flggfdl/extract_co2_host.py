"""Prototype source-preserving, once-at-init native CO2 table builder.

No forecast-step routine is included. The physical initializer bodies are
copied byte-for-byte from WRF; the executable owns only explicit grid input
and table output. Runtime/product integration is a separate measured decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROUTINES=('ANTEMP','COEINT','CO2INS','CO2INT','CO2IN1','CO2PTZ','PATH','QINTRP','QUADSR','SIGP','SINTR2','CO2O3')
CO2=('CO251','CDT51','CDT58','C2D51','C2D58','CO258','STEMP','GTEMP',
     'CO231','CO238','C2D31','C2D38','CDT31','CDT38','CO271','CO278',
     'C2D71','C2D78','CDT71','CDT78','CO2M51','CO2M58','CDTM51','CDTM58','C2DM51','C2DM58')


def main(source,output,receipt):
    original=source.read_text()
    prefix=original[:original.index('CONTAINS')]
    prefix=prefix.replace('MODULE MODULE_RA_GFDLETA','MODULE GFDLETA_CO2_INIT',1)
    prefix=re.sub(r'(?im)^\s*USE MODULE_CONFIGURE[^\n]*\n','\n',prefix)
    prefix=re.sub(r'(?ims)^#ifdef FERRIER_GFDL\n.*?^#endif\n','',prefix)
    bodies=[];hashes={}
    for name in ROUTINES:
        start=re.search(r'(?im)^[ \t]*(?:REAL[ \t]+)?(?:SUBROUTINE|FUNCTION)[ \t]+'+name+r'\b',original)
        end=re.search(r'(?im)^[ \t]*END[ \t]+(?:SUBROUTINE|FUNCTION)[ \t]+'+name+r'\b[^\n]*\n',original)
        assert start and end and start.start()<end.end(),name
        body=original[start.start():end.end()]
        bodies.append(body);hashes[name]=hashlib.sha256(body.encode()).hexdigest()
    module=prefix+'CONTAINS\n'+''.join(bodies)+'END MODULE GFDLETA_CO2_INIT\n'
    program='''program co2_host_builder
  use GFDLETA_CO2_INIT
  implicit none
  integer :: nz,u,n
  real :: ptop,pptop,pcld
  real,allocatable :: full(:),half(:)
  character(len=1024) :: input,output
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u) nz,ptop
  allocate(full(nz+1),half(nz));read(u)full,half;close(u)
  pptop=ptop/1000.
  PTOPC=[105000.,64200.,35000.,15000.]
  ! CO2PTZ, called by CO2O3, computes the authoritative final LTOP.
  ! Do not overwrite that native initialized state after the call.
  call CO2O3(full,half,pptop,nz,nz+1,nz+2)
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
  write(u)nz,LTOP,PTOPC
'''
    program+=''.join('  write(u)'+name+'\n' for name in CO2)
    program+='  close(u)\nend program\n'
    text=module+program;output.write_text(text)
    receipt.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        extracted_sha256=hashlib.sha256(text.encode()).hexdigest(),routine_sha256=hashes,
        body_identity=True,scope='prototype CPU initializer only, no forecast-step or runtime oracle calls; startup/tables/packaging decision pending'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','output','receipt'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.source,a.output,a.receipt)
