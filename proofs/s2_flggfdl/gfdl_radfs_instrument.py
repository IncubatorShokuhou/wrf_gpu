"""Diagnostic-only native RADFS inputs/environment/results, unit99."""
import hashlib
import json
from pathlib import Path
import re
import sys
from transcribe_gfdl import read_routine
from transcribe_gfdl_lw import CO2
from gfdl_sw_instrument import BOUNDS

OUTPUTS={'swh','hlw','flwup','fswup','fswdn','fswdns','fswups','flwdns','flwups','fswdnsc'}


def instrument(path):
    original=path.read_text();r=read_routine(path,'radfs')
    inputs=[k for k in r.args if k not in OUTPUTS|BOUNDS|{'imd','jmd','jndx'}]
    outputs=[k for k in r.args if k in OUTPUTS]
    before='      WRITE(99) L,R1\n'
    before+=''.join('      WRITE(99) '+name+'\n' for name in (*inputs,*CO2))
    after=''.join('      WRITE(99) '+name+'\n' for name in outputs)
    begin=re.search(r'(?im)^\s*SUBROUTINE RADFS\s*&',original)
    end=re.search(r'(?im)^\s*END SUBROUTINE RADFS',original)
    assert begin and end
    body=original[begin.start():end.start()]
    marker='      MYIS=its; MYIE=ite\n';assert body.count(marker)==1
    # The native RETURN follows completion of both selected physics branches.
    ret='      RETURN\n';assert body.count(ret)==1
    body=body.replace(marker,marker+before).replace(ret,after+ret)
    modified=original[:begin.start()]+body+original[end.start():]
    assert modified.replace(before,'').replace(after,'')==original
    path.write_text(modified)
    path.with_suffix('.radfs_capture.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_writes_restores_original=True,inputs=inputs,outputs=outputs,
        arguments=r.args,shapes=r.shapes,types=r.types,co2_tables=CO2,environment=('r1',),
        scope='unit99 diagnostic writes only; exact full caller arrays and dynamic nativeR1; finalBYTE required'),indent=2)+'\n')


if __name__=='__main__':instrument(Path(sys.argv[1]))
