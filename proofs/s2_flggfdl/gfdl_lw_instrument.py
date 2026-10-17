"""Capture actual LWR88 arguments plus initialized native CO2 module tables."""
import hashlib
import json
from pathlib import Path
import re
import sys
from transcribe_gfdl import read_routine
from transcribe_gfdl_lw import CO2
from gfdl_sw_instrument import BOUNDS

OUTPUTS={'heatra','grnflx','topflx'}


def instrument(path):
    original=path.read_text();routine=read_routine(path,'lwr88')
    inputs=[k for k in routine.args if k not in OUTPUTS|BOUNDS]
    outputs=[k for k in routine.args if k in OUTPUTS]
    before='      WRITE(98) L\n'
    before+=''.join('      WRITE(98) '+name+'\n' for name in (*inputs,*CO2))
    after=''.join('      WRITE(98) '+name+'\n' for name in outputs)
    begin=re.search(r'(?im)^\s*SUBROUTINE LWR88\(',original)
    end=re.search(r'(?im)^\s*END SUBROUTINE LWR88',original)
    assert begin and end
    body=original[begin.start():end.start()]
    # All argument dimensions are resolved before the first physical loop.
    marker='      MYIS=its; MYIE=ite\n'
    assert body.count(marker)==1
    modified=original[:begin.start()]+body.replace(marker,marker+before)+after+original[end.start():]
    assert modified.replace(before,'').replace(after,'')==original
    path.write_text(modified)
    path.with_suffix('.lwr88_capture.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_writes_restores_original=True,inputs=inputs,outputs=outputs,
        arguments=routine.args,shapes=routine.shapes,types=routine.types,co2_tables=CO2,
        scope='unit98 diagnostic writes only; initialized native CO2 provider is grid-specific; finalBYTE required'),indent=2)+'\n')


if __name__=='__main__':instrument(Path(sys.argv[1]))
