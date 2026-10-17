"""Diagnostic native full RADTN call, initialized static bundle and outputs."""
import hashlib
import json
from pathlib import Path
import re
import sys
from transcribe_gfdl import read_routine
from transcribe_gfdl_lw import CO2
from gfdl_sw_instrument import BOUNDS

OUTPUTS={'glw','gsw','czmean','rswin','rswinc','tendl','tends','cldfra',
         'rswtoa','rlwtoa','cfracl','cfracm','cfrach','acfrst','acfrcv','ncfrst','ncfrcv'}
DOMAIN={'ltop':('i',(3,)),'ptopc':('r',(4,))}


def instrument(path):
    original=path.read_text();routine=read_routine(path,'radtn')
    inputs=[k for k in routine.args if k not in OUTPUTS|BOUNDS]
    outputs=[k for k in routine.args if k in OUTPUTS]
    # Accumulators are both inputs and outputs; freeze their actual start.
    inout=[k for k in ('acfrst','acfrcv','ncfrst','ncfrcv') if k in outputs]
    # ETARA fills only KTS:KTE of its padded mass arrays. RADTN reads exactly
    # those LM=KME-1 active slots; the extra mass slot has no native value.
    active_mass={'t','q','qcw','qice','pflip'}
    before='      WRITE(100) LM,R1\n'
    before+=''.join('      WRITE(100) '+(name+'(:,1:LM,:)' if name in active_mass else name)+'\n' for name in (*inputs,*inout,*CO2,*DOMAIN))
    after=''.join('      WRITE(100) '+name+'\n' for name in outputs)
    begin=re.search(r'(?im)^\s*SUBROUTINE RADTN\(',original)
    end=re.search(r'(?im)^\s*END SUBROUTINE RADTN',original)
    assert begin and end
    body=original[begin.start():end.start()]
    # LM is defined before the first domain loop and before any physics.
    marker='      LP1=LM+1\n';assert body.count(marker)==1
    modified=original[:begin.start()]+body.replace(marker,marker+before)+after+original[end.start():]
    assert modified.replace(before,'').replace(after,'')==original
    path.write_text(modified)
    path.with_suffix('.radtn_capture.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_writes_restores_original=True,inputs=inputs+inout,outputs=outputs,
        arguments=routine.args,shapes=routine.shapes,types=routine.types,
        co2_tables=CO2,domain_static=DOMAIN,environment=('r1',),active_mass_only=sorted(active_mass),
        scope='unit100 diagnostic writes only; accumulator initial state included; native full-outputBYTE required'),indent=2)+'\n')


if __name__=='__main__':instrument(Path(sys.argv[1]))
