"""Native SWR93 argument/result stream with only diagnostic writes added."""
import hashlib
import json
from pathlib import Path
import re
import sys
from transcribe_gfdl import read_routine

OUTPUTS={'fswc','hswc','ufswc','dfswc','fswl','hswl','ufswl','dfswl',
         'gdfvb','gdfnb','gdfvd','gdfnd'}
BOUNDS={'ids','ide','jds','jde','kds','kde','ims','ime','jms','jme','kms','kme',
        'its','ite','jts','jte','kts','kte'}
CLOUD={'camt','ktopsw','kbtmsw','crr','ctt'}


def instrument(path):
    original=path.read_text();r=read_routine(path)
    inputs=[k for k in r.args if k not in OUTPUTS|BOUNDS|{'nclds'}]
    outputs=[k for k in r.args if k in OUTPUTS]

    def operand(name,output=False):
        if name=='kbtmsw':return name+'(:,1:NCLDS(ITS)+2)'
        if name in CLOUD:
            return name+'(:,:,1:NCLDS(ITS)+1)' if name in {'crr','ctt'} else name+'(:,1:NCLDS(ITS)+1)'
        if output and name in {'hswc','hswl'}:return name+'(:,1:L)'
        return name

    before='      WRITE(97) L,NCLDS(ITS)\n'
    before+=''.join('      WRITE(97) '+operand(k)+'\n' for k in inputs)
    after=''.join('      WRITE(97) '+operand(k,True)+'\n' for k in outputs)
    begin=re.search(r'(?im)^\s*SUBROUTINE SWR93\(',original)
    end=re.search(r'(?im)^\s*END SUBROUTINE SWR93',original)
    assert begin and end
    body=original[begin.start():end.start()]
    marker='      MYIS1=MYIS+1    ! ??\n'
    assert body.count(marker)==1
    early='      IF (KCLDS .EQ. 0)  RETURN\n'
    assert body.count(early)==1
    early_dump='      IF (KCLDS .EQ. 0) THEN\n'+after+'      ENDIF\n'
    body=body.replace(marker,marker+before).replace(early,early_dump+early)+after
    modified=original[:begin.start()]+body+original[end.start():]
    assert modified.replace(before,'').replace(early_dump,'').replace(after,'')==original
    path.write_text(modified)
    path.with_suffix('.swr93_capture.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_writes_restores_original=True,inputs=inputs,outputs=outputs,
        arguments=r.args,shapes=r.shapes,types=r.types,
        scope='unit97 source operands; cloud slots1..NCLDS+1 and KBTMSW sentinel at NCLDS+2; heating only1..L; no physical change; finalBYTE required',
        cloud_slot_offsets={k:2 if k=='kbtmsw' else 1 for k in CLOUD}),indent=2)+'\n')


if __name__=='__main__':instrument(Path(sys.argv[1]))
