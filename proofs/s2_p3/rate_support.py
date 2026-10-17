"""Private original-WRF rate observations; adds I/O, no arithmetic changes."""
import re

import numpy as np

SCALARS = ('qc','qr','qv','nc','nr','t','rho','qvs','qvi','qccon','qcnuc','qcevp',
           'qcacc','qcaut','qrcon','qrevp','qcon_satadj')
CATEGORIES = ('qccol','qchetc','qcheti','qcshd')
NAMES = SCALARS + CATEGORIES


def instrument_rates(source):
    arguments = ','.join(name+'(i,k)' if index < 9 else name for index,name in enumerate(NAMES))
    for tag, pattern in ((1,r'(?im)^(\s*sinks\s*=\s*\(qcaut\+qcacc[^\n]*)$'),
                         (2,r'(?im)^(\s*iice_loop3:\s*do\s+iice[^\n]*)$')):
        source,count = re.subn(pattern, lambda match: f'\n call s2_p3_rates({tag},it,i,k,ncat,{arguments})\n'+match[0], source)
        assert count == 1, (tag,count)
    routine = f'''
subroutine s2_p3_rates(tag,it,i,k,ncat_rate,{','.join(NAMES)})
integer,intent(in) :: tag,it,i,k,ncat_rate
integer :: unit
real,intent(in) :: {','.join(SCALARS)}
real,intent(in) :: {','.join(name+'(ncat_rate)' for name in CATEGORIES)}
open(newunit=unit,file='rates.bin',access='stream',form='unformatted',status='unknown',position='append')
write(unit) tag,it,i,k,ncat_rate
write(unit) {','.join(NAMES)}
close(unit)
end subroutine s2_p3_rates
'''
    source,count = re.subn(r'(?im)^(\s*end\s+module\b.*)$',lambda match:routine+match[0],source)
    assert count == 1
    return source


def records(path, tag):
    data=path.read_bytes(); position=0; result={}
    while position<len(data):
        observed,it,i,k,ncat=np.frombuffer(data,np.int32,count=5,offset=position)
        position+=20
        values={}
        for name in NAMES:
            count=int(ncat) if name in CATEGORIES else 1
            value=np.frombuffer(data,np.float32,count=count,offset=position)
            position+=count*4
            values[name]=value if name in CATEGORIES else value[0]
        if observed==tag:
            result[int(it),int(i),int(k)]=values
    assert position==len(data)
    return result
