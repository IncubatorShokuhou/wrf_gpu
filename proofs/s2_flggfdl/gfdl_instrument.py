"""Diagnostic writes at unchanged ETARA/RADTN operand boundaries (unit95)."""
from pathlib import Path
import hashlib
import json
import sys


def instrument(path):
    original=path.read_text()
    markers=[('      CALL RADTN (DT,TFLIP,QFLIP,QWFLIP,QIFLIP,',
        '      WRITE(95) KTE,TFLIP(:,KTS:KTE,:),PFLIP(:,KTS:KTE,:),P8WFLIP, &\n'
        '     & QFLIP(:,KTS:KTE,:),QWFLIP(:,KTS:KTE,:),QIFLIP(:,KTS:KTE,:),CUTOP,CUBOT\n'),
        ('      IF (CNCLD) THEN',
        '      WRITE(95) LM,TMID,PMID,QMID,QWMID,QIMID,CSMID,CUTOP,CUBOT\n')]
    modified=original
    for needle,dump in markers:
        assert modified.count(needle)==1
        modified=modified.replace(needle,dump+needle)
    restored=modified
    for _,dump in markers:restored=restored.replace(dump,'')
    assert restored==original
    path.write_text(modified)
    path.with_suffix('.instrumentation.json').write_text(json.dumps(dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        instrumented_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        stripping_writes_restores_original=True,
        scope='unchanged physical/control operands, diagnostic writes only; final BYTE required'),indent=2)+'\n')


if __name__=='__main__':instrument(Path(sys.argv[1]))
