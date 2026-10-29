"""Manager creates this ROOT receipt only after ALISIOS Nightly DONE; never auto-creates it."""
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1]);j=json.loads(p.read_text())
assert j['schema']=='v045-rc.nightly-done.v1'and j['Nightly_DONE']is True
assert j['by']=='manager'
parse=lambda t:datetime.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()
assert parse(j['completed_utc'])<=parse(sys.argv[2])<parse(sys.argv[3])
assert parse(sys.argv[2])<=datetime.datetime.now(datetime.timezone.utc).timestamp()<parse(sys.argv[3])
print('NIGHTLY_DONE admission valid')
