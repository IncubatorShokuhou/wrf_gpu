"""Write the exact four-entry v9 plan after tools are frozen."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
entries=[('w44-rc-cold-swiss','entry_prime_swiss.sh',15),('w44-rc-cold-prod','entry_prime_prod.sh',30),('w44-rc-cold-wn3','entry_prime_wn3.sh',55),('w44-rc-warm','entry_rw_warm.sh',45),('w44-rc-rw3','entry_rw3.sh',75)]
steps=[dict(lane=lane,path=str(D/file),sha256=hashlib.sha256((D/file).read_bytes()).hexdigest(),cap_min=cap) for lane,file,cap in entries]
(D.parent/'NIGHT_PLAN_V5.json').write_text(json.dumps(dict(schema='w44-rc.night-plan.v1',steps=steps),indent=2)+'\n')
print('NIGHT_PLAN.json created; CPU exact-call smokes and manager GPU admission PENDING')
