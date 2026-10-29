"""Write the exact five-entry v9 plan after tools are frozen."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
entries=[('v45-rc-cold-swiss','entry_prime_swiss.sh',15),('v45-rc-cold-prod','entry_prime_prod.sh',30),('v45-rc-cold-wn3','entry_prime_wn3.sh',55),('v45-rc-warm','entry_rw_warm.sh',45),('v45-rc-rw3','entry_rw3.sh',75)]
steps=[dict(lane=lane,path=str(D/file),sha256=hashlib.sha256((D/file).read_bytes()).hexdigest(),cap_min=cap) for lane,file,cap in entries]
(D.parent/'NIGHT_PLAN_V6.json').write_text(json.dumps(dict(schema='v45-rc.night-plan.v1',steps=steps,scheduling=dict(RC_cap_min=220,BallBerry_after_min=0,order='RC RW1-RW4; BallBerry CANCELLED manager15:25:42Z',hard_end_utc='2026-10-29T05:30:00Z',latest_Nightly_DONE_approx='2026-10-28T23:50:00Z',overhead_margin_min=120),O72=dict(min_free_GiB=411,own_launcher_precheck=True,on_low='ARM O72 HOLD; no invocation/registry START'),paths=dict(tools=str(D),source='<USER_HOME>/wrf_gpu2_lanes/v45-rc/archive',bundle='<USER_HOME>/wrf_gpu2_lanes/v45-rc/runtime/bundle',short_session='<DATA_ROOT>/wrf_gpu2/scratch/v45-rc/session_254c48237452_gpu_v6',rw3_session='<DATA_ROOT>/wrf_gpu2/scratch/v45-rc/runs/rw3_<UTC>_<PID>',DATA_writes_before_GA_START=0)),indent=2)+'\n')
print('NIGHT_PLAN.json created; CPU exact-call smokes and manager GPU admission PENDING')
