"""Admission for every exact CLI arm, including CPU --dry-run smokes (E225)."""
from decimal import Decimal
from pathlib import Path
from gpuwrf.io.namelist_check import _parse_wrf_namelist

def check(args):
    n=_parse_wrf_namelist(Path(args.namelist or args.input_dir/'namelist.input').read_text())
    dom=n['domains'];tc=n['time_control']
    def first(key,default=None):
        value=dom.get(key,[default]);return value[0] if isinstance(value,(list,tuple)) else value
    dt=Decimal(str(first('time_step')))+Decimal(str(first('time_step_fract_num',0)))/Decimal(str(first('time_step_fract_den',1)))
    hours=args.hours
    assert hours is not None,'all sealed release arms explicitly declare hours'
    steps=Decimal(str(hours))*3600/dt
    assert steps==steps.to_integral_value(),('forecast is not aligned to root dt',hours,str(dt))
    print('EXACT CLI ALIGNMENT PASS',hours,'hours root_dt',dt,'root_steps',steps,flush=True)
