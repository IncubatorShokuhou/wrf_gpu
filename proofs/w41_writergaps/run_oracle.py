"""Run the lane PV/green-roof BEP+BEM oracle arms and save parsed fixtures.

JAX_PLATFORMS=cpu taskset -c 13 nice -n 19 python proofs/w41_writergaps/run_oracle.py --build <dir with fp32/fp64 oracle3> --save <out>
The URBPARM.TBL of each arm is the pristine WRF table with only the named lines replaced.
"""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v040/urban_bep_bem'))
from build_oracle import parse_dump, sha, WRF  # noqa: E402

ARMS={
    'ctl_ac_on':{},
    'ctl_ac_off':{'SW_COND':'0, 0, 0'},
    'pv04_ac_on':{'PV_FRAC_ROOF':'0.4, 0.4, 0.4'},
    'pv10_ac_off':{'PV_FRAC_ROOF':'1.0, 1.0, 1.0','SW_COND':'0, 0, 0'},
    'gr05_ac_on':{'GR_FLAG':'1','GR_FRAC_ROOF':'0.5, 0.5, 0.5'},
    'gr10pv03_ac_on':{'GR_FLAG':'1','GR_FRAC_ROOF':'1.0, 1.0, 1.0','PV_FRAC_ROOF':'0.3, 0.3, 0.3'},
    'gr05irr_ac_off':{'GR_FLAG':'1','GR_FRAC_ROOF':'0.5, 0.5, 0.5','SW_COND':'0, 0, 0',
                      'IRHO':', '.join(['1']*24)},
    'gr05dt_ac_on':{'GR_FLAG':'1','GR_FRAC_ROOF':'0.5, 0.5, 0.5'},
    'gr05t1_ac_on':{'GR_FLAG':'1','GR_FRAC_ROOF':'0.5, 0.5, 0.5','GR_TYPE':'1'},
    'gr05dry_ac_on':{'GR_FLAG':'1','GR_FRAC_ROOF':'0.5, 0.5, 0.5'},
}
# Constructed initial states (pvgr_oracle.F90 env): green soil 4 K above the roof so wall_gr's hfgr is O(10 W/m2).
# Dry soil: SMOIS_URB 0.04 < flux_flat_roof wilting point qw=0.06 -> gx<0 clamp, f4=0, rsveg=rsmax.
ARM_ENV={'gr05dt_ac_on':{'PVGR_TRV_OFFSET':'4.0'},'gr05dry_ac_on':{'PVGR_SMOIS_URB':'0.04'}}


def table(changes):
    text=(WRF/'run/URBPARM.TBL').read_text()
    for name,value in changes.items():
        text,n=re.subn(r'^'+name+r'\s*:.*$',name+': '+value,text,flags=re.M)
        assert n==1,(name,n)
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--save',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=list(ARMS))
    a=p.parse_args();a.save.mkdir(parents=True,exist_ok=True);manifest={}
    for arm in a.arms:
        for precision in ('fp64','fp32'):
            run=a.build/precision/('run_'+arm);run.mkdir(parents=True,exist_ok=True)
            (run/'URBPARM.TBL').write_text(table(ARMS[arm]))
            with (run/'dump.txt').open('w') as out,(run/'stderr.txt').open('w') as err:
                subprocess.run([str((a.build/precision/'oracle3').resolve()),precision,str(HERE.parent/'v040/urban_bep_bem/regimes.txt')],
                               cwd=run,stdout=out,stderr=err,check=True,timeout=120,env=dict(os.environ,**ARM_ENV.get(arm,{})))
            target=a.save/f'{arm}_{precision}.npz';shapes=parse_dump(run/'dump.txt',target)
            manifest[target.name]={'sha256':sha(target),'table_sha256':sha(run/'URBPARM.TBL'),'changes':ARMS[arm],'env':ARM_ENV.get(arm,{}),
                                   'oracle_sha256':sha(a.build/precision/'oracle3'),'fields':len(shapes)}
    manifest['driver_sha256']=sha(HERE/'pvgr_oracle.F90')
    (a.save/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v.get('fields') for k,v in manifest.items() if isinstance(v,dict)}))


if __name__=='__main__':
    main()
