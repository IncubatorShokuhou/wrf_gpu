"""Private real-Swiss CPU-WRF OBS_DOMAIN case; never links writable outputs.

The executable is the same clean DM build used by RD11's original CPU reference.
Only inputs and physics tables are symlinked; namelists, IO selection, OBS_DOMAIN,
wrfout and rsl files belong to the new lane-local run directory.
"""
from datetime import datetime,timedelta
import hashlib,json,re,shutil,sys
from pathlib import Path
import netCDF4
import numpy as np

CASE=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')


def edit(text,key,value,group):
    pattern=rf'^\s*{key}\s*=.*$'
    line=f' {key} = {value},'
    if re.search(pattern,text,re.M|re.I):return re.sub(pattern,line,text,flags=re.M|re.I)
    if re.search(rf'&{group}\b',text,re.I):return re.sub(rf'(&{group}\s*\n)',lambda m:m[1]+line+'\n',text,count=1,flags=re.I)
    return text+f'\n&{group}\n{line}\n/\n'


def report(date,lat,lon,station,sounding,rows,elev):
    text=f' {date:%Y%m%d%H%M%S}\n  {lat:9.4f} {lon:9.4f} \n'
    text+=f'  {station:40s}   {station:40s}   \n'
    platform='FM-35 TEMP' if sounding else 'FM-12 SYNOP'
    text+=f'  {platform:16s}  {"OSSE":16s}  {elev:8.0f}  {"T" if sounding else "F":>4}  {"F":>4}  {len(rows):5d}\n'
    for row in rows:text+=' '+''.join(f'{v:11.3f} {q:11.3f} ' for v,q in row)+'\n'
    return text


def main(out,enabled):
    out.mkdir(parents=True,exist_ok=False)
    for p in CASE.iterdir():
        if p.name.startswith(('wrfout','rsl.','namelist','README','OBS_DOMAIN')) or p.name=='wrf.exe':continue
        if p.is_file():(out/p.name).symlink_to(p)
    exe=CASE/'wrf.exe'
    # Copy the resolved executable, so any later reference deployment cannot
    # silently change the measured run. Its clean binary has no WRFGPU2 dumps.
    shutil.copyfile(exe,out/'wrf.exe');(out/'wrf.exe').chmod(0o755)
    with netCDF4.Dataset(CASE/'wrfinput_d01') as ds:
        lat=np.asarray(ds['XLAT'][0]);lon=np.asarray(ds['XLONG'][0]);hgt=np.asarray(ds['HGT'][0])
        psfc=np.asarray(ds['PSFC'][0]);t2=np.asarray(ds['T2'][0])
        times=ds['Times'][0].tobytes().decode()
        start=datetime.strptime(times,'%Y-%m-%d_%H:%M:%S')
    end=start+timedelta(seconds=54)
    text=(CASE/'namelist.input').read_text()
    values={'run_days':0,'run_hours':0,'run_minutes':0,'run_seconds':54,
      'end_year':end.year,'end_month':end.month,'end_day':end.day,'end_hour':end.hour,'end_minute':end.minute,'end_second':end.second,
      'history_interval':0,'history_interval_s':18,'frames_per_outfile':1,'restart':'.false.',
      'iofields_filename':"'obs_iofields.txt'",'ignore_iofields_warning':'.false.'}
    for k,v in values.items():text=edit(text,k,str(v),'time_control')
    for k,v in {'max_dom':1,'time_step':18,'nproc_x':1,'nproc_y':1}.items():text=edit(text,k,str(v),'domains')
    opts={'grid_fdda':0,'obs_nudge_opt':int(enabled),'max_obs':64,'fdda_end':10.,
      'obs_nudge_wind':1,'obs_nudge_temp':1,'obs_nudge_mois':1,'obs_nudge_pstr':1,
      'obs_coef_wind':3e-4,'obs_coef_temp':3e-4,'obs_coef_mois':3e-4,'obs_coef_pstr':3e-4,
      'obs_rinxy':9.,'obs_rinsig':.1,'obs_twindo':.05,'obs_ionf':2,
      'obs_sfcfact':1.,'obs_sfcfacr':1.,'obs_sfc_scheme_horiz':0,'obs_sfc_scheme_vert':0,
      'obs_scl_neg_qv_innov':1,'obs_ipf_in4dob':'.false.','obs_ipf_errob':'.false.','obs_ipf_nudob':'.false.'}
    for k,v in opts.items():text=edit(text,k,str(v),'fdda')
    (out/'namelist.input').write_text(text)
    (out/'obs_iofields.txt').write_text('+:h:0:OBS_SAVWT\n' if enabled else '')
    missing=(-888888.,-888888.)
    obs=''
    for ix,iy,label in ((20,20,'INSIDE'),(6,6,'DISTANT')):
        row=[missing,missing,(float(hgt[iy,ix]),0),(float(t2[iy,ix])+1,0),(5.,0),(-2.,0),(75.,0),(float(psfc[iy,ix]),0),missing]
        obs+=report(start,lat[iy,ix],lon[iy,ix],label,False,[row],float(hgt[iy,ix]))
    rows=[[(85000.,0),(1500.,0),(278.,0),(8.,0),(-3.,0),(70.,0)],
          [(75000.,0),(2500.,0),(273.,0),(10.,0),(-1.,0),(60.,0)],
          [(65000.,0),(3500.,0),(268.,0),(12.,0),(1.,0),(50.,0)]]
    obs+=report(start,lat[20,20],lon[20,20],'SOUNDING',True,rows,float(hgt[20,20]))
    obs+=report(start+timedelta(hours=1),lat[20,20],lon[20,20],'OUTSIDE_TIME',False,[row],float(hgt[20,20]))
    (out/'OBS_DOMAIN101').write_text(obs)
    meta={'reference_case':str(CASE),'executable_source':str(exe.resolve()),
          'executable_sha256':hashlib.sha256((out/'wrf.exe').read_bytes()).hexdigest(),
          'options':opts,'dt_s':18,'steps':3,'start':start.isoformat(),'end':end.isoformat(),
          'inputs':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('wrfinput_d01','wrfbdy_d01','OBS_DOMAIN101','namelist.input')}}
    (out/'PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),bool(int(sys.argv[2])))
