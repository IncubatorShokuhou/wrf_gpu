"""Small real Swiss CPU coupled gate, with disclosed crop-map changes for coverage.

Native loader and operational step, no physics substitutions. CPU Thomas
installation matches the established O1 probe. All files/caches are lane-owned.
Usage: python coupled_smoke.py TREE OUTPUT_JSON OPT_IRR OPT_IRRM [OPT_ALB OPT_SNF SST_SKIN]
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

TREE,OUTPUT=map(Path,sys.argv[1:3])
irr,irrm=map(int,sys.argv[3:5])
alb,snf,skin=map(int,sys.argv[5:8]) if len(sys.argv)>5 else (1,4,1)
btr,rsf=map(int,sys.argv[8:10]) if len(sys.argv)>8 else (1,1)
dveg=int(sys.argv[10]) if len(sys.argv)>10 else 4
land_options=dict(opt_alb=alb,opt_snf=snf,opt_irr=irr,opt_irrm=irrm,opt_btr=btr,opt_rsf=rsf,dveg=dveg)
land_options.update(json.loads(os.environ.get("U2_LAND_EXTRA_OPTIONS","{}")))
assert all(key=="dveg" or key.startswith("opt_") for key in land_options)
assert all(isinstance(value,int) for value in land_options.values())
sys.path[:0]=[str(TREE/"src"),str(TREE/"proofs/o1_nlbind")]
import jax
import jax.numpy as jnp
import numpy as np
import cpu_safe
cpu_safe.install()
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.namelist_binding import require_namelist_honoured
from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform=="cpu"
assert Path(om.__file__).is_relative_to(TREE)
assert not Path("/tmp/wrf_gpu2_quiet").exists()
state_contract._gpu_device=lambda:jax.devices("cpu")[0]
SOURCE=Path(os.environ.get("U2_LAND_CASE","<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu"))
LANE=Path("<USER_HOME>/wrf_gpu2_lanes/u2-land")
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
record=dict(source_tree=str(TREE),source_case=str(SOURCE),platform="cpu",options=dict(sst_skin=skin,**land_options),
            cache="disabled",steps=3,created_utc=datetime.now(timezone.utc).isoformat(),
            scope="CPU integration/finite gate; real Swiss geometry/forcing with controlled crop map, NOT a forecast-fidelity comparison")
t0=time.monotonic()
try:
    with tempfile.TemporaryDirectory(dir=LANE,prefix="coupled_case_") as temp:
        case=Path(temp)/"case"
        case.mkdir()
        for path in SOURCE.iterdir():
            if path.name not in ("namelist.input","wrfinput_d01"):
                (case/path.name).symlink_to(path)
        shutil.copyfile(SOURCE/"wrfinput_d01",case/"wrfinput_d01")
        text=(SOURCE/"namelist.input").read_text()
        for key,value in (("sst_skin",skin),("sf_lake_physics",0)):
            pattern=rf"^\s*{key}\s*=.*$"
            line=f" {key}={value},"
            if re.search(pattern,text,re.M|re.I):text=re.sub(pattern,line,text,flags=re.M|re.I)
            else:text=re.sub(r"(&physics\s*\n)",lambda m:m[1]+line+"\n",text,count=1,flags=re.I)
        options=" ".join(f"{key}={value}," for key,value in land_options.items())
        if re.search(r"&noah_mp\b",text,re.I):
            for key in land_options:
                text=re.sub(rf"^\s*{key}\s*=.*$","",text,flags=re.M|re.I)
            text=re.sub(r"(&noah_mp\s*\n)",lambda m:m[1]+options+"\n",text,count=1,flags=re.I)
        else:text+="\n&noah_mp\n"+options+"\n/\n"
        (case/"namelist.input").write_text(text)
        with Dataset(case/"wrfinput_d01","a") as ds:
            land=np.asarray(ds["LANDMASK"][0])>.5
            if "SNOWH" in ds.variables:land&=np.asarray(ds["SNOWH"][0])<.01
            indices=np.argwhere(land)
            assert len(indices)>=4
            selected=indices[len(indices)//2:len(indices)//2+4]
            record["coverage_crop_cells"]=selected.tolist()
            shape=land.shape
            if land_options.get('opt_soil',1)==2:
                for k in range(1,5):
                    name='SOILCL'+str(k)
                    if name not in ds.variables:ds.createVariable(name,'f4',('Time','south_north','west_east'))
                    ds[name][:]=np.asarray(ds['ISLTYP'][0])[None,...]
                record['soil_map_controls']='Replicated real surface category; four disclosed crop cells use layer categories1/2/3/4'
            if land_options.get('opt_soil',1)==3:
                if 'soil_comp' not in ds.dimensions:ds.createDimension('soil_comp',8)
                if 'SOILCOMP' not in ds.variables:ds.createVariable('SOILCOMP','f4',('Time','soil_comp','south_north','west_east'))[:]=0.
                record['soil_map_controls']='Original composition retained, or allocator-zero fallback; four disclosed crop cells use sand/clay profiles'
            if land_options.get('opt_tdrn',0):
                if 'TD_FRACTION' not in ds.variables:ds.createVariable('TD_FRACTION','f4',('Time','south_north','west_east'))[:]=0.
                record['tile_controls']='Last two disclosed cells have original sand category, SMOIS/SH2O.33, soil/skin290K and TD_FRACTION1 for nonzero drainage'
            for key in ("IRFRACT","SIFRACT","MIFRACT","FIFRACT","PLANTING","HARVEST"):
                if key not in ds.variables:
                    ds.createVariable(key,"f4",("Time","south_north","west_east"))[:]=0.
            if land_options.get('opt_crop',0)==1:
                if 'crop_cat' not in ds.dimensions:ds.createDimension('crop_cat',5)
                if 'CROPTYPE' not in ds.variables:ds.createVariable('CROPTYPE','f4',('Time','crop_cat','south_north','west_east'))[:]=0.
                ds['CROPTYPE'][:]=0.
                record['crop_map_controls']='Existing crop fractions zeroed; only four disclosed cells select Liu crop with valid calendar and season-GDD maps'
                if 'SEASON_GDD' not in ds.variables:ds.createVariable('SEASON_GDD','f4',('Time','south_north','west_east'))[:]=0.
            for iy,ix in selected:
                for key in ("IVGTYP","LU_INDEX"):
                    if key in ds.variables:ds[key][0,iy,ix]=14
                for key in ("VEGFRA","SHDMAX","SHDMIN"):
                    if key in ds.variables:ds[key][0,iy,ix]=70.
                ds["SMOIS"][0,:,iy,ix]=.12
                ds["SH2O"][0,:,iy,ix]=.12
                for key,value in (("IRFRACT",.8),("SIFRACT",.3),("MIFRACT",.4),("FIFRACT",.3),("PLANTING",1.),("HARVEST",365.)):
                    ds[key][0,iy,ix]=value
                if land_options.get('opt_crop',0)==1:
                    ds['CROPTYPE'][0,:,iy,ix]=[.7,.2,.05,.05,1.]
                    ds['SEASON_GDD'][0,iy,ix]=1770.
                if land_options.get('opt_soil',1)==2:
                    for k in range(1,5):ds['SOILCL'+str(k)][0,iy,ix]=k
                if land_options.get('opt_soil',1)==3:ds['SOILCOMP'][0,:,iy,ix]=[60.,50.,40.,30.,10.,20.,30.,40.]
            if land_options.get('opt_tdrn',0):
                for iy,ix in selected[-2:]:
                    ds['ISLTYP'][0,iy,ix]=1
                    ds['SMOIS'][0,:,iy,ix]=.33;ds['SH2O'][0,:,iy,ix]=.33
                    ds['TSLB'][0,:,iy,ix]=290.;ds['TSK'][0,iy,ix]=290.
                    ds['TD_FRACTION'][0,iy,ix]=1.
                    if land_options.get('opt_soil',1)==2:
                        for k in range(1,5):ds['SOILCL'+str(k)][0,iy,ix]=1
        nml=parse_namelist(case/"namelist.input")
        require_namelist_honoured(nml,("d01",))
        cfg=pipeline.NestedPipelineConfig(case,Path(temp)/"out",Path(temp)/"proof",hours=1,max_dom=1)
        _,bundles,_,_,_,carries=pipeline._load_domains(cfg,("d01",))
        nl,carry=bundles["d01"].namelist,carries["d01"]
        record["load_s"]=time.monotonic()-t0
        record["bound_options"]=nl.noahmp_options.__dict__
        record['flag_manifest']={key:value for key,value in os.environ.items() if key.startswith('GPUWRF_')}
        record['option_resident_bytes']={name:sum(int(value.size*value.dtype.itemsize) for value in jax.tree.leaves(getattr(carry,name)) if hasattr(value,'dtype'))
            for name in ('sst_skin','noahmp_irrigation','noahmp_carbon','noahmp_groundwater','noahmp_tile_drain')}
        record['xla_flags']=os.environ.get('XLA_FLAGS','')
        if land_options.get('opt_crop',0)==1:
            record['active_crop_columns']=int(np.count_nonzero(np.asarray(nl.noahmp_carbon_static.parameters.croptype)))
            assert record['active_crop_columns']>=4,'Liu crop smoke is vacuous'
        cb=om.build_clock_base(nl)
        start=time.monotonic()
        restored=None
        for index in range(3):
            carry=om._advance_chunk_fori(carry,nl,index+1,cb,n_steps=1,cadence=int(nl.radiation_cadence_steps))
            jax.block_until_ready(carry)
            if index==1:
                from gpuwrf.io.restart import write_restart,read_restart
                restart=Path(temp)/'two_step.pkl'
                write_restart(carry,nl,nl.grid,index+1,restart,noahmp_static=nl.noahmp_static)
                restored,*_=read_restart(restart,namelist=nl)
        record["step_wall_s"]=time.monotonic()-start
        resumed=om._advance_chunk_fori(restored,nl,3,cb,n_steps=1,cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(resumed)
        left,lt=jax.tree.flatten(carry);right,rt=jax.tree.flatten(resumed)
        assert lt==rt,'Restart changed carry tree'
        mismatches=[]
        for i,(a,b) in enumerate(zip(left,right,strict=True)):
            a,b=np.asarray(a),np.asarray(b)
            if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():mismatches.append(i)
        record['restart_continuity']=dict(leaves=len(left),different=mismatches)
        assert not mismatches,'Restart continuation differs'
        leaves=[np.asarray(value) for value in jax.tree.leaves(carry)]
        record["nonfinite"]=sum(int((~np.isfinite(value)).sum()) for value in leaves if value.dtype.kind=="f")
        record["guards"]=None if carry.census is None else int(np.asarray(carry.census.guards).sum())
        record["skin_present"]=carry.sst_skin is not None
        record["irrigation_present"]=carry.noahmp_irrigation is not None
        if carry.noahmp_irrigation is not None:
            record["irrigation_sums"]={key:float(np.asarray(value).sum()) for key,value in carry.noahmp_irrigation._asdict().items()}
            assert sum(record["irrigation_sums"][key] for key in ("irsivol","irmivol","irfivol"))>0.,"irrigation smoke is vacuous"
        record["carbon_present"]=carry.noahmp_carbon is not None
        if carry.noahmp_carbon is not None:
            record["carbon_sums"]={key:float(np.asarray(value).sum()) for key,value in carry.noahmp_carbon._asdict().items()}
            assert any(record["carbon_sums"].values()),"dynamic carbon smoke is vacuous"
        record['groundwater_present']=carry.noahmp_groundwater is not None
        if carry.noahmp_groundwater is not None:
            record['groundwater_sums']={key:float(np.asarray(value).sum()) for key,value in carry.noahmp_groundwater._asdict().items()}
        record['tile_drain_present']=carry.noahmp_tile_drain is not None
        if carry.noahmp_tile_drain is not None:
            record['qtdrain_sum']=float(np.asarray(carry.noahmp_tile_drain).sum())
            assert record['qtdrain_sum']>0.,'Tile drainage smoke is vacuous'
        assert record["nonfinite"]==0 and record["guards"]==0,record
        record["status"]="PASS"
except Exception as error:
    import traceback
    record.update(status="FAIL",error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc()[-3000:])
finally:
    record["wall_s"]=time.monotonic()-t0
    OUTPUT.write_text(json.dumps(record,indent=2,default=str)+"\n")
print(json.dumps(record,indent=2,default=str))
if record["status"]!="PASS":sys.exit(1)
