"""Native source-grid file preparation controls; no forecast parity claims."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.cycling.core import cycle, _times
from gpuwrf.io.netcdf_lock import Dataset
from test_atmosphere_cycle import _loaded_source_path, fixture_case


def native_fixture(root, domain=1):
    files=fixture_case(root,domain=domain)
    for path,old in ((files[0],True),(files[1],False)):
        with Dataset(path,"r+") as ds:
            nt=len(ds.dimensions["Time"])
            ds.createVariable("SAVE_TOPO_FROM_REAL","i4",("Time",))[:nt]=0 if old else 1
            ds.createVariable("XTIME","f4",("Time",))[:nt]=np.arange(nt,dtype=np.float32)*180
            ds["XTIME"].units="minutes since old issue" if old else "minutes since fresh issue"
            ds.createVariable("RDN","f4",("Time","bottom_top"))[:nt]=np.broadcast_to(np.array([0,-1/.35,-1/.35],np.float32),(nt,3))
            for name,value in {"P00":100000,"T00":290,"TLP":50,"TISO":200,"P_STRAT":0,"TLP_STRAT":0}.items():
                ds.createVariable(name,"f4",("Time",))[:nt]=value
            ds.createVariable("T_INIT","f4",("Time","bottom_top","south_north","west_east"))[:nt]=0
            ds.START_DATE="2023-01-15_00:00:00" if old else "2023-01-15_03:00:00"
            ds.SIMULATION_START_DATE=ds.START_DATE
            if old:
                ds["HGT"][:nt]+=5
                ds["PHB"][:nt]+=np.float32(5*9.81)
                ds["PB"][:nt]+=100
                ds["P"][:nt]-=100
                ds["MUB"][:nt]+=200
                ds["MU"][:nt]-=200
    return files


def native_cycle(files,out):
    return cycle(files[0],files[1],out,atmosphere=True,native_atmosphere=True,
                 wrfbdy=files[2],boundary_output=out.with_name(out.name+"_bdy"))


def test_native_keeps_resolved_source_and_resets_clock(tmp_path):
    files=native_fixture(tmp_path)
    report=native_cycle(files,tmp_path/"native")
    with Dataset(files[0]) as src,Dataset(files[1]) as fresh,Dataset(tmp_path/"native") as out:
        assert _times(out)==_times(fresh)
        for name in ("HGT","PB","PHB","MUB","U","V","W","T","THM","P","PH","MU","QVAPOR","QCLOUD","QKE"):
            np.testing.assert_array_equal(out[name][0],src[name][1])
        assert out["SAVE_TOPO_FROM_REAL"][0]==0
        assert out["XTIME"][0]==0
        assert out["XTIME"].units==fresh["XTIME"].units
        assert out.START_DATE==fresh.START_DATE
        assert out.SIMULATION_START_DATE==fresh.SIMULATION_START_DATE
    inventory=report["domains"][0]["native"]
    assert inventory["source_topography_flag"]==0
    assert "HGT" in inventory["copied"]
    assert "XTIME" not in inventory["copied"]
    assert inventory["clock"]["XTIME"]["source"]=="fresh real.exe issue clock"


def test_native_input_only_profile_uses_source_pb(tmp_path):
    files=native_fixture(tmp_path)
    with Dataset(files[0],"r+") as ds:
        ds.renameVariable("T_INIT","REMOVED_T_INIT")
        ds.renameVariable("ALB","REMOVED_ALB")
    report=native_cycle(files,tmp_path/"native")
    derived=report["domains"][0]["native"]["derived"]
    assert {"ALB","T_INIT"}<=set(derived)
    with Dataset(files[1]) as fresh,Dataset(tmp_path/"native") as out:
        assert not np.array_equal(out["ALB"][0],fresh["ALB"][0])
        assert not np.array_equal(out["T_INIT"][0],fresh["T_INIT"][0])


def test_native_missing_history_advection_slot_is_provenanced(tmp_path):
    files=native_fixture(tmp_path)
    with Dataset(files[0],"r+") as ds:
        ds.renameVariable("qke_adv","REMOVED_ADV")
    report=native_cycle(files,tmp_path/"native")
    inventory=report["domains"][0]["native"]
    assert "qke_adv" not in inventory["input_only_from_fresh"]
    assert inventory["derived"]["qke_adv"]["rule"].endswith("QKE")


@pytest.mark.parametrize("kind",["unresolved_terrain","parent","coordinate","time","nonfinite"])
def test_native_refuses_incompatible_data(tmp_path,kind):
    files=native_fixture(tmp_path)
    with Dataset(files[0],"r+") as ds:
        if kind=="unresolved_terrain":ds["SAVE_TOPO_FROM_REAL"][:]=1
        elif kind=="parent":ds.PARENT_ID=2
        elif kind=="coordinate":ds["C1H"][1]+=1
        elif kind=="time":ds["Times"][1]=np.frombuffer(b"2023-01-15_04:00:00",dtype="S1")
        else:ds["QCLOUD"][1]=np.nan
    with pytest.raises(ValueError):
        native_cycle(files,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
    assert not list(tmp_path.glob(".gpuwrf-native-template-*"))


def test_native_cold_self_identity_remains_exact(tmp_path):
    _,template,bdy=native_fixture(tmp_path)
    report=native_cycle((template,template,bdy),tmp_path/"identity")
    assert report["domains"][0]["native"]["identity"]
    with Dataset(template) as src,Dataset(tmp_path/"identity") as out:
        assert set(src.variables)==set(out.variables)
        for name in src.variables:
            assert np.asarray(src[name][:]).tobytes()==np.asarray(out[name][:]).tobytes(),name


def test_native_cli_has_no_jax_import(tmp_path):
    files=native_fixture(tmp_path)
    code="""import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='jax' or fullname.startswith('jax.') or fullname=='jaxlib' or fullname.startswith('jaxlib.'):
   raise AssertionError('offline native cycling imported '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['gpuwrf','cycle']+sys.argv[1:]
from gpuwrf.cli import main
raise SystemExit(main())
"""
    p=subprocess.run([sys.executable,"-c",code,"--source",str(files[0]),"--wrfinput",str(files[1]),
                      "--output",str(tmp_path/"out"),"--atmosphere","--native-atmosphere","--wrfbdy",str(files[2]),
                      "--boundary-output",str(tmp_path/"out_bdy")],capture_output=True,text=True,timeout=60,
                      env={**os.environ,"JAX_PLATFORMS":"cpu","GPUWRF_JAX_CACHE_DIR":str(tmp_path/"math-cache"),
                           "PYTHONPATH":_loaded_source_path()})
    assert p.returncode==0,p.stderr


LOW_BDY={"SST":(290.,291.5),"TMN":(285.,286.25),"VEGFRA":(40.,55.),"ALBBCK":(.17,.19),"SEAICE":(0.,0.)}


def _with_low_boundary(files):
    for path,old in ((files[0],True),(files[1],False)):
        with Dataset(path,"r+") as ds:
            nt=len(ds.dimensions["Time"])
            for name,(warm,fresh) in LOW_BDY.items():
                ds.createVariable(name,"f4",("Time","south_north","west_east"))[:nt]=warm if old else fresh
    return files


def test_native_applies_wrfda_update_low_bdy(tmp_path):
    # WRFDA da_update_bc defaults (update_low_bdy/keep_tsk_wrf/keep_snow_wrf=.true.,
    # update_lsm=.false.): statics from the fresh analysis, water TSK fresh, land
    # TSK/snow/soil cycled, snow removed over water (da_update_bc.f90:365-517).
    files=_with_low_boundary(native_fixture(tmp_path))
    report=native_cycle(files,tmp_path/"native")
    with Dataset(files[0]) as src,Dataset(files[1]) as fresh,Dataset(tmp_path/"native") as out:
        water=np.asarray(fresh["IVGTYP"][0])==fresh.ISWATER
        assert water.sum()==1
        for name in LOW_BDY:
            np.testing.assert_array_equal(out[name][0],fresh[name][0])
        np.testing.assert_array_equal(out["TSK"][0],np.where(water,fresh["TSK"][0],src["TSK"][1]))
        assert np.all(np.asarray(src["SNOW"][1])[water]>0)
        for name in ("SNOW","SNOWC","SNOWH"):
            np.testing.assert_array_equal(out[name][0],np.where(water,0,src[name][1]))
        for name in ("SMOIS","TSLB","SH2O"):
            np.testing.assert_array_equal(out[name][0],src[name][1])
    low=report["domains"][0]["native"]["low_boundary_update"]
    assert low["iswater"]==17 and low["water_cells"]==1 and low["snow_over_water_removed_cells"]==1
    assert set(LOW_BDY)|{"TSK","SNOW","SNOWC","SNOWH","IVGTYP","ISLTYP","LANDMASK","XLAND"}<=set(low["fields"])
    assert not set(low["fields"])&set(report["domains"][0]["native"]["copied"])


def test_native_low_bdy_refuses_field_missing_from_fresh_analysis(tmp_path):
    files=_with_low_boundary(native_fixture(tmp_path))
    with Dataset(files[1],"r+") as ds:
        ds.renameVariable("TMN","REMOVED_TMN")
    with pytest.raises(ValueError,match="TMN"):
        native_cycle(files,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
