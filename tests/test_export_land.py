"""Custom real.exe category compatibility uses the shared doctor authority."""
import json
import re

import netCDF4
import numpy as np
import pytest

from gpuwrf.cli import main
from gpuwrf.doctor.land import preflight, category_issues
from test_doctor import case, inspect_case


def custom(case,count,category=5):
    nl=case/"namelist.input"
    nl.write_text(re.sub(r",\s*(?=\w+\s*=)", ",\n", nl.read_text().replace("num_land_cat=3",f"num_land_cat={count}")))
    with netCDF4.Dataset(case/"wrfinput_d01","a") as ds:
        ds.NUM_LAND_CAT=count
        # Land-use edits must update the dependent categorical and mask fields.
        land=ds.variables["LANDMASK"][0].astype(bool)
        for name in ("LU_INDEX","IVGTYP"):
            ds.variables[name][0]=np.where(land,category,2)
        ds.variables["HGT"][0]=np.arange(12).reshape(3,4)*10.
    # The sample TEST table was 3 entries; a custom dataset needs its own table.
    (case/"LANDUSE.TBL").write_text(f"TEST\n{count},1,'parameters'\nSUMMER\n1,.1,'Forest'\n2,.1,'Water'\n")


@pytest.mark.parametrize("count",[21,24,40,61])
def test_custom_real_inputs_dry_run(case,count,capsys):
    custom(case,count)
    report=inspect_case(case)
    assert report["status"]=="PASS",report
    assert main(["run","--input-dir",str(case),"--output-dir",str(case/"out"),"--dry-run"])==0
    plan=json.loads(capsys.readouterr().out)
    assert len(plan["land_category_check"]["checked"])==1
    assert not plan["land_category_check"]["unchecked"]


@pytest.mark.parametrize("category",range(31,42))
def test_lcz_without_urban_refused_by_cli_and_doctor(case,category,capsys):
    custom(case,61,category)
    with netCDF4.Dataset(case/"wrfinput_d01","a") as ds: ds.MMINLU="MODIFIED_IGBP_MODIS_NOAH_LCZ"
    report=inspect_case(case)
    assert report["status"]=="FAIL"
    assert any(i["code"]=="LCZ_WITHOUT_URBAN" and i["severity"]=="ERROR" for i in report["issues"])
    assert main(["run","--input-dir",str(case),"--output-dir",str(case/"out"),"--dry-run"])==2
    assert "LCZ_WITHOUT_URBAN" in capsys.readouterr().err


@pytest.mark.parametrize("value",[np.nan,np.inf,0,61.5,62])
def test_bad_category_refused(case,value):
    custom(case,61)
    with netCDF4.Dataset(case/"wrfinput_d01","a") as ds: ds.variables["LU_INDEX"][0,1,1]=value
    with pytest.raises(ValueError,match="CATEGORY_RANGE"):
        preflight(case,case/"namelist.input",["d01"])


def test_count_mismatch(case):
    custom(case,21)
    with netCDF4.Dataset(case/"wrfinput_d01","a") as ds: ds.NUM_LAND_CAT=24
    with pytest.raises(ValueError,match="differs"): preflight(case,case/"namelist.input",["d01"])


def test_urban_category_does_not_override_scheme_or_table_validation(case):
    custom(case,61,31)
    with netCDF4.Dataset(case/"wrfinput_d01") as ds:
        assert category_issues(ds,61,1)==[]
        assert category_issues(ds,24,1)[0][0]=="CATEGORY_RANGE"
