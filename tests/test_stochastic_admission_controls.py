"""Original check_a_mundo bounds and Registry INTEGER parsing, before JAX."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from gpuwrf.io.namelist_check import UnsupportedSchemeError,validate_operational_namelist
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.stochastic_config import bind_stochastic,_INTEGER_KEYS

GAUSSIAN=("sppt","rand_perturb","spp","spp_conv","spp_pbl","spp_lsm")
BOUNDS=(("kminforct",2),("lminforct",2),("kmaxforct",4),("lmaxforct",4))


def parsed(body,tmp_path):
    path=tmp_path/"namelist.input"
    path.write_text("&domains\n max_dom=2,\n/\n&stoch\n"+body+"\n/\n")
    return parse_namelist(path)


@pytest.mark.parametrize("switch",GAUSSIAN)
@pytest.mark.parametrize("selection",["1,0","0,1"])
@pytest.mark.parametrize("key,value",BOUNDS)
def test_gaussian_custom_bounds_fatal_like_original_wrf(switch,selection,key,value,tmp_path):
    nml=parsed(f" skebs=1,\n {switch}={selection},\n {key}={value},",tmp_path)
    with pytest.raises(UnsupportedSchemeError,match=key+".*SKEBS-only"):
        validate_operational_namelist(nml)
    for domain in ("d01","d02"):
        with pytest.raises(ValueError,match=key+".*SKEBS-only"):
            bind_stochastic(nml,domain)


@pytest.mark.parametrize("switch",GAUSSIAN)
def test_gaussian_explicit_default_bounds_remain_admitted(switch,tmp_path):
    nml=parsed(f" {switch}=0,1,\n kminforct=1,lminforct=1,kmaxforct=1000000,lmaxforct=1000000,",tmp_path)
    validate_operational_namelist(nml)


def test_custom_skebs_only_bounds_are_preserved(tmp_path):
    nml=parsed(" skebs=1,\n kminforct=2,lminforct=2,kmaxforct=4,lmaxforct=4,",tmp_path)
    validate_operational_namelist(nml)
    cfg=bind_stochastic(nml,"d01")
    for p in cfg.patterns[:2]:
        assert (p.kmin,p.lmin,p.kmax,p.lmax)==(2,2,4,4)


def test_integer_controls_match_original_registry_types():
    source=Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/registry.stoch").read_text()
    keys=set(re.findall(r"^rconfig\s+integer\s+(\w+)\s+namelist,stoch\s",source,re.M|re.I))
    assert _INTEGER_KEYS=={k.lower() for k in keys}


@pytest.mark.parametrize("key",sorted(_INTEGER_KEYS))
@pytest.mark.parametrize("value",["0.5","1.0","'1'",".true.","2147483648","-2147483649"])
def test_parsed_integer_controls_reject_before_casting(key,value,tmp_path):
    nml=parsed(f" skebs=1,\n {key}={value},",tmp_path)
    with pytest.raises(UnsupportedSchemeError,match=key+".*INTEGER"):
        validate_operational_namelist(nml)
    with pytest.raises(ValueError,match=key+".*INTEGER"):
        bind_stochastic(nml,"d01")


@pytest.mark.parametrize("key",["skebs_vertstruc","sppt_vertstruc","rand_pert_vertstruc",
    "vertstruc_spp_conv","vertstruc_spp_pbl","vertstruc_spp_lsm","stoch_vertstruc_opt"])
def test_vertical_modes_reject_unsupported_integers(key,tmp_path):
    nml=parsed(f" skebs=1,sppt=1,rand_perturb=1,spp=1,\n {key}=2,",tmp_path)
    with pytest.raises(UnsupportedSchemeError,match="0/1|0or1"):
        validate_operational_namelist(nml)


def test_critic_parser_witnesses_refused_without_loading_jax(tmp_path):
    cases=["sppt=1,kminforct=2,","rand_perturb=1,kmaxforct=4,",
        "spp_pbl=0,1,lminforct=2,","skebs=1,skebs_vertstruc=0.5,",
        "sppt=1,sppt_vertstruc=1.5,","skebs=1,kminforct=1.5,",
        "sppt=1,sppt_vertstruc='1',","skebs=1,kmaxforct=2147483648,"]
    code='''import sys,importlib.abc,json
from pathlib import Path
sys.argv[0:1]=["gpuwrf","manifest"]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name=="jax" or name.startswith("jax."):raise AssertionError("validation imported JAX")
sys.meta_path.insert(0,Block())
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.namelist_check import validate_operational_namelist,UnsupportedSchemeError
for body in json.loads(sys.argv[-1]):
    path=Path(sys.argv[-2])
    path.write_text("&domains\\n max_dom=2,\\n/\\n&stoch\\n"+body+"\\n/\\n")
    nml=parse_namelist(path)
    try:validate_operational_namelist(nml)
    except UnsupportedSchemeError:pass
    else:raise AssertionError(body)
assert "jax" not in sys.modules
'''
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,"-c",code,str(tmp_path/"backend-dark.input"),json.dumps(cases)],cwd=repo,check=True,timeout=30,
        env={**os.environ,"JAX_PLATFORMS":"cpu","GPUWRF_JAX_CACHE":"0","PYTHONPATH":str(repo/"src")})
