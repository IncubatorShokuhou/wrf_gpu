"""Original Registry controls, backend-dark validation, and both CLI binding seams."""
from dataclasses import field,make_dataclass
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from gpuwrf.io.stochastic_config import bind_stochastic,stochastic_refusals
from gpuwrf.io.namelist_binding import bind_namelist_options,bound_options,namelist_honour_report


def test_all_pattern_control_defaults_match_original_registry():
    source=Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/registry.stoch").read_text()
    values={}
    for kind,key,raw in re.findall(r"^rconfig\s+(integer|logical|real)\s+(\w+)\s+namelist,stoch\s+\S+\s+(\S+)",source,re.M):
        values[key.lower()] = (raw.lower()==".true." if kind=="logical" else int(raw) if kind=="integer" else float(raw))
    values.update(skebs=1,sppt=1,rand_perturb=1,spp=1)
    cfg=bind_stochastic({"stoch":values},"d01")
    for index,group in enumerate(("sppt","rand_pert","spp_conv","spp_pbl","spp_lsm"),start=2):
        p=cfg.patterns[index]
        assert (p.stddev,p.cutoff,p.lengthscale,p.timescale)==tuple(values[k+group] for k in
            ("gridpt_stddev_","stddev_cutoff_","lengthscale_","timescale_"))
    for index,suffix in enumerate(("psi","t")):
        p=cfg.patterns[index]
        assert (p.backscatter,p.decorrelation,p.exponent)==tuple(values[k+suffix] for k in
            ("tot_backscat_","ztau_","rexponent_"))
    assert cfg.seeds==(811,53,17,171,217,317) and cfg.nens==1


@pytest.mark.parametrize("key,index",[("skebs",0),("sppt",2),("rand_perturb",3),
    ("spp_conv",4),("spp_pbl",5),("spp_lsm",6),("stoch_force_opt",0)])
@pytest.mark.parametrize("domain",["d01","d02"])
def test_independent_stochastic_option_reaches_central_pipeline_binding(key,index,domain):
    nml={"stoch":{key:1},"domains":{"max_dom":2}}
    defaults=bound_options(nml,domain)
    defaults.setdefault("cu_physics", 0)  # OperationalNamelist default; this fixture selects stochastic controls only.
    Tiny=make_dataclass("BoundNamelist",[(k,type(v),field(default=v)) for k,v in defaults.items()]+
                       [("stochastic_config",object,field(default=None))])
    bound=bind_namelist_options(Tiny(),nml,domain)
    assert bound.stochastic_config is not None
    assert bound.stochastic_config.patterns[index] is not None
    assert bound.stochastic_config.root == (domain=="d01")
    if key in ("spp_conv","spp_pbl","spp_lsm"):
        assert bound.stochastic_config.produce[index] == (domain=="d01")
    assert not stochastic_refusals(nml,(domain,))
    for filename in ("daily_pipeline.py","nested_pipeline.py"):
        text=(Path(__file__).resolve().parents[1]/"src/gpuwrf/integration"/filename).read_text()
        assert "bind_namelist_options(" in text


def test_spp_master_expands_suboptions_globally_like_check_a_mundo():
    for domain in ("d01","d02"):
        cfg=bind_stochastic({"stoch":{"spp":[0,1]}},domain)
        assert all(cfg.patterns[index] is not None for index in (4,5,6))


def test_independent_spp_child_selection_allocates_all_domains_but_only_selected_consumer():
    nml={"stoch":{"spp_pbl":[0,1]}}
    root,child=(bind_stochastic(nml,d) for d in ("d01","d02"))
    assert all(root.patterns[i] is not None and child.patterns[i] is not None for i in (4,5,6))
    assert not any(root.produce) and not root.apply_pbl
    assert child.produce[5] and child.apply_pbl


@pytest.mark.parametrize("controls",[{"sppt":2},{"spp_pbl":1,"iseed_spp_pbl":2**32},
    {"skebs":1,"skebs_vertstruc":2},{"sppt":1,"timescale_sppt":0},
    {"sppt":1,"gridpt_stddev_sppt":float("nan")},{"spp":1,"hrrr_cycling":False},
    {"multi_perturb":1},{"sppt":1,"pert_mynn":True},{"skebs":1,"misspelled_seed":7}])
def test_unsupported_controls_refused_before_execution(controls):
    errors=stochastic_refusals({"stoch":controls})
    assert errors
    report,_=namelist_honour_report({"stoch":controls},("d01",))
    assert report


def test_binding_validation_does_not_import_jax():
    code='''import sys,importlib.abc
sys.argv=["gpuwrf","manifest"]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name=="jax" or name.startswith("jax."):
            raise AssertionError("host stochastic validation imported JAX")
sys.meta_path.insert(0,Block())
from gpuwrf.io.stochastic_config import bind_stochastic,stochastic_refusals
from gpuwrf.io.namelist_binding import namelist_honour_report
assert bind_stochastic({"stoch":{"skebs":1,"spp":1}},"d01").patterns[5] is not None
assert stochastic_refusals({"stoch":{"sppt":2}})
assert namelist_honour_report({"stoch":{"multi_perturb":1}},("d01",))[0]
assert "jax" not in sys.modules
'''
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,"-c",code],check=True,timeout=30,cwd=repo,
        env={**os.environ,"JAX_PLATFORMS":"cpu","GPUWRF_JAX_CACHE":"0",
             "PYTHONPATH":str(repo/"src")})
