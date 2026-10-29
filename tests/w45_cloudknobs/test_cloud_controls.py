"""Optimizer admission/plumbing only; analytic fixtures do not validate WRF physics."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.optimize import cloud_controls as cc, continuous

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cloud_study_fixtures", ROOT/"tests/w41_optcont/test_continuous.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
study_files = fixtures.study_files


@pytest.mark.parametrize("control", [k for k, v in cc.catalog().items() if v["admitted"]])
def test_admitted_matrix_and_real_expansion(control):
    params, used = continuous._parameters([{"name": "cloud", "control": control}], {"domains": {}}, ["d01", "d02"])
    assert not used
    p, = params
    assert p["runtime"]["units"] and p["runtime"]["provenance"]
    for factor in (*p["bounds"], 1.):
        expanded = cc.expand(params, [np.log(factor)])["domains"]
        assert expanded["d01"] == expanded["d02"]
        row = p["runtime"]
        assert expanded["d01"][row["family"]][row["name"]] == float(np.float32(row["default"]*factor))


@pytest.mark.parametrize("control", [
    "diffusion_parameters.c_s", "thompson_parameters.ccn_conc", "thompson_parameters.av_i",
    "thompson_parameters.av_r", "thompson_parameters.autoconversion_rate",
    "thompson_parameters.autoconversion_threshold", "mynn_parameters.cloudpdf", "mynn_parameters.typo",
])
def test_unavailable_matrix(control):
    with pytest.raises(ValueError, match="unavailable"):
        continuous._parameters([{"name": "cloud", "control": control}], {"domains": {}})


@pytest.mark.parametrize("change", [{"classes": [2]}, {"domains": ["d01"]}, {"domains": ["d01", "d01"]},
                                       {"bounds": [0, 2]}, {"bounds": [.1, 3]}, {"bounds": [True, 2]},
                                       {"bounds": [float("nan"), 2]}, {"prior_sd_log": 0}])
def test_invalid_runtime_prior_refused(change):
    item = {"name": "cloud", "control": "thompson_parameters.tno", **change}
    with pytest.raises(ValueError):
        continuous._parameters([item], {"domains": {}}, ["d01", "d02"])


def cloud_study(study_files, *, arm=None):
    path, raw = study_files
    raw = copy.deepcopy(raw)
    raw.pop("basis")
    raw["parameters"] = [{"name": "droplets", "control": "thompson_parameters.nt_c"},
                         {"name": "mixing", "control": "mynn_parameters.alp1"}]
    raw["scorer"]["truth"] = {"droplets": 1.2, "mixing": .8}
    raw["eki"] = {"members": 2, "iterations": 1, "seed": 11}
    if arm is not None:
        raw["cloud_arm"] = arm
    path.write_text(json.dumps(raw))
    return path, raw


def test_public_optimize_plan_cloud_study_no_model(study_files, tmp_path, capsys):
    from gpuwrf.cli import main
    path, _ = cloud_study(study_files, arm={"icloud": 3})
    dest = tmp_path/"cloud-plan.json"
    assert main(["optimize", "plan", "--study", str(path), "--output", str(dest)]) == 0
    study = json.loads(dest.read_text())
    assert study["basis"] is None
    assert study["configuration"]["cloud_arm"]["selections"] == {"icloud": 3}
    assert study["cases"][0]["config"]["physics"]["icloud"] == [3]
    assert len(study["parameters"]) == 2
    assert study["planned_case_hours"] == 63  # 39 EKI/evaluation + 24 sensitivity probe case-hours
    assert "PLANNED" in capsys.readouterr().out


def test_native_pixel_vectors_refused_before_case_or_model_work(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    path, _ = cloud_study(study_files)
    stations = tmp_path/"stations.csv"
    stations.write_text(stations.read_text().replace("s00,", "msg3:17,"))
    def case_opened(*a, **k):
        raise AssertionError("case opened before native-pixel guard")
    monkeypatch.setattr(workflows, "_case", case_opened)
    with pytest.raises(ValueError, match="superobservations.*ensemble-space EKI"):
        continuous.load_continuous(path)


@pytest.mark.parametrize("arm", [{"icloud_bl": 1}, {"bl_mynn_edmf": 1}, {"bl_mynn_cloudpdf": 2}])
def test_mynn_discrete_arms_refuse_inactive_scheme(arm):
    config = {"physics": {"bl_pbl_physics": 1, "mp_physics": 8, "cu_physics": 0,
                          "ra_lw_physics": 4, "ra_sw_physics": 4}}
    with pytest.raises(ValueError, match="active bl_pbl_physics=5"):
        cc.cloud_arm(arm, config, 1)


def test_wrf_forced_icloud_is_not_presented_as_another_arm():
    config = {"physics": {"bl_pbl_physics": 5, "mp_physics": 8, "cu_physics": 11,
                          "ra_lw_physics": 4, "ra_sw_physics": 4}}
    with pytest.raises(ValueError, match="forces icloud=1"):
        cc.cloud_arm({"icloud": 3}, config, 1)


@pytest.mark.parametrize("max_dom", [1, 3])
def test_admitted_pdf_arm_overrides_the_inherited_case_variant(max_dom):
    config = {"physics": {"bl_pbl_physics": 5, "mp_physics": 8, "cu_physics": 0,
                          "ra_lw_physics": 4, "ra_sw_physics": 4, "bl_mynn_cloudpdf": 0}}
    bound, _ = cc.cloud_arm({"bl_mynn_cloudpdf": 2}, config, max_dom)
    assert bound["physics"]["bl_mynn_cloudpdf"] == 2  # Registry count-one selection, not max_domains.
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    require_namelist_honoured(bound, tuple(f"d{i:02d}" for i in range(1, max_dom+1)))


@pytest.mark.parametrize("arm", [{"bl_mynn_cloudpdf": 0}, {"bl_mynn_cloudpdf": 1},
                                 {"icloud": True}, {"icloud": 4}, {"cal_cldfra3": 1}])
def test_discrete_missing_branches_refused(study_files, arm):
    path, _ = cloud_study(study_files, arm=arm)
    with pytest.raises(ValueError, match="cloud_arm|unavailable"):
        continuous.load_continuous(path)


@pytest.mark.parametrize("selection", [0, 1, 2, 3])
def test_discrete_icloud_arm_gets_its_own_key(study_files, selection):
    path, _ = cloud_study(study_files, arm={"icloud": selection})
    study = continuous.load_continuous(path)
    assert study["cases"][0]["config"]["physics"]["icloud"] == [selection]
    assert "cloud-" in study["program"]["configuration"]
    keys = [cc.cloud_arm({"icloud": n}, study["cases"][0]["config"], 1)[1]["key"] for n in range(4)]
    assert len(set(keys)) == 4


@pytest.mark.parametrize("knob", ["icloud_bl", "bl_mynn_edmf"])
def test_existing_mynn_off_branches_are_discrete(study_files, knob):
    path, _ = cloud_study(study_files, arm={knob: 0})
    study = continuous.load_continuous(path)
    assert study["cases"][0]["config"]["physics"][knob] == [0]
    assert study["configuration"]["cloud_arm"]["selections"] == {knob: 0}


def test_entrainment_refused_in_edmf_off_arm(study_files):
    path, raw = cloud_study(study_files, arm={"bl_mynn_edmf": 0})
    raw["parameters"] = [{"name": "entrain", "control": "mynn_parameters.entrainment"}]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="active MYNN EDMF"):
        continuous.load_continuous(path)


def test_inactive_runtime_consumer_refused():
    params, _ = continuous._parameters([{"name": "a", "control": "mynn_parameters.alp1"}], {"domains": {}})
    cfg = {"physics": {"mp_physics": 8, "bl_pbl_physics": 1}}
    with pytest.raises(ValueError, match="bl_pbl_physics=5"):
        cc.validate_activity(params, cfg, ["d01"])


def test_driver_materializes_member_runtime_values(study_files, tmp_path):
    """Run driver through first scheduling boundary; no forward model is run."""
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.data import identity
    path, _ = cloud_study(study_files)
    study = continuous.load_continuous(path)
    out = tmp_path/"run"
    out.mkdir()
    captured = []

    class Scheduled(Exception):
        pass

    def prefetch(pairs):
        captured.extend(pairs)
        raise Scheduled

    with pytest.raises(Scheduled):
        continuous.driver(study)(prefetch=prefetch, files=None, output=out, ledger={})
    truth = captured[0][0]
    assert truth["controls"] is None
    record = truth["runtime_parameters"]
    assert record == identity(record["path"])
    payload = json.loads(Path(record["path"]).read_text())
    assert payload["domains"]["d01"]["thompson_parameters"]["nt_c"] == 120.e6
    assert payload["domains"]["d01"]["mynn_parameters"]["alp1"] == float(np.float32(.23*.8))
    case = Path(study["cases"][0]["input_dir"])
    parameter_file = case/workflows.RUNTIME_PARAMETERS_NAME
    parameter_file.write_text(json.dumps(payload))
    plan = workflows.prepare("batch", tmp_path/"batch", cases=[case], single_batch=True, hours=3,
                             runtime_parameters=[parameter_file])
    assert "--case-runtime-parameters" in plan["launcher_options"]
    member = Path(plan["members"][0]["case"])
    manifest = json.loads((member/"member_manifest.json").read_text())
    assert any(row["name"] == workflows.RUNTIME_PARAMETERS_NAME for row in manifest["inputs"])
    assert json.loads((member/workflows.RUNTIME_PARAMETERS_NAME).read_text()) == payload


@pytest.mark.parametrize("rank", [2, 1, 0])
def test_complete_runtime_study_through_real_member_preparation(study_files, tmp_path, monkeypatch, rank):
    """Analytic forward stub; exercises EKI -> study -> ensemble -> member JSON without WRF."""
    from gpuwrf.ens import workflows
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.namelist_parser import parse_namelist
    path, _ = cloud_study(study_files, arm={"icloud": 3})
    study = continuous.load_continuous(path)
    values = []

    def launch(plan_path, **kwargs):
        plan = json.loads(Path(plan_path).read_text())
        assert "--case-runtime-parameters" in plan["launcher_options"]
        for member in plan["members"]:
            case = Path(member["case"])
            cfg = parse_namelist(case/"namelist.input")
            assert cfg["physics"]["icloud"] == 3
            payload = json.loads((case/workflows.RUNTIME_PARAMETERS_NAME).read_text())["domains"]["d01"]
            a = payload["thompson_parameters"]["nt_c"]/100.e6
            b = payload["mynn_parameters"]["alp1"]/.23
            values.append((a, b))
            day = cfg["time_control"]["start_day"]
            out = Path(plan["out_root"])/member["id"]/"wrfout"
            (out/"proofs").mkdir(parents=True)
            for hour in (1, 2, 3):
                stamp = f"2026-01-{day:02d}_{hour:02d}:00:00"
                with Dataset(out/f"wrfout_d01_{stamp}", "w") as ds:
                    for dim, count in (("Time", 1), ("DateStrLen", 19), ("south_north", 2), ("west_east", 3)):
                        ds.createDimension(dim, count)
                    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(stamp)], dtype="S1")
                    ds.setncatts({"GRID_ID": 1, "SIMULATION_START_DATE": f"2026-01-{day:02d}_00:00:00", "MAP_PROJ": 3})
                    responses = (np.log(a), np.log(b)) if rank == 2 else ((np.log(a)+np.log(b),)*2 if rank else (0., 0.))
                    for name, field, unit in (("XLAT", fixtures.LAT, "degree_north"),
                                              ("XLONG", fixtures.LON, "degree_east"),
                                              ("LANDMASK", np.ones((2, 3)), ""),
                                              ("T2", np.full((2, 3), 280+responses[0]), "K"),
                                              ("U10", np.full((2, 3), 5+responses[1]), "m s-1"),
                                              ("V10", np.zeros((2, 3)), "m s-1")):
                        v = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
                        v[:] = field
                        v.units = unit
            (out/"proofs/nested_pipeline_run.json").write_text(json.dumps({"metadata": {"nested_aot": {
                "domains": {"d01": {"cheap_key": "ANALYTIC_STUB_ONLY", "loaded": True, "source": "aot_blob"}}}}}))
            (out.parent/"receipt.json").write_text(json.dumps({"rc": 0, "wall_s": 1., "aot_cache_warm": True,
                "aot_cold_executables": [], "vmhwm_kb": 1_000_000, "vram_peak_mib": 0}))
        return 0

    monkeypatch.setattr(workflows, "launch", launch)
    result = continuous.run(study, tmp_path/"complete", max_parallel=2, wave_mode=False)
    assert result["scheduled_case_hours"] == (63 if rank else 42)
    assert result["identifiability"]["rank"] == rank
    assert result["identifiability"]["null_branches"] == {
        "mynn_parameters": "prior", "thompson_parameters": "prior"}
    assert result["identifiability"]["runtime_cost"]["successful_forecast_wall_s"] == 10.
    for role in ("prior", "posterior"):
        from gpuwrf.optimize.data import identity
        record = result["runtime_parameters"][role]
        assert record == identity(record["path"])
    if rank < 2:
        estimate = np.log([result["estimate_multipliers"][p["name"]] for p in study["parameters"]])
        np.testing.assert_allclose(np.asarray(result["identifiability"]["frozen_null_basis"]) @ estimate, 0, atol=1.e-14)
    if not rank:
        assert result["iterations"] == [] and result["estimate_multipliers"] == {"droplets": 1., "mixing": 1.}
    assert len(set(values)) >= 4  # truth, ensemble, prior, posterior have their own actual JSON values
    ledger = json.loads((tmp_path/"complete/ledger.json").read_text())
    assert all(c["runtime_parameters"] for c in ledger["candidates"].values())


def test_full_cpu_program_and_actual_namelist_keys_share_values(monkeypatch):
    """Full HLO protobuf includes constants (E224); actual bound namelist key + CPU operand program.

    This is a scalar-plumbing proof, not a whole-root physics/GPU identity claim.
    """
    import hashlib
    import jax
    import jax.numpy as jnp
    from gpuwrf import runtime_parameters as rp
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash, carry_aval_hash
    from gpuwrf.coupling import physics_couplers
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    monkeypatch.setattr(physics_couplers, "_MYNN_EDMF", True)
    for suffix in ("NATIVE_REAL", "FULL_COLUMN", "MIXED_PHASE_WRF", "COLUMN_SED", "SED_FP32", "WRF_CONSTANTS"):
        monkeypatch.setenv("GPUWRF_THOMPSON_"+suffix, "1")
    monkeypatch.setenv("GPUWRF_THOMPSON_IMPLICIT_SED", "0")
    monkeypatch.setenv("GPUWRF_MP_COLUMN_TILING", "0")
    monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", "1")
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., run_physics=True)
    controls = [k for k, v in cc.catalog().items() if v["admitted"]]
    # Binder compatibility census, not an EKI study with nine active parameters.
    params = [cc.parameter({"name": f"p{i}", "control": c}, ["d01"]) for i, c in enumerate(controls)]
    a = rp.bind_domain_parameters(nl, cc.expand(params, np.zeros(len(params)))["domains"], "d01")
    b = rp.bind_domain_parameters(nl, cc.expand(params, np.log(np.full(len(params), 1.5)))["domains"], "d01")
    assert static_config_hash(a) == static_config_hash(b) != static_config_hash(nl)
    assert carry_aval_hash((a,), {}) == carry_aval_hash((b,), {})
    traces = []

    def operands(bound, x):
        traces.append(1)
        # Consume every value as a live operand; a baked value changes the full protobuf.
        return sum(x*v for family in (bound.mynn_parameters, bound.thompson_parameters) for v in family.values())

    fn = jax.jit(operands)
    # Raw protobufs contain fresh module IDs even for the same Lowered object.
    # Full-constant text omits those bookkeeping IDs without eliding literals.
    hlo_a = fn.lower(a, jnp.float32(2)).compiler_ir("hlo").as_hlo_text(print_large_constants=True)
    hlo_b = fn.lower(b, jnp.float32(2)).compiler_ir("hlo").as_hlo_text(print_large_constants=True)
    assert hashlib.sha256(hlo_a.encode()).digest() == hashlib.sha256(hlo_b.encode()).digest()
    assert traces == [1]
    assert float(fn(a, jnp.float32(2))) != float(fn(b, jnp.float32(2)))
    def constants(value, large=True):
        return jax.jit(lambda x: x+jnp.asarray(value)).lower(jnp.zeros(4096)).compiler_ir("hlo").as_hlo_text(print_large_constants=large)
    value = np.random.default_rng(224).uniform(.7, 1.3, 4096).astype(np.float32)
    before = constants(value)
    assert before == constants(value)
    elided = constants(value, False)
    value[2048] = 2
    assert elided == constants(value, False)  # reproduce E224 blindness, not just a shape/structure change
    assert before != constants(value)  # a constant-value mutant must be visible
