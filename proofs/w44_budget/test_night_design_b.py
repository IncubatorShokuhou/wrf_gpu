"""Backend-dark controls for the design-B launcher and host selection hook."""
from dataclasses import dataclass,replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_night_driver',HERE/'night_probe_design_b.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


@dataclass(frozen=True)
class Namelist:
    dt_s:float
    radiation_cadence_steps:int=33
    tke_budget:int=0
    bl_mynn_output:int=0
    bl_mynn_edmf:int=1


@dataclass(frozen=True)
class Bundle:
    namelist:object
    static:object


@dataclass(frozen=True)
class Carry:
    state:object
    mynn_budget:object=None
    def replace(self,**kw):return replace(self,**kw)


class Controls(unittest.TestCase):
    def test_same_loaded_operands_and_public_selectors_all_domains(self):
        for on,edmf in ((False,1),(True,1),(True,0)):
            bundles={};carries={};seeds=[];protected=[]
            for i,dt in enumerate((54,18,6),1):
                name=f'd{i:02d}';bundles[name]=Bundle(Namelist(dt),object())
                carries[name]=Carry(SimpleNamespace(theta=SimpleNamespace(shape=(44,7,12))))
            config=lambda nl:dict(tke_budget=nl.tke_budget,bl_mynn_output=nl.bl_mynn_output,bl_mynn_edmf=nl.bl_mynn_edmf)
            def seed(shape,cfg):
                seeds.append((shape,cfg));return object() if on else None
            def protect(carry,nl):protected.append(carry);return carry
            values=(object(),bundles,object(),object(),object(),carries)
            out,resolved=D.configure_loaded(values,on,edmf,seed_history=seed,config_of=config,protect=protect,validate=lambda nl:None)
            self.assertIs(out[0],values[0]);self.assertIs(out[2],values[2])
            self.assertEqual(len(protected),3);self.assertEqual(len(seeds),3)
            for name in carries:
                self.assertIs(out[-1][name].state,carries[name].state)
                self.assertIs(out[1][name].static,bundles[name].static)
                self.assertEqual(config(out[1][name].namelist),dict(tke_budget=int(on),bl_mynn_output=int(on),bl_mynn_edmf=edmf))
                self.assertEqual(bundles[name].namelist.tke_budget,0)
                self.assertEqual(resolved[name]['dt'],bundles[name].namelist.dt_s)

    def test_components_are_serial_in_same_gpu_process(self):
        source=(HERE/'night_probe_design_b.py').read_text()
        self.assertIn("runpy.run_path(str(HERE/'twin_probe.py')",source)
        self.assertNotIn('subprocess.Popen',source)
        self.assertNotIn("subprocess.run([sys.executable",source)
        self.assertIn("('prime_off',False,True,1),('prime_on',True,True,1)",source)

    def test_e41_only_extends_auxiliary_discovery(self):
        source=(HERE/'budget_cubin_gate.py').read_text()
        self.assertIn('module.STEP_PROGRAM=re.compile',source)
        self.assertNotIn('256',source);self.assertNotIn('1000',source);self.assertNotIn('500',source)
        self.assertIn('return module.main()',source)


class AotReportKeys(unittest.TestCase):
    """Recorded product report (P70c, verbatim) -> no key re-spelling anywhere in the check."""
    # Verbatim P70c record (also proofs/w44_budget/fixtures/p70c_nested_aot_report.json); embedded so the
    # sealed launcher copy of this test is self-contained. sha256 of this JSON text below.
    RECORD='{"domains":{"d01":{"cache_hit_count":1,"cached":true,"cheap_key":"bdda955f33b3ab3e83b690fabe98ad0801c2e44923eed0a5defcf1a8a3bca713","error":null,"keys":{"cheap_key:bdda955f33b3ab3e83b690fabe98ad0801c2e44923eed0a5defcf1a8a3bca713":{"cache_hit_count":1,"load_attempt_count":1,"load_attempt_wall_seconds":14.532542173998081,"load_count":1,"load_wall_seconds":14.532542173998081}},"load_attempt_count":1,"load_attempt_wall_seconds":14.532542173998081,"load_count":1,"load_wall_seconds":14.532542173998081,"loaded":true,"name":"d01","real_load":false,"source":"aot_memory_cache"},"fused/d02":{"aot_addresses":["hlo","cheap_key"],"aot_alias_mode":"hardlink","aot_blob_bytes":469309917,"aot_cheap_meta_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/k_c42638680b5037676b554a82579ae382c5e8a60178446ac7335704afda9679ba.meta","aot_cheap_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/k_c42638680b5037676b554a82579ae382c5e8a60178446ac7335704afda9679ba.xlaexec","aot_error":null,"aot_hlo_meta_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/5cde3ef0545dc5ba80fc40ea4d98017b4b5c06996535133b6025495b660f00b4.meta","aot_hlo_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/5cde3ef0545dc5ba80fc40ea4d98017b4b5c06996535133b6025495b660f00b4.xlaexec","aot_kept_buffers":459,"aot_meta_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/k_c42638680b5037676b554a82579ae382c5e8a60178446ac7335704afda9679ba.meta","aot_path":"<USER_HOME>/wrf_gpu2_lanes/w44-budget/cpu_rehearsal_cache_5438aca4b5/jax/aot/0.4.3-jax0.10.0-jaxlib0.10.0-cpu/fused_d02/k_c42638680b5037676b554a82579ae382c5e8a60178446ac7335704afda9679ba.xlaexec","aot_recovery":null,"aot_recovery_compile_seconds":0.0,"aot_recovery_reason":null,"aot_recovery_seconds":0.0,"aot_trace_metadata":"rich","aot_written":true,"blob_sha256":"9c2d2151d2b38d1c516549baee09c0f3b0b72b045248a615df2f340bc3d508ec","cheap_key":"c42638680b5037676b554a82579ae382c5e8a60178446ac7335704afda9679ba","error":null,"hlo_sha256":"5cde3ef0545dc5ba80fc40ea4d98017b4b5c06996535133b6025495b660f00b4","keys":{},"loaded":false,"name":"fused/d02","source":"fallback:fused-jit-compiled+aot-captured"}},"resolved":{"d01":{"dt":54.0},"d02":{"dt":18.0},"d03":{"dt":6.0}},"resolved_note":"driver resolved-config keys (product domain names) with dt from the crop namelist (E225 check: 54/18/6 s)","source":"P70c gpu.log (v12 rehearsal, 2026-10-25T02:16Z): domain_tree.nested_aot_report()[\\"domains\\"] at the first prime_off segment, verbatim"}'
    RECORD_SHA256='0ba75c30e07d1559054232e9d08b9a83e188a3cc84d7c1f9c0772438e2e587d5'

    @classmethod
    def setUpClass(cls):
        import hashlib,json
        assert hashlib.sha256(cls.RECORD.encode()).hexdigest()==cls.RECORD_SHA256
        cls.rec=json.loads(cls.RECORD)
        cls.report,cls.domains=cls.rec['domains'],cls.rec['resolved']

    def test_recorded_product_report_is_accepted(self):
        self.assertEqual(sorted(self.report),['d01','fused/d02'])
        self.assertTrue(D.aot_domains_ok(self.report,self.domains))

    def test_split_children_are_accepted(self):
        split={n:dict(source='aot_blob') for n in self.domains}
        self.assertTrue(D.aot_domains_ok(split,self.domains))

    def test_missing_root_nested_or_source_is_refused(self):
        nested_only={k:v for k,v in self.report.items() if k not in self.domains}
        root_only={k:v for k,v in self.report.items() if k in self.domains}
        no_source={k:dict(v,source=None) if k not in self.domains else v for k,v in self.report.items()}
        for bad in (nested_only,root_only,no_source,{}):
            self.assertFalse(D.aot_domains_ok(bad,self.domains),sorted(bad))

    def test_check_contains_no_spelled_program_name(self):
        import inspect
        source=inspect.getsource(D.aot_domains_ok)
        for spelled in ('fused/','fused_',"'d01'","'d02'","'d03'"):
            self.assertNotIn(spelled,source)


if __name__=='__main__':unittest.main()
