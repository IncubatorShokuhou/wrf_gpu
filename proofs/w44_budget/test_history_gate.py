"""Numeric-gate controls on original reference arrays; no physics self-compare."""
import copy
from pathlib import Path
import unittest
import numpy as np
from history_gate import evaluate,TKE,EDMF,PBL


class GateControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b=Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget')
        with np.load(b/'P14_pristine_reference/wn3_d03_day.npz') as f:
            cls.ref={n:f['dqke' if n=='Dtke' else n.lower()].copy() for n in TKE+EDMF+PBL}
        with np.load(b/'P25_original_conditioning/wn3_d03_day.npz') as f:cls.floor=f['Dtke_profile_floor'].copy()

    def test_healthy_original(self):
        self.assertTrue(evaluate(self.ref,self.ref,self.floor)['passed'])

    def test_every_field_rejects_nonfinite_on_either_side(self):
        for field in self.ref:
            for side in ('actual','reference'):
                for bad in (np.nan,np.inf,-np.inf):
                    with self.subTest(field=field,side=side,bad=bad):
                        a={k:v.copy() for k,v in self.ref.items()};r={k:v.copy() for k,v in self.ref.items()}
                        (a if side=='actual' else r)[field][0,0]=bad
                        with self.assertRaisesRegex(AssertionError,'nonfinite'):
                            evaluate(a,r,self.floor)

    def test_floor_nonfinite_negative_refuses(self):
        for bad in (np.nan,np.inf,-np.inf,-1.):
            floor=self.floor.copy();floor[0]=bad
            with self.assertRaisesRegex(AssertionError,'invalid original floor'):
                evaluate(self.ref,self.ref,floor)

    def test_missing_each_field_refuses(self):
        for field in self.ref:
            a=dict(self.ref);a.pop(field)
            with self.assertRaisesRegex(AssertionError,'missing'):
                evaluate(a,self.ref,self.floor)

    def zero_pbl(self):return {n:np.zeros_like(self.floor) for n in PBL}

    def ulp_pbl(self):
        # A generous REAL floor (RTHBLTEN 4 ulp(400 K)/6 s) must not hide the critic's survivors.
        return {n:np.full_like(self.floor,4*np.spacing(np.float32(400))/6 if n=='RTHBLTEN' else 1e-9) for n in PBL}

    def test_critic_pbl_survivors_now_fail(self):
        # cr-w44-budget F1: these three passed the old absolute 1e-3 PBL bound.
        for floors in (None,self.zero_pbl(),self.ulp_pbl()):
            for label,name,mutate in (('G1','RQVBLTEN',lambda a:a*np.float32(1.5)),('G2','RQVBLTEN',np.zeros_like),
                                      ('G3','RQCBLTEN',lambda a:-a)):
                with self.subTest(label=label,floor=None if floors is None else float(floors['RTHBLTEN'][0])):
                    self.assertTrue(np.count_nonzero(self.ref[name]))
                    a={k:v.copy() for k,v in self.ref.items()};a[name]=mutate(a[name])
                    r=evaluate(a,self.ref,self.floor,pbl_floor=floors)
                    self.assertFalse(r['passed']);self.assertFalse(r['terms'][name]['passed'])

    def test_pbl_bound_is_relative_plus_floor(self):
        a={k:v.copy() for k,v in self.ref.items()}
        colmax=np.max(np.abs(self.ref['RTHBLTEN'][0]))
        a['RTHBLTEN'][0,0]+=np.float32(.04*colmax)
        self.assertTrue(evaluate(a,self.ref,self.floor,pbl_floor=self.zero_pbl())['terms']['RTHBLTEN']['passed'])
        a['RTHBLTEN'][0,0]+=np.float32(.03*colmax)
        self.assertFalse(evaluate(a,self.ref,self.floor,pbl_floor=self.zero_pbl())['terms']['RTHBLTEN']['passed'])
        floors=self.zero_pbl();floors['RTHBLTEN'][0]=.2*colmax
        self.assertTrue(evaluate(a,self.ref,self.floor,pbl_floor=floors)['terms']['RTHBLTEN']['passed'])

    def test_pbl_floor_invalid_or_incomplete_refuses(self):
        for bad in (np.nan,np.inf,-1.):
            floors=self.zero_pbl();floors['RQVBLTEN'][0]=bad
            with self.assertRaisesRegex(AssertionError,'invalid original PBL floor'):
                evaluate(self.ref,self.ref,self.floor,pbl_floor=floors)
        floors=self.zero_pbl();floors.pop('RQIBLTEN')
        with self.assertRaisesRegex(AssertionError,'missing/unexpected PBL floor'):
            evaluate(self.ref,self.ref,self.floor,pbl_floor=floors)

    def test_above_approved_dtke_bound_still_fails(self):
        a={k:v.copy() for k,v in self.ref.items()}
        bound=max(.05*abs(float(a['Dtke'][0,0])),self.floor[0])
        a['Dtke'][0,0]+=np.float32(2*bound)
        r=evaluate(a,self.ref,self.floor)
        self.assertFalse(r['passed']);self.assertFalse(r['terms']['Dtke']['columns'][0]['passed'])


if __name__=='__main__':unittest.main()
