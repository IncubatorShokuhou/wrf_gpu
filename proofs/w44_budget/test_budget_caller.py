"""CPU-only isolation, selection, serialization and shared-operand controls."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import mynn_budget_adapter as A
from gpuwrf.diagnostics import mynn_budget as H


class CallerControls(unittest.TestCase):
    def test_default_calls_only_original_adapter(self):
        state=object();result=object()
        with patch.object(A.production,'mynn_adapter_with_source_leaves',return_value=result) as original, \
             patch.object(A,'_build',side_effect=AssertionError('default built a twin')):
            actual,history=A.mynn_sources_with_history(state,6.,budget_config=H.configuration({}),restart=True)
        self.assertIs(actual,result);self.assertIsNone(history)
        original.assert_called_once_with(state,6.,None,restart=True)

    def test_production_and_twin_receive_identical_operand_objects(self):
        state,surface,coefficients=object(),object(),dict(alp1=.24)
        physical,pblh,histories=object(),object(),object()
        seen=[]
        def original(*args,**kwargs):seen.append((args,kwargs));return physical,pblh
        def twin(*args,**kwargs):seen.append((args,kwargs));return histories
        got=A._column_pair(original,twin,state,6.,surface=surface,mynn_coefficients=coefficients,sqfac=2.)
        self.assertTrue(all(a is b for a,b in zip(got,(physical,pblh,histories),strict=True)))
        self.assertIs(seen[0][0][0],state);self.assertIs(seen[1][0][0],state)
        for key in ('surface','mynn_coefficients','sqfac'):
            self.assertIs(seen[0][1][key],seen[1][1][key])

    def test_conflicting_legacy_edmf_cannot_change_physics_when_budgets_enable(self):
        with patch.object(A.production,'_MYNN_EDMF',False), \
             patch.object(A,'_build',side_effect=AssertionError('built a changed physical program')):
            with self.assertRaisesRegex(ValueError,r'physics_couplers\._MYNN_EDMF=False'):
                A.mynn_sources_with_history(object(),6.,budget_config=H.configuration(dict(tke_budget=1)))

    def test_selected_groups_and_column_layout(self):
        values=tuple(jnp.full((6,4),i,jnp.float32) for i in range(21))
        for tke,output in ((0,0),(1,0),(0,1),(1,1)):
            config=H.configuration(dict(tke_budget=tke,bl_mynn_output=output))
            history=H.from_columns(values,2,3,config)
            fields=H.history_fields(history)
            self.assertEqual(len(fields),5*tke+16*output)
            for name,value in fields.items():
                self.assertEqual(value.shape,(4,2,3));self.assertEqual(value.dtype,jnp.float32)
            H.validate_restart(history if tke or output else None,config,1,(4,2,3))
        variable=jnp.arange(24,dtype=jnp.float32).reshape(6,4)
        actual=H.from_columns((variable,)*21,2,3,H.configuration(dict(tke_budget=1)))
        np.testing.assert_array_equal(actual.tke[0],np.moveaxis(np.asarray(variable).reshape(2,3,4),-1,0))

    def test_restart_refuses_missing_changed_or_invalid_history(self):
        config=H.configuration(dict(tke_budget=1,bl_mynn_output=1));shape=(4,2,3)
        history=H.seed(shape,config)
        H.validate_restart(history,config,1,shape)
        for bad,marker in ((None,None),(history,0),(history._replace(output=None),1),
                           (history._replace(tke=(jnp.full(shape,jnp.nan),)*5),1),
                           (history._replace(tke=(jnp.zeros((1,),jnp.float32),)*5),1)):
            with self.assertRaises(ValueError):H.validate_restart(bad,config,marker,shape)
        with self.assertRaises(ValueError):H.validate_restart(history,H.configuration({}),1,shape)

    def test_schema_roundtrip_and_off_has_no_device_leaves(self):
        config=H.configuration(dict(tke_budget=1))
        history=H.seed((4,2,3),config)
        flat,definition=jax.tree.flatten(history)
        rebuilt=jax.tree.unflatten(definition,flat)
        self.assertIsInstance(rebuilt,H.MynnBudgetHistory)
        self.assertIsNone(rebuilt.output)
        self.assertIsNone(H.seed((4,2,3),H.configuration({})))
        self.assertEqual(jax.tree.leaves(None),[])

    def test_configuration_refuses_invalid_or_unresolved_switches(self):
        for key in ('tke_budget','bl_mynn_output','bl_mynn_edmf'):
            for value in (2,-1,[0,1]):
                with self.assertRaises(ValueError):H.configuration({key:value})
        self.assertEqual(H.configuration(SimpleNamespace(tke_budget=1)),
                         dict(tke_budget=1,bl_mynn_output=0,bl_mynn_edmf=1))


if __name__=='__main__':unittest.main()
