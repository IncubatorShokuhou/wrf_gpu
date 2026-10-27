# Autoconversion threshold and rate

Separate CPU-qualified controls; the combined program, GPU component/root,
E41 and cost gates remain open. The four-parameter Nt GPU packet does not
select these controls.
The public runtime binder refuses both experimental names until their own
GPU/E41/cost gates pass; low-level CPU qualification remains available.

Both controls are gpuwrf extensions to source literals in pristine WRF
`phys/module_mp_thompson.F`, not named WRF PARAMETERs or namelist options:

- `autoconversion_threshold`: default 1e-5 kg m⁻³, the cloud-water density
  guard `rc(k) > 0.01e-3` at :2180. Values below WRF R1=1e-12 are refused
  because inactive cloud has no initialized slope for that branch.
- `autoconversion_rate`: default .027, the dimensionless coefficient in
  `zeta = 0.027*rc(k)*zeta1` at :2186. This is the coefficient itself, not a
  relative multiplier. Exactly zero is supported. WRF's joint rate limiters
  remain unchanged.

Changed-value CPU-WRF references recompile the original module with only the
respective literal changed. Original source, input columns and table assets
remain intact. The native implementation uses two REAL scalar operands;
there is no new table, State leaf or output allocation. Actual GPU cost is
unmeasured.

- AC00: original-operator coverage on the 24 real mixed-phase columns found
  97 positive autoconversion cells. Halving/doubling the default threshold
  was inactive because the droplet-size activation condition was stronger.
  Before examining candidate errors, the changed threshold was set to
  1e-3 kg m⁻³ (removing 90 active cells in 14 columns), and the changed rate
  to .054. These are nonvacuous validation values, not forecast recommendations.
- AC02/AC06: substituting the default literals gives the original operator
  AST exactly; fixed-width internal aliases also preserve native OFF raw HLO
  `12a9494b…` (628924 B). Changed ON values have identical raw HLO
  `f4abfff9…` (629254 B) and canonical graph, with one native Pallas call.
- AC03: CPU compiled default outputs are 20/20 BYTE on the real 24×44 corpus,
  dt=18 s. Each changed control affects five leaves. Only OFF and ON trace.
- AC04/AC05: threshold 1e-3 and rate .054 separately pass the original
  10-field WRF increment bounds; retaining the default value fails each
  omission control. No tolerance is changed.
- AC07: 97 tests pass in 2.38 s, including zero rate on a real column,
  threshold admission, restart value digests and value-independent keys.

Evidence: `<USER_HOME>/wrf_gpu2_lanes/w44-thparams/AC_SUMMARY.json`, SHA256
`509456e052e62433ca6b8bb9cf8a2211678ace1681dbb3d76d84fb873fd59293`.
The summary identifies immutable source `631bc12073745dd561a8b56173523f40645f4040`
and the AC00–AC07 artifacts. These measurements establish isolated CPU
behavior, not combined-parameter or GPU qualification.
