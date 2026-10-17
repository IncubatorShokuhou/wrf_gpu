Jensen component audit — WRF 4.7.1
================================

`INIT_SENSITIVITY.json` records the unchanged full public caller under
`-finit-real=inf` versus the frozen `-finit-real=zero` build. All 34 REAL4/REAL8
column, scalar and input JSON files are BYTE identical and finite. Integer init
is zero, logical init false, and FP traps are off in both builds. This is evidence
for the exercised corpus, not proof that every possible input is independent of
uninitialized storage. The pristine bodies have not been edited.

`vaporgrow` declares RDOUT INTENT(OUT) at original line 3090 but assigns it only
on the cold branch (line 3293). The warm branch at 3340–3346 leaves the caller's
RHODEPOUT slot held; the actual caller initializes it to RHOI at line 772. The
port preserves that observed held value explicitly. The 162 warm calls per
precision in the actual-call corpus exercise this behavior. It does not add an
ice-density repair or modify original WRF.

The original aggregation routine reads T into QQ before initializing T
(original 4004–4010). QQ feeds the energy-transfer scratch QRXFER, which is never
used to compute the returned mass/number/diameter results. The public init
sensitivity above finds no resulting output difference. The port can omit that
unused energy scratch while retaining the returned arithmetic.

Compiled rounding disclosure
----------------------------

The literal vapor path matches all 28,824 pristine output values BYTE. Its
integer powers use the original compiler's multiply chains; the reference test
backend uses scalar compiler libm rather than NumPy SIMD log/power.

The compiled CPU path has small residual differences outside a separately
frozen original-WRF enclosure from 29 actual arguments independently perturbed
by one ULP and original FMA/no-FMA builds (3,401,232 finite observations).
`VAPOR_COMPILED_DEVIATIONS.json` records every affected field, row, value and
original lower/upper bound. The maximum excess is 28 ULP in either precision:
2.4092642e-6 relative in REAL4 and 3.5416832e-15 in REAL8. This is disclosed under
the wave standing compiler policy; full coupled D6 decides scientific
acceptability. No additional fp64 fallback, masking clamp, or general numerical
tolerance was introduced.

The compiled test requires the independent original enclosure for every other
value and seals the exact observed deviations separately as a regression check.
That seal is not independent fidelity evidence. Jensen remains unregistered
until the complete scheme, wiring and required checks are finished; GPU and D6
qualification remain pending.
