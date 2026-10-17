# NSSL Ni source ownership

`_pbl_qni_active` mirrors Registry membership of `scalar:qni`. State.Ni is
materialized for schemes that do not own this WRF scalar, so presence of that
leaf does not authorize a PBL scalar tendency.

- `Registry.EM_COMMON:3037,3055`: NSSL's base package has only moist species;
  `nssl2mconc` adds qni only when `nssl_2moment_on==1`.
- `module_check_a_mundo.F:3397-3430`: 17/22 enable two-moment mode; 19/21
  normalize to 18 with two-moment mode disabled.
- `module_pbl_driver.F:1866-1877`: MYNN copies RQNIBLTEN to scalar_tend(P_QNI)
  only with Flag_qni. CAM-UW's source insertion checks
  P_QNI >= PARAM_FIRST_SCALAR (`module_physics_addtendc.F:963`).
- `module_mp_nssl_2mom.F:2916-2932,3503-3512`: CCI input/output is conditional
  on ipconc>=5. There is **no direct Ni update** for mp19/21. The lane's initial
  contract premise about retaining a non-advected Ni update was corrected from
  these original source branches. Its real carried mass/volume/CCN outputs stay
  on the existing direct post-RK MP state replacement path.
- `solve_em.F:3809-4016`: microphysics runs after the scalar RK loops; it does
  not provide MYNN's frozen RK scalar tendency.

`lower_cases.py` uses c14/c15's exact CLI inputs and `_load_domains`, restores
the native Pallas facade after CPU initialization and lowers the release
`_advance_chunk_fori` program to CUDA with JAX's explicit lowering platform.
It passes the actual AOT argument pytree, including the traced int32 start
step, instead of capturing namelist/clock inputs in a wrapper. c14 selects
mp19/cu99; the actual c15 input and GPU receipt select mp21/cu0 (the original
lane contract's cu84 description was stale).
It executes no GPU code and does not claim CUDA executable compilation or D6.

`build_reference.py` copies the original NSSL body byte for byte and specializes
the already audited legacy19/21 oracle caller. It executes two calls on actual
Swiss hour-six columns (44 levels), with observed QNICE deliberately retained
as non-owned CCI to prove nonvacuous preservation. These are MP caller gates,
not two steps of the full WRF forecast. Full GPU/coupled D6 remains v-prep's job.

The fix adds only a static Python predicate. Active-QNI/default programs retain
their old arithmetic. Inactive-QNI programs remove the erroneous tendency
construction; no new carry, intermediate array, launch or transfer is added.
GPU timing/peak-memory consequences are unmeasured until v-prep reruns c14/15.

The shared consumer suite has 335 tests. Its first CPU arm reached the 600 s
limit after 78 PASS reports, all in the first smoke module. The same immutable
tree and identical ordered collection resumed only the other 257 tests,
which passed; `NI10_resume.json` binds the original log hash and both node
lists. Those 78 numerical checks were not repeated. Targeted checks add 73
PASS; producer mutations produce four (old gate) and six (deleted active
sources) genuine assertion failures with zero setup errors.
