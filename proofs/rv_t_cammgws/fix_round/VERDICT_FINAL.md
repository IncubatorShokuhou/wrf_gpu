# MERGE — lane/t-cammgws 3ec98090e1

Source tree `67102aecc183992adc21e42b41f17c22dfe8a693`. The four items in the first FIX verdict are closed. This admits the lane implementation with the documented opt-in resource limits; it does not declare the pending coupled D6 gate passed.

## FIX closure

1. Compiler identity now hashes all installed stock-Triton package bytes, including the Python stages, plus route/kernel sources. It enters the mp11 baked/AOT key and the in-process compile memo. The package digest is cached per process, consistent with an immutable installed toolchain. Exact 3.6-family validation and actionable missing-dependency errors are present. Independently rerun route suite: **14 PASS in 10.24 s**. Independent replacement of the content digest with a constant: **KILLED by AssertionError, zero setup errors**.
2. CG35 export/reload are different PIDs and bind the same executable SHA `1b1d27e123f9b85adabe83961b4faf196fbe1cafce7c55d6d275b5715986696d`. The reload process forbids stock compilation and executes the deserialized custom call: **27/27 outputs BYTE, fatal BYTE**. Single adapter/full_column route is shared by nested/fused/batch consumers. This proves the new custom-call reload mechanism, not a full nested forecast.
3. The obsolete mp11 CPU-ONLY/refused prefix and incorrect plugin-compiler wording are removed; stock Triton dependency and coupled/resource disclosures remain. km3 admission behavior is unchanged by this FIX.
4. Independently checked **52/52 receipt manifest files**, no missing files or SHA mismatches. The PTX embedded in CG35 is pinned as `9e34e7f47dc95d6d9a49b2eccc38a333c414adad489b1c78681e1d4f1b1aa1fe`. An offline CUDA13.2 ptxas assembly of that PTX is pinned/scanned as cubin `fc11e7442b39836567865961f070cba01dc1f943e0fbc2c35617bbea5d7b6a9f`: 255 registers, STACK1240 B, no >1k nonleaf callee. **This is an offline assembly, not a captured loaded image**; the packet explicitly preserves that limit and correctly says the XLA E41 PASS scans only eight glue kernels. The known STACK excess remains disclosed under the opt-in policy.

Numerical control evidence carries forward from the first review: 972 paid GPU arrays finite and within the frozen bounds; original CPU test126 PASS and independent frozen Swiss/PROD identities retained. Producer final-tree default identities match f672a573/6fc715a2. No GPU work was performed by this critic, and no frozen numerical arm was repeated.

## BYTE and c89 reuse — **NO for the complete candidate**

The compiler FIX alone is arithmetic-neutral relative to 9a40424b0: independently normalized ASTs match for `phys_cammg.full_column`, the stock compiler stages, and the CUDA lowering after removing the new compiler-identity memo argument. **No files under physics/cammg changed.** This is source-based equivalence of the isolated route change, not a measured old/new whole-forecast BYTE comparison; PTX debug locations and executable/cache hashes can change.

However, `3ec98090e1` includes main merge `9034beb62f`, which contains the F1 land-precipitation correction. `_initial_carry_for_run` now seeds `noahmp_precipitation` for every nonzero MP, whereas 9a40424b0 seeded only MP8/5/7/15/38. The real c89 configuration is MP11 + CAM-UW BL9 + Noah-MP4. **Independent execution of both original/final initialization bodies on the same tiny CPU state confirms old None versus new five-field precipitation carry.** Post-RK MP now stores rain/snow in that carry and the next Noah-MP surface call consumes it (`noahmp_surface_hook.py`, `view.replace(**precipitation._asdict())`). The paid active44 column has nonzero rain/snow. This is an active scientific coupling change, not documentation or keying.

Therefore **outputs of the whole candidate are not certified BYTE vs 9a40424b0, and tonight's 9a40424b0 c89 run cannot supply the final composed 3ec98090e1 24h D6 gate**. Preserve it as exact-source 9a404 validation and stock-route evidence. Do not revert the correctness fix merely to reuse it. The final composed source needs its affected c89 coupled gate once, or a separately established full-program equivalence proof that resolves this active coupling difference.

## Cost and remaining operational gates

Content hashing costs producer-measured 1.04 s once per process, outside the timestep loop; no extra column launches or arithmetic were added by the compiler FIX. CG35 deserialization is 5.5 ms, but first execution still costs **26.1 s of in-process PTX assembly per fresh process/shape**, then warm one-column17.8 ms. No cross-process native-binary cache benefit is claimed. The main-merge precipitation change adds five fp32 surface carry arrays (about35,280 B for42×42, inferred payload only); compiled traffic/launch cost has not been timed by this critic. C89 full forecast fidelity and operational timing/memory remain separate gates.

Independent evidence: `FIX_SOURCE_AUDIT.json`, `fix_tests.xml`, `fix_tests.log`, `fix_audit.py`, `fix_audit.log`. The first immutable verdict is preserved unchanged.
