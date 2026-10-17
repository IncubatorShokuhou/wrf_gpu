# REJECT — s2-cammg 23fc64fea

Independent critic: rv-s2-cammg, 2026-10-14. Scope: opt-in CPU qualification and caller wiring; no GPU execution.

## Blocking finding — P1: MYNN heat diffusivity never reaches CAM-MG

The admitted `mp_physics=11, bl_pbl_physics=5` configuration passes permanent zero `KVH3D` after the first step. `physics/cammg/adapter.py:45` reads `state.cam_kvh3d`; the only nonzero producer is the CAM-UW branch in `coupling/scan_adapters.py:1913`. The non-CAM-UW path in `runtime/operational_mode.py:9559–9564` updates TKE alone. MYNN's existing `out.kh` is not retained for this consumer. Cold binding initializes the held diffusivity to zero and does not load `EXCH_H`.

Pristine WRF instead passes `KVH3D=EXCH_H` (`phys/module_microphysics_driver.F:3031`). MYNN populates that shared field (`module_pbl_driver.F:1744`, `module_bl_mynnedmf_driver.F:596`). CAM activation/mixing consumes it (`module_cam_mp_ndrop.F:310`); it is not an unused diagnostic. The deliberate first-step zeroing in the CAM driver does not authorize zeroing later calls.

Measured witness: all 48 original fixture calls × 52 fields first passed literal BYTE. Changing only `KVH3D` from its frozen 5 m²/s profile to zero changes physical returns in 15/24 second-step calls. In `c14_s2`, the qualified literal body returns `nc3d[0]=10,182,575 #/kg`, versus pristine WRF `10,187,772`; the independently frozen original-WRF enclosure is `[10,187,577, 10,187,967]`. `qndrop` and `wsedl3d` also leave that enclosure; fatal remains zero. This is a caller-sensitivity witness using the original-qualified literal body, not a coupled-forecast comparison. See `missing-kvh-gate.json` and `column-audit.json`.

Required correction: retain and pass the actual selected PBL's heat diffusivity at WRF's time and vertical-index seam, including MYNN. Audit the adjacent generic `0.5*QKE` TKE substitution against each actual WRF caller; physical equivalence alone is not caller proof. Add an independent producer-to-CAM handoff gate with nonzero diffusivity and a deletion mutant, then run only affected final-candidate checks. Preserve WRF's lack of a mandatory mp11/bl9/ra3 pairing. Reuse the existing computed coefficient and allocated carry; assess extra launches, copies, temporaries and compile cost without recomputing PBL.

## Recipe omission — P2: support matrix row

Add the experimental mp11 row and limits to `docs/release/SUPPORT_MATRIX_v0.3.4.md`'s v0.4 development table. The candidate adds `docs/CAMMGMP.md` and a roadmap row but omits the common recipe's support-matrix entry.

## Evidence accepted

- [M] Independent targeted suite: **54 PASS, 19.07 s** (own tests, contracts, v018, NetCDF lock).
- [M] **48 × 52 literal BYTE** to frozen pristine-WRF outputs; pristine source SHA inventory verified.
- [M] **1,296 retained native output arrays** (48 × 27) finite and inside the original enclosure. Independent AST/asset reuse seal passes for all 13 scientific modules; the original measured source and reachable reference have identical scientific blobs. No duplicate native compilation.
- [M] Independent precipitation-removal mutant: **2 genuine AssertionErrors, 0 setup errors**, 0.60 s.
- [M] Full real Swiss own-step CUDA lowering on CPU: **PASS**, 49,773,683 StableHLO characters, 56.344 s; native CAM custom-call present. Corrected setup uses Triton, native trace dispatch and the captured CUDA target; initial default-Mosaic launcher refusal is separately retained. This is lowering, not native compilation or execution.
- [M] Independent full default hashes match: Swiss **f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d**, PROD d01 **6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b**.
- Producer's source-bound Swiss two-step receipt reports NF/fatal/guards **0**, full restart BYTE and 4.33 GiB peak RSS under 6 GiB/no swap. It establishes finite continuation and restart, not correct PBL caller operands. Packaging, both loaders, fatal enforcement, restart/history and actual AOT namespace collision evidence inspected; targeted tests passed.

Performance/memory: default program identity is preserved. Producer measured CPU native temporary storage 241,600 B per 24-level column and compile 46.7–65.4 s; one GPU program per column is source-inferred. Actual GPU latency, VRAM, compilation/readiness, E41/roofline and forecast D6 remain unknown/pending and are honestly disclosed. No GPU or forecast-fidelity approval is given.

All review CPU/JAX arms used immutable 23fc64fea, pinned cores 16–19 at nice19, MemoryMax6G/MemorySwapMax0, explicit QUIET guards and timeouts; persistent cache NONE. No producer files were edited. Experiments RC01–RC09 are appended to main. Verdict applies to this exact candidate.
