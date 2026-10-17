# MERGE-WITH-FIXES — s2-glacier

Reviewed `lane/s2-glacier` at **0349f53d5ed240f0f46332e8c63b0f826f55417c**, against main base `f202fa34955a088f2adc19987276c9009fc2f101`. CPU qualification only. Reviewer: rv-s2-glacier, GPT Sol, 2026-10-14.

The physical glacier port, its operational partition, held QSNOWXY, four restart formats, namelist admission and PhysicsStepSpecs pass this review. One factual documentation correction is required before merge. Do not change the qualified arithmetic for that correction.

## Required fix — P2: QRAINXY is a defined per-point hold

`proofs/s2_glacier/UPSTREAM_WRF_BUGS.md:14`, `KNOWN_ISSUES.md:24`, `proofs/s2_glacier/QUALIFICATION.md:86`, and `src/gpuwrf/io/scheme_catalog.py:1358` incorrectly claim the pristine glacier driver retains an uninitialized/stale QRAIN from a preceding natural-soil column. That mechanism is absent:

- `module_sf_noahmpdrv.F:823` loads `QRAIN = QRAINXY(I,J)` before selecting the glacier branch.
- The glacier call has no QRAIN argument and leaves that local unchanged.
- `:1259` writes QRAIN back to the same point's QRAINXY.
- `NOAHMP_INIT`, `:2141`, initializes each point's QRAINXY to zero.

Therefore the qualified cold-initialized glacier path's `QRAINXY=0` is WRF-consistent, not a diagnostic deviation caused by undefined or preceding-column data. The compiler-initialization exception does not apply to this value. Source evidence: RVG07_qrain_source.json, pristine driver SHA256 `29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572`.

**Exact action:** correct all four locations above to describe WRF's initialized, per-point hold; remove the upstream-bug/stale-local attribution and the claimed deviation for the qualified cold path. State that arbitrary nonzero externally seeded QRAINXY is outside this qualification, rather than claiming full parity for that untested input. Retain the genuine inactive-HCPCT and missing glacier opt_snf=5 disclosures. Add a small source-bound note/check for the `:823/:1259/:2141` chain so this conclusion remains auditable. This requires prose/catalog metadata and a source audit only; no new carry, kernel, forecast, GPU arm or tolerance change is required.

## Independently verified evidence

- **332 tests PASS**, zero skips/errors/failures, 488.83 s: all three new glacier test modules, contracts, v0.18 fail-closed pins, namelist binding/honour, catalog, NetCDF locking and M7 checkpoint consumers. RVG01.xml, log.
- **Fresh original WRF:** rebuilt the unchanged glacier module, SHA256 `a35d61ea29f660f4f4c6384bc019d4a57ddc8b0c2ca0f3ef7c6e621b79b4dece`. All **4,896 calls / 331,584 defined values** reproduce the tracked savepoints byte for byte. Zero nonfinite defined outputs and zero defined-output sensitivity to zero versus NaN compiler initialization. Another **12,292 original-WRF calls** reproduce every endpoint of the frozen 58-case input-ULP envelope byte for byte. RVG02_original.json.
- **Actual production stages:** both GLA variants, 58 cases, **38 exported fields per variant**, all finite and inside the independently regenerated pristine input-ULP envelope plus its registered one-output-ULP margin. This supplements the owner's monolithic compiled-column test with the operational three-Pallas-stage implementation. RVG03_staged.json.
- **Genuine mutant:** deleted the actual operational glacier dispatch in process, preserving real module globals. One explicit `AssertionError: natural path selected instead of glacier routing`, zero setup errors, 2.05 s. The unmodified seam passed RVG01. RVG04_mutant.json, log.
- **Release identity:** independently reran the official native-facade CPU probe, redirecting only scratch. Swiss SHA256 **f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d**, 14,504,210 characters, matches the manager-approved glacier correction. PROD d01 SHA256 **6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b**, 14,708,888 characters, remains exact. Swiss, PROD.
- **Ice-free path:** independently re-censused all 57 inputs with zero missing: Swiss 22 glacier points, PROD's two and WN3's 54 domains zero. All input hashes/counts match the producer's census. Re-ran reduced-AST and constructor/pytree proof: 19 existing arithmetic/driver modules byte identical, original 32/14-child trees and None auxiliary data preserved. This is universal source evidence for WN3, not 54 measured GPU executions. RVG05_census.json, source proof, binding.
- **Reuse:** verified all five reported test XML hashes and final wiring/restart blobs, and final five physical module ASTs, against the qualified composition. Both real Swiss CPU three-step receipts have zero nonfinite values/guards and byte-exact resumed fourth steps on 183 leaves. Those completed forecasts are reused; no duplicate forecast was run. RVG08_receipts.json, RVG10_binding.json.
- **Native resources:** independently scanned all six preserved CPU-compiled cubins: STACK=0, LOCAL=0, PTX fp64=0, resolved callgraphs PASS; registers 148/128/128 and 151/96/128. These are compiler artifacts, not GPU execution measurements. RVG06_native.json.

## Performance, memory and scope

Ice-free source/pytree preservation adds zero model work or carry bytes. Affected grids use three glacier column kernels plus full natural/glacier gather/scatter traffic. Swiss adds a 7,056-byte rate plane, land carry 373,968 → 381,024 bytes. The reported per-stage logical output buffers, 8,096/8,272/8,976 bytes before backend elimination/aliasing, exclude natural-tile copies. Preserved CPU native compile totals are 194.389/170.151 s for GLA1/2; they are not GPU speed estimates. The natural-soil tile reshape and gather/scatter remain material GPU layout/traffic risks to measure.

GPU wall/launch counts/peak VRAM, whole operational executable E41 and the approved Swiss 24-hour comparison against original CPU-WRF remain mandatory manager final-validation gates. The new default Swiss graph cannot reuse the old glacier-free scientific qualification. No GPU was used and no release-fidelity or maximum-speed closure is granted here.

After the exact documentation correction, integrate under this one review with source/AST identity to the qualified math, affected metadata/source checks, and the existing numerical receipts. Do not repeat numerical validation for a prose-only successor.
