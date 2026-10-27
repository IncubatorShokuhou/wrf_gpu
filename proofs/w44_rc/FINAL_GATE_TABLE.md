# v0.4.4 RC CPU gate

Final runtime source5317c96541, src a22dea7112; delivery commit is the commit containing this gate table.

| Gate | Result | Evidence |
|---|---|---|
| Composition | IN1,5,6,7,8,9+registry; OUT2,3,4. Exact patch proofs; no manual source conflicts | COMPOSITION.json, ITEM9_COMPOSITION.json, REGISTRY_COMPOSITION.json |
| Tier A | 4209PASS /34SKIP /0 unresolved | CPU_CLASSIFICATION.json |
| Release defaults | 102PASS /1XFAIL /0FAIL | CPU_CLASSIFICATION.json |
| Registry affected | 56PASS;22 additional package checks | registry_affected.json |
| Failure classification | Two cache assertions reproduce on exact700a with cache disabled; both sources2PASS with private cache enabled. Reader coverage passes700a, failsf157, fixed5317. Two W4 files absent both trees=N_A, notPASS. | classification/*.json, TIERA_ELIGIBILITY.json |
| Native/E41 | Five prescreens STACK16/32/32/32/32B. Final five full-constant compiler protobufs and native bodies/locations/launches EXACT4b. OptimizedHLO/E41 carry by exact compiler inputs; no new optimized/GPU measurement. | REGISTRY_FINAL_IDENTITY.json |
| Identity vs v0.4.3 | Five valid DIFFERENT pairs: topo+QSQ+E84 source attribution; existing native bodies/calls/grids exact | IDENTITY_TABLE.json, IDENTITY_ATTRIBUTION.json |
| Registry identities | Package/runtime-data/checkpoint/bundle IDs move. Actual PROD program-key components unchanged; GPU exec-key bytes unmeasured. | REGISTRY_IDENTITY_EFFECTS.json |
| Launchers | Independent READYe541;5exactV9entries+all8RW3paths+posts0;4SwissfiniteCPUroots,18largePLAN_ONLY/N_A under manager exception. Final pin metadata delta external. | LAUNCHER_DELTA_V5.json, FINAL_PIN_DELTA.json (scratch) |
| Known issues / scientific scope | kInput664 component-vs-canonical scope, internal AOT bypass/legacy metadata, OUT flips/E84 deferred families disclosed. Final GPU/RW1–4/D6/72h/speed/RSS/VRAM remain unrun. | KNOWN_ISSUES_RC.md, RW_SESSION_PLAN.md |

No GPU was started by this lane; no merge or release authorization is implied.
