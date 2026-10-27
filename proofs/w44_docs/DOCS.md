# Merged-tool documentation — partial review

Scope: seven tool pages for already-merged features, the regional entry guide,
indexed command entries and the documentation checks. README, unfinished
warmstart/budget/topo/parameter pages and all runtime changes are excluded.
Base: main `76105c5318` (includes `d1db02bd62`).

## User pages and evidence

| Tool | Page | Qualification source |
| --- | --- | --- |
| Input override | `docs/initial_overrides.md` | `proofs/w44_override/OVERRIDE.md`; host/file only, edited GPU forecast pending |
| Porosity helper | `docs/soil_porosity.md` | `proofs/w44_override/SOIL_POROSITY_GATE.json`; explicit host predicate, no control-fidelity claim |
| Radiation diagnostics / QI_BL | `docs/radiation_diagnostics.md` | `proofs/w44_raddiag/RADDIAG.md`, CPU/GPU receipts; original-WRF tests and emitted-history GPU comparison, paired counters incomplete |
| PBL clouds in radiation | `docs/radiation_pbl_clouds.md` | Same radiation report; scalar icloud_bl shared across domains |
| Non-MODIS refusal | `docs/noahmp_land_use.md` | `proofs/w44_lutab/LUTAB.md` / `RECEIPT.json`; host admission and unchanged default programs |
| Checkpoint branch / short windows | `docs/checkpoint-branching.md` | `proofs/w44_branch/BRANCH.md`, output/runtime-data contracts; host admission, GPU branch checks pending |
| Precompile queue | `docs/precompile.md` | `proofs/w44_precompile/PRECOMPILE.md` / CPU packet; ordinary-GPU priming queue, deviceless GPU compilation unqualified |

`docs/regional_optimization.md` links the shipped pages and existing guides;
it does not depend on the excluded unfinished pages. Stable configuration IDs
remain unchanged. Public prose uses geographic descriptions and tool names
instead of internal W4/W5/PROD/WN3/manager labels. The development GPU wrapper's
lease requirement remains explicitly explained.

## Critic delta

Original review: `cr-w44-docs` FIX on `a25b4a1f91`; the same critic checks
this delta. Preserve that review; do not rewrite it.

1. Seven short, ID-bearing command headings replace the long single regional
   link paragraph. Search uses the unchanged builder and app.js. The test
   exercises actual shipped ranking and requires eight queries, including
   `precompile` and `branch`, to find their dedicated URL in the sidebar's
   first eight results.
2. Undefined internal labels are removed from the public prose. A style check
   allows exact command/configuration names and source link targets but rejects
   those labels in prose. Measurements, option meanings and open qualifications
   remain unchanged.
3. A fresh docs-only stack begins directly on current main. It supplies all
   page/link/checker dependencies without importing README or unfinished pages.
   The README draft is a separate commit on top and must not be merged as part
   of this partial review.

The previous lane history remains at
`archive/w44-docs-before-crfix-20261024` (`a25b4a1f91`). Its numerical-source
identity, six CPU example passes and technical review carry forward. No model,
GPU, full benchmark or numerical-validation rerun is needed for these prose,
index and integration changes.

## Checks

- `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/w44_docs/verify.py --require-merged-tools --output CHECK.json`
- `JAX_PLATFORMS=cpu python tools/docsite/build.py --out BUILT_SITE`
- `JAX_PLATFORMS=cpu node proofs/w44_docs/check_search.js TREE BUILT_SITE SEARCH.json`

Run on an immutable docs-only snapshot, before adding the README commit.
Record CLI/link/Registry/style and search results, unchanged model/builder/app
source, literal example-code identity, and the exact changed-file list. The
docs-only tip must contain no README, approval-file, runtime or unfinished-page
diff versus main. The separate README commit may be applied later.

Status: **READY-FOR-CRITIC — DELTA ONLY**, same critic `cr-w44-docs`.
Immutable docs-only content `c2f0b2fdcb`, directly on main `76105c5318`:

| Check | Result |
| --- | --- |
| Exact CLI, local links, WRF names | 22 commands, 48 links, 49 history names PASS across the seven pages and entry guide |
| Public language | PASS; internal labels absent from prose, exact command/config IDs preserved |
| Site build | 20 pages plus search; 186 indexed entries PASS |
| Actual shipped search | All eight queries find their dedicated entry in the first eight sidebar results; precompile/branch each rank first |
| Negative search control | Removing branch index entry causes a genuine AssertionError, no setup failure |
| Integration | Direct main parent, README identical main, no approval/runtime/unfinished-page diff |
| Builder/search source | Exact main bytes; no refactor |
| Prior six CPU examples | Carried by unchanged runner, runtime and literal example code; no repeat |

The smaller CLI/link counts reflect the corrected review boundary: README,
unrelated existing pages and unfinished pages are excluded from this standalone
stack. `receipts/DOC11.json` binds the complete changed-file list, source/page
hashes, exact search results and preservation checks. Evidence/logs are under
`<USER_HOME>/wrf_gpu2_lanes/w44-docs/DOC11/`.

Full v0.4.4 release readiness and all still-open implementing-lane gates remain
open. Select only the docs-content and docs-seal commits supplied in MAILBOX;
the separate README commit above them is outside this partial review.
