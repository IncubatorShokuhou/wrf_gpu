# MERGE — lane/t-cammgws-redfix c9de2b0b9f

Parent `673cf0823`; exactly one changed file, `tests/test_aot_cheap_key.py`, +35/-0. Both commits have identical `src` tree `c7f490542f40c50627680fc3eb7272561a88bd79`. Cherry-pick only this test commit.

The allowlist entry is justified: `bootstrap_refusal` reads environment/CLI inputs to decide whether import is refused; it does not mutate model constants. In `gpuwrf/__init__.py` a truthy result is printed and immediately raises SystemExit(2), before the JAX import/configuration hooks. Every surviving import retains `_cammg_refusal=None`; source inspection shows its only uses are assignment, guard and error message. The new test pins admitted MP11 and refused km3 behavior.

The exemption is the fully qualified call `gpuwrf.io.cammg_config.bootstrap_refusal`, not a module, binding, environment prefix, statement or bare function-name exemption. The scanner's existing AST traversal still inspects arguments and containing conditions, and scoped name resolution distinguishes unrelated same-name functions. This satisfies E116.

Independent CPU checks: **43 PASS**, 56 deselected, 37.30 s (scanner position/alias/rebinding/class regressions, real full import-time census, new bootstrap invariant and registry resolution). Five synthetic exact-call controls show only the exact bootstrap call exempted; argument, condition, neighboring env reads and an unrelated same-name function all remain offenders. Removing this entry reproduces the original call offender. A deliberately broadened allowance is **KILLED by AssertionError, zero setup errors** (`REDFIX_AUDIT.json`).

Test-only change: model arithmetic, compiler artifacts, cache keys, launches, allocation and readiness are unchanged from the parent by exact source-tree identity. No GPU work or repeated numerical validation is needed. Review snapshot and pytest files are removed after the last consumer.
