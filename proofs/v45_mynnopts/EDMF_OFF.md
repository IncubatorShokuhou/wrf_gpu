# bl_mynn_edmf=0: existing implementation audit

Base `3b93841d70` already implements this selection. Audit commit `383bf5162c` changes no
production bytes: default native HLO/constants/Pallas bodies and resource costs
therefore carry unchanged (E222/E224), with no duplicate forecast run.

The per-domain `&physics bl_mynn_edmf=0` binds through `mynn_budget_options`.
The operational source caller selects `mynn_sources_with_history`, whose static
adapter disables the master `_MYNN_EDMF` branch. With both history switches zero
it runs one production MYNN solve, allocates no diagnostic twin/history, and
removes plume work. It requires MYNN5 and native fp32; momentum-only disable
with the master on, TKE mass flux and downdrafts remain refused. Registry
package histories MAXMF/MAXWIDTH/ZTOP_PLUME are omitted with the master off.

Original-WRF day/night two-step column evidence is retained in
`proofs/w44_budget/receipts/P33_edmf0_day_oracle.json` and
`P35_edmf0_night_oracle.json`; production identity and cost are in P32/P34.
Those are real WN3 d03 columns, **not new PROD-column measurements**. Tests in
`tests/test_w44_mynn_budgets.py` cover binding, static keys, histories, restart
and refusals. `proofs/w44_budget/BUDGET.md` gives the measured scope and gates.

No isolated GPU speed/memory benefit is measured [U]. Source removes plume
computation [I]; native compiled consequences still require the plan in
GPU_PLAN.md. This is CPU-qualified, composed-GPU-pending support as already
documented in `docs/mynn-budgets.md`. The manager's 23:08Z amendment accepts the labelled
WN3/Swiss evidence; V045 adds actual runtime forwarding assertions for EDMF off
with each new PDF. These do not replace a native coupled forecast gate.
