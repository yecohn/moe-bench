## Problem Statement

The HTML report (`results/<run_id>/report.html`) currently has ~12 sections split across two dividers ("Scientific summary" and "Drill into candidates"). It tries to serve three audiences at once — decision-makers, design-space explorers, and hypothesis-testers — and as a result none of them are served well. Specific concrete pain points:

1. **The headline recommendation is split.** "Decision summary" at the top compares only *backends* (vllm vs sglang at their best configs). The actual *candidate-level* recommendation ("ship `vllm_tp2dp2_ep`") lives in a separate "Global winner" section below a divider. A reader who skims the top of the page learns which backend to use but has to scroll to learn which config.
2. **The global winner is computed by a buggy geomean.** `_geomean` in `html_report.py` filters zeros from its input, which under the default `goodput_at_slo` objective means SLO-violating workloads are silently dropped from a candidate's score rather than penalizing it. A candidate that violates SLO on 5 of 6 workloads and wins on the 6th gets its global score computed from just that one workload, making it look more robust than candidates that pass SLO on all six.
3. **Failures are invisible.** The header shows "N failures" as a counter but no detail. A reader cannot tell whether the global winner is genuinely best or just a survivor of a sweep where most candidates crashed.
4. **The constraint slider is an isolated feature.** Dragging the TTFT/TPOT sliders re-ranks only the constraint-leaderboard section. The Decision summary hero and the robustness leaderboard remain pinned to the YAML-defined SLOs, so a reader exploring "what if my SLO is 1000ms instead of 2000ms" sees an inconsistent page.
5. **No quantitative evidence accompanies the recommendation.** The hero says "vllm wins 4 of 6 cells with 1.3× geomean." It does not say *which config*, by *how much* it beats the runner-up, or what its *worst-case* workload looks like. A reader cannot judge whether the win is decisive or marginal.
6. **The page is long for big sweeps.** Candidate-detail cards render for every candidate; with 24 candidates the section is hostile to navigate. No top-K cap exists.
7. **Section names communicate content, not purpose.** "Scientific summary" and "Drill into candidates" describe *what's there*. They don't tell a reader *what question each section answers*.

The compound effect: a reader trying to make a deployment decision in 30 seconds gets a partial answer, has to scroll for the rest, can't trust the answer without verifying the SLO assumptions, and may not realize a quarter of the sweep failed. The report needs to commit to a primary job and lay out its content to serve that job.

## Solution

Reframe the report around one primary job — **Decide which backend + config to deploy** — and redesign the top of the page to answer that question in one screenful, with the existing explore-mode content preserved untouched below the fold.

From the user's perspective, the new report:

- **Opens with a single, complete recommendation.** A merged hero card says "Deploy `<backend>` with config `<candidate>`," includes the resolved server parameters, names the objective and SLOs ("Best by `goodput_at_slo` subject to `p99_ttft ≤ 2000ms, p99_tpot ≤ 100ms`"), and quantifies the win in three numbers (wins / margin over runner-up / worst-case ratio with the worst workload named by label).
- **Shows failures up front when they exist.** A conditional callout panel appears above the hero only when at least one cell failed, listing affected candidates and workloads with their failure status. Clean runs show no callout at all.
- **Lets the reader pressure-test the recommendation interactively.** The TTFT/TPOT sliders re-rank the hero, the robustness leaderboard, and the constraint leaderboard in sync. A reader can drag the sliders to their actual production SLOs and immediately see whether the winner changes.
- **Surfaces only the top-5 candidates above the fold,** with an expander to reveal the rest. Works equally well for 9-candidate and 24-candidate sweeps.
- **Reorganizes the page into two purpose-named regions.** A top "Decide" divider holds the hero, the leaderboard, and the slider. A bottom "Explore" divider holds every existing chart and per-candidate card untouched, for readers who want to dig deeper.
- **Caps the candidate-detail cards section** at a configurable top-K so long sweeps don't produce hostile pages.

Behind the user-facing surface, the geomean is fixed: zero scores are floored to a configurable value (default 0.05) before the geometric mean is computed, so SLO-violating workloads multiplicatively penalize a candidate's global score instead of being silently dropped.

The optimization itself is unchanged. `objective.maximize` and `objective.constraints` in the YAML still define what "best" means; `rank.py` still performs the optimization; the report continues to be a presentation layer over `rankings.csv`.

## User Stories

1. As a deployment owner, I want a single recommendation at the top of the page naming both the backend and the specific server config to ship, so I do not have to scroll to learn which config to actually deploy.
2. As a deployment owner, I want the recommendation to display the resolved server parameters (or a one-click expander to reveal them), so I can copy them directly into my production config without leaving the page.
3. As a deployment owner, I want the hero to state the objective and SLOs the recommendation is conditional on, so I am never guessing what "best" means.
4. As a deployment owner, I want the hero to state how many workloads the winner won, by how much it beats the runner-up, and what its worst-case workload looks like, so I can judge whether the win is decisive enough for my use case.
5. As a deployment owner, I want the worst-case workload identified by label (e.g. `prompt4096_out128_conc64`), so I can immediately decide whether that workload matches my production traffic.
6. As an operator running a sweep, I want a conditional failure callout above the recommendation when any cells failed, so I know to discount the recommendation when sweep coverage was incomplete.
7. As an operator running a sweep, I want the failure callout to list which candidates and which workloads failed, so I can diagnose whether the failure pattern is concentrated in one part of the design space or spread across it.
8. As an operator running a sweep, I want the failure callout to be hidden entirely on clean runs, so the page is not cluttered with empty status panels.
9. As a designer evaluating tradeoffs, I want to drag the TTFT/TPOT sliders and see the hero card update live with a new winner if applicable, so I can interactively confirm the recommendation holds under my real SLOs.
10. As a designer evaluating tradeoffs, I want the robustness leaderboard above the fold to re-rank in sync with the sliders, so I can see the second- and third-place candidates change as I tighten or relax SLOs.
11. As a designer evaluating tradeoffs, I want the slider movement to also update the SLO text in the hero's "subject to..." line, so the hero never displays a recommendation that disagrees with the visible SLO settings.
12. As a reader new to the platform, I want the page divided into two purpose-named regions ("Decide" and "Explore"), so the structure tells me what each part of the report is for.
13. As a reader who trusts the recommendation, I want the "Decide" region to fit in one screenful, so I can act on the recommendation without scrolling.
14. As a reader who does not trust the recommendation, I want every existing explore-mode chart and per-candidate detail preserved below the divider, so I can verify the recommendation against the underlying data.
15. As a reader with a big sweep (20+ candidates), I want the robustness leaderboard limited to the top 5 by default with a "show all" expander, so the above-the-fold region stays compact.
16. As a reader with a big sweep, I want the candidate-detail cards section capped at a configurable top-K, so the page does not become unmanageably long.
17. As a sweep designer, I want `report.leaderboard_rows`, `report.top_k_candidate_cards`, and `report.robustness.zero_floor` to be configurable in the YAML, so I can tune the report presentation without code changes.
18. As a sweep designer, I want a candidate that violates SLO on most workloads to score worse globally than a candidate that passes SLO on all workloads, so the global ranking reflects robustness honestly.
19. As a sweep designer, I want the zero-floor value to be configurable (default 0.05) so I can tighten or loosen the penalty for SLO violations without changing code.
20. As a future maintainer, I want the recommendation-computation logic (winner, runner-up, margin, worst-case, objective/SLO labels) to live in a single pure function that takes rankings + candidates + slo + objective and returns a structured `Recommendation` object, so I can change the hero's HTML without touching the logic that decides what to show.
21. As a future maintainer, I want the failure-summary computation to live in a single pure function that takes failures + candidates + workloads and returns a structured `FailureSummary` object (or `None` for clean runs), so the callout rendering is a thin wrapper over testable logic.
22. As a future maintainer, I want unit tests for both pure functions covering: a normal sweep with a clear winner, a sweep with SLO violations on the winner's secondary workloads, a sweep with a one-shot lucky candidate (relies on zero-floor), a clean run (failure summary returns None), and a degraded run (failure summary lists affected cells), so trust-critical logic does not silently regress.
23. As an operator post-run, I want the report.html to gracefully handle the edge case of zero `objective.constraints` (hero drops the "subject to" clause; slider section is hidden), so the report does not break for SLO-disabled batch-style sweeps.
24. As an operator post-run, I want the report.html to gracefully handle the edge case of only one backend producing valid measurements (hero recommends within that backend, no head-to-head phrasing), so single-backend runs still produce a usable recommendation.
25. As an operator post-run, I want the report.html to gracefully handle the edge case of zero valid candidates (failure callout dominates, hero shows "No valid candidate met the objective" instead of a recommendation), so a fully-failed sweep produces a comprehensible report rather than a crash.

## Implementation Decisions

### New deep modules (testable in isolation)

- **`compute_recommendation(rankings, candidates, slo, objective) -> Recommendation`** — a pure function that takes the already-ranked CSV-equivalent data structures and returns a `Recommendation` data object with: `winner_backend`, `winner_candidate_id`, `winner_params`, `runner_up_candidate_id`, `wins`, `total_workloads`, `margin_pct_over_runner_up`, `worst_case_ratio`, `worst_case_workload_label`, `objective_label`, `slo_label`. Encapsulates the geomean computation (now zero-floored), tiebreak rules, runner-up identification, worst-case extraction, and human-readable label generation. No HTML, no I/O, no Plotly.
- **`compute_failure_summary(failures, candidates, workloads) -> FailureSummary | None`** — a pure function that returns `None` for clean runs, else a `FailureSummary` data object with: `total_cells`, `failed_cells`, `rate_pct`, `by_candidate` (list of candidates with at least one failure), `by_workload` (list of workloads with at least one failure), `statuses` (dict mapping each failure status to its count). The HTML failure-callout section becomes a thin renderer over this.

### Geomean fix

- **`rank.py`'s `_geomean`** (or equivalent helper) — change semantics from "filter zeros then geomean" to "replace zeros with `zero_floor` then geomean." `zero_floor` is read from `report.robustness.zero_floor` in the YAML; default `0.05`. The threshold is configurable so a future operator can tighten or loosen the penalty without code changes.

### HTML report changes

- **Merged Decision summary hero.** Combines the current `_section_decision_summary` (backend-level) and `_section_global_winner` (candidate-level) into one renderer that consumes a `Recommendation` from `compute_recommendation`. The standalone "Global winner" section is removed.
- **Conditional failure callout section.** Renders only when `compute_failure_summary` returns a non-`None` result. Anchored above the Decide divider.
- **Divider rename.** "Scientific summary" → "Decide" with subtitle "The recommendation and how to verify it." "Drill into candidates" → "Explore" with subtitle "Design-space tools and per-candidate detail."
- **Robustness leaderboard cap.** Defaults to top-5 rows with a `<details>` expander revealing the rest. Configurable via `report.leaderboard_rows` (default 5).
- **Candidate cards cap.** Top-K cards rendered, with a `<details>` expander for the rest. Configurable via `report.top_k_candidate_cards` (default 8). Cards beyond top-K are sorted by robustness rank.
- **Slider scope expansion.** The existing constraint-slider JS currently re-ranks only the constraint leaderboard. Expand it to also re-rank: (a) the hero recommendation (potentially changing the winner and refreshing wins / margin / worst-case), (b) the robustness leaderboard top-5 table, (c) the SLO line in the hero's "subject to..." clause.
- **Hero copy.** Three quantitative lines: (1) the recommendation, (2) the objective + SLO under which it holds, (3) wins / margin-over-runner-up / worst-case-ratio with the worst-case workload labeled.

### YAML schema additions

The `report:` block (currently only used for `legend_params`) gains three new optional keys:

- `report.leaderboard_rows` — integer, default 5.
- `report.top_k_candidate_cards` — integer, default 8.
- `report.robustness.zero_floor` — float, default 0.05.

All are optional; existing configs that omit them get the defaults and behave identically to the new layout.

### Section order in the final HTML

1. Header (unchanged).
2. Failure callout (conditional).
3. "Decide" divider.
4. Decision summary hero (merged).
5. Robustness leaderboard (top-5 + expander).
6. Constraint-conditional leaderboard (slider; same JS but expanded to drive 4 and 5 too).
7. "Explore" divider.
8. Operating curve.
9. Latency vs prompt length (TTFT + TPOT).
10. Decision map.
11. Tuning sensitivity.
12. Pareto explorer.
13. Parameter sweep parcoords.
14. Per-workload drill-down.
15. Candidate detail cards (top-K + expander).

### Degraded-state behavior

- No `objective.constraints` in YAML: hero's "subject to..." clause is omitted; constraint-slider section is hidden; recommendation is computed from raw `objective.maximize` alone.
- Only one backend with valid measurements: hero recommends the best candidate of that backend; runner-up is the second-best of the same backend; comparative phrasing ("beats runner-up") still works. No "vllm vs sglang" wording is emitted.
- Zero valid candidates: failure callout dominates; hero shows "No valid candidate met the objective" with the SLO context. No leaderboard rendered.

## Testing Decisions

Tests in this codebase are precedented by `tests/test_config_overrides.py` — pytest, the platform venv, no fixtures beyond a small local `_base_config()` helper. The same pattern fits the new pure functions cleanly.

### What makes a good test here

- **External behavior, not structure.** A test asserts that "given rankings X with SLO Y, `compute_recommendation` reports `vllm_tp2dp2_ep` as the winner with margin 12% and worst-case workload `prompt4096_out128_conc64`." It does not assert that the function calls a particular helper or uses a particular intermediate data shape.
- **Pure-function inputs and outputs.** Synthetic rankings / failures dicts as inputs; structured `Recommendation` / `FailureSummary` outputs verified field-by-field. No HTML strings, no Plotly figures, no I/O.
- **Survives refactor.** If the implementation moves from one helper to another, the tests still pass as long as the published behavior is unchanged.

### Tests to write

**For `compute_recommendation`:**

- Normal sweep with one clear winner across all workloads.
- Winner is best globally but second on one workload (verifies worst-case ratio and worst-workload label are correct).
- Candidate violates SLO on 5 of 6 workloads but wins one big — must NOT rank above a robust candidate (verifies zero-floor is applied).
- Tie between two candidates by geomean — tiebreak by median rank, then by win count.
- Only one backend has valid measurements (single-backend recommendation path).
- Zero valid candidates after constraint filtering — returns a sentinel / null recommendation rather than crashing.
- Non-default `objective.maximize` (e.g. `median_output_tok_s_per_gpu`) — the `objective_label` field reflects the configured objective.
- Empty `objective.constraints` — `slo_label` is empty / "no SLO" sentinel; recommendation still produced.

**For `compute_failure_summary`:**

- Clean run (no failures) — returns `None`.
- One candidate, all its workloads failed — `by_candidate` lists that candidate; `by_workload` lists every workload; `rate_pct` is computed correctly.
- One workload, all candidates failed on it — converse shape; `rate_pct` correctly reflects partial failure.
- Mix of failure statuses — `statuses` dict counts each correctly.
- Failures present but every candidate has at least one valid workload — `by_candidate` is correct subset; existing report still has a valid recommendation downstream.

### Tests deliberately NOT written

- **HTML output.** No snapshot tests on the rendered HTML. Snapshot tests rot quickly when copy evolves; the trust-critical logic is upstream in the pure functions.
- **Plotly figure structure.** Plotly figures are tested by visual inspection only, consistent with the rest of `html_report.py`.
- **Slider JS.** No browser-level test for the constraint slider. Manual inspection on a real `report.html` is the verification path, consistent with current practice.
- **YAML schema validation.** New `report.*` keys are optional with defaults; no test asserts they appear in the schema. A misconfigured key produces the default value, which is acceptable.

## Out of Scope

- **Cost-aware ranking.** The platform does not know GPU-hour cost. A `goodput_per_gpu_hour` objective would require external pricing data and is its own PRD.
- **Per-workload-class recommendations.** The "conditional set" decision shape (one winner per workload class) was considered and rejected during grilling in favor of a single global winner with worst-case-workload disclosure. A future PRD could add workload grouping.
- **Soft SLO penalty.** The cliff-edge behavior of `goodput_at_slo` (1ms violation → score 0) is unchanged. A soft-penalty variant would require changes to `rank.py:compute_goodput_at_slo` and is its own PRD.
- **Choosing the objective at view-time.** The objective remains fixed by `objective.maximize` in the YAML; re-ranking under a different objective requires editing the YAML and re-running `rank` + `report`. The slider only adjusts constraint thresholds within the current objective.
- **Failure prevention / cleanup.** The failure callout surfaces failures that have already occurred. Reducing the failure rate (more robust startup checks, better cleanup) is its own work.
- **CSV schema changes.** `rankings.csv` and `failures.csv` keep their current columns. The new computation reads what's already there.
- **A new objective for the sweep.** This PRD changes how `goodput_at_slo` aggregates across workloads (the zero-floor fix). It does not introduce new objective options.
- **Internationalization.** Hero copy is English only.

## Further Notes

- The zero-floor default of `0.05` is approximately "the worst non-zero score a candidate can achieve" under typical sweeps. It is intentionally high enough to mean a single SLO violation costs ~20× in the geomean factor, which empirically distinguishes robust candidates from flaky ones without producing absurd rankings. Operators with a high tolerance for some violations may lower it to e.g. 0.5; operators who want strict robustness may raise it to 0.1.
- The merged hero displays the resolved server parameters of the winner. For long parameter lists, the parameter block should be a `<details>` expander to keep the hero compact. Default to expanded for short lists (≤6 params) and collapsed for longer ones.
- The slider re-rank is computed in the browser using the same JS that already powers the constraint leaderboard. The data shape exposed to the page is `rankings.csv`-equivalent JSON. No new server-side computation runs at view time; the redesign reuses the existing embedded measurement JSON.
- After this lands, `context.md` should be updated to remove the "Robustness zero-floor fix described above" item from its Pending design items list — it's done.
- The existing `legend_params` key under `report:` is unaffected by the new `report.*` keys; they nest alongside it.
