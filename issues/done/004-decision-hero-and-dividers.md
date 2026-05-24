## Parent PRD

`issues/prd-report-redesign.md`

## What to build

Replace the current top-of-page Decision summary (backend-only) and the separate Global winner section with a single merged hero that names both the backend and the candidate to deploy, states what "best" means, and quantifies the win. Also rename the two dividers to communicate purpose. See the Solution section, Implementation Decisions → "Merged Decision summary hero" and "Divider rename", and the Degraded-state behavior subsection of the parent PRD.

Two parts:

1. A new pure function `compute_recommendation(rankings, candidates, slo, objective) -> Recommendation` returning the fields: `winner_backend`, `winner_candidate_id`, `winner_params`, `runner_up_candidate_id`, `wins`, `total_workloads`, `margin_pct_over_runner_up`, `worst_case_ratio`, `worst_case_workload_label`, `objective_label`, `slo_label`. Encapsulates the geomean (using the zero-floor from issue 002), tiebreak rules, runner-up identification, worst-case extraction, and human-readable label generation.
2. HTML changes:
   - Merged hero renders `compute_recommendation` output: recommendation line, expandable server-params block, objective + SLO line ("Best by `<objective>` subject to `<slo_label>`"), evidence triple (wins / margin / worst-case-ratio on `<workload-label>`).
   - The standalone "Global winner" section is removed.
   - Divider 1: "Scientific summary" → "Decide" with subtitle "The recommendation and how to verify it."
   - Divider 2: "Drill into candidates" → "Explore" with subtitle "Design-space tools and per-candidate detail."
   - Degraded-state rendering: no constraints → omit "subject to..." clause and hide constraint-slider section; only one backend present → recommendation within that backend, no head-to-head phrasing; zero valid candidates → "No valid candidate met the objective" hero, no leaderboard.

End-to-end: opening `report.html` shows one merged hero stating exactly what to deploy, why it's best, and how decisive the win is, plus two purpose-named dividers structuring the rest of the page.

## Acceptance criteria

- [ ] `compute_recommendation` exists as a pure function and returns the full `Recommendation` shape listed above (no HTML, no I/O, no Plotly).
- [ ] Uses the zero-floored geomean from issue 002 when computing the global score.
- [ ] Unit tests cover: clear winner across all workloads; winner second on one workload (worst-case fields correct); SLO-violator does NOT outrank a robust candidate (zero-floor working through `compute_recommendation`); tiebreak by median rank then win count; only-one-backend path; zero-valid-candidates returns a sentinel/null shape; non-default `objective.maximize` reflected in `objective_label`; empty constraints reflected in `slo_label`.
- [ ] HTML report shows a single merged hero with: recommendation line including both backend and candidate names; resolved server parameters (expanded by default for ≤6 params, collapsed otherwise); objective+SLO line; wins, margin, worst-case ratio with workload label.
- [ ] HTML report no longer renders a standalone "Global winner" section anywhere.
- [ ] Dividers display "Decide" and "Explore" with the specified subtitles.
- [ ] When `objective.constraints` is empty/null: hero omits the "subject to..." clause and the constraint-slider section is not rendered.
- [ ] When only one backend has valid measurements: hero recommends within that backend without "vllm wins" / "sglang wins" comparative phrasing.
- [ ] When zero candidates have valid measurements: hero shows the "No valid candidate met the objective" message and no leaderboard renders.
- [ ] Re-running `moe-bench report` on existing run dirs produces a report.html that satisfies the above.

## Blocked by

- Blocked by `issues/002-zero-floor-geomean.md` (the recommendation's global ranking uses the corrected geomean; merging this first would ship a hero pinned to buggy ranking).

## User stories addressed

- User story 1
- User story 2
- User story 3
- User story 4
- User story 5
- User story 12
- User story 13
- User story 14
- User story 20
- User story 22 (partial — recommendation half)
- User story 23
- User story 24
- User story 25
