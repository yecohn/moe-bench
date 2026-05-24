## Parent PRD

`issues/prd-report-redesign.md`

## What to build

Surface sweep failures prominently in the HTML report when any cells failed, and hide the section entirely on clean runs. See the Problem Statement bullet 3 and Implementation Decisions → "Conditional failure callout section" in the parent PRD.

Two parts:

1. A new pure function `compute_failure_summary(failures, candidates, workloads)` that returns `None` for a clean run, else a structured `FailureSummary` object with `total_cells`, `failed_cells`, `rate_pct`, `by_candidate` (list of candidates with at least one failure), `by_workload` (list of workloads with at least one failure), and `statuses` (dict mapping each failure status like `server_ready_timeout` / `oom` / etc. to its count).
2. A conditional HTML section that renders only when `compute_failure_summary` returns non-`None`, positioned above the Decision summary hero. Lists the affected candidates and workloads with their status. Clean runs produce no failure section.

End-to-end: a run with failures opens to a visible callout listing what broke; a clean run shows no extra content above the recommendation.

## Acceptance criteria

- [ ] `compute_failure_summary` exists as a pure function (no HTML, no I/O, no Plotly) taking the failures CSV-equivalent list plus candidates + workloads context.
- [ ] Returns `None` when the failures list is empty.
- [ ] Returns a `FailureSummary` object when failures exist, with all fields populated (`total_cells`, `failed_cells`, `rate_pct`, `by_candidate`, `by_workload`, `statuses`).
- [ ] Unit tests cover: clean run returns `None`; one candidate all failed; one workload all candidates failed on it; mixed-status failures (counts in `statuses` correct); partial failures (some valid, some invalid).
- [ ] The HTML report renders the callout section above the Decision summary when failures exist.
- [ ] The HTML report renders no failure section (no empty panel, no placeholder) when the run is clean.
- [ ] The callout lists affected candidates by name and affected workloads by label, plus the failure-rate percentage and per-status counts.
- [ ] Re-running `moe-bench report` on `results/qwen3-a3b-4gpu-parallelism/` (which has failed candidates) produces an HTML report with the callout visible.
- [ ] Re-running `moe-bench report` on `results/stack-smoke-2026-05-22T114430Z/` (a clean run) produces an HTML report with no failure section.

## Blocked by

None - can start immediately. (Independent of 002; different files and concerns.)

## User stories addressed

- User story 6
- User story 7
- User story 8
- User story 21
- User story 22 (partial — failure-summary half)

## Notes discovered during implementation

**Upstream gap in `normalize.normalize_run` (out of scope for this issue):**

This issue ships a working `compute_failure_summary` + renderer. The pure function and HTML wrapper behave correctly for whatever `failures.csv` contains — verified with a synthetic failures.csv where the callout renders as designed.

However, on `results/qwen3-a3b-4gpu-parallelism/` (which has 9 SGLang candidates that all hit `server_ready_timeout`), the callout does NOT render. Reason: `normalize.normalize_run` iterates `raw/<backend>/<serve>/<workload>/result.json` files to build `failures.csv`. When a server never comes up, no per-workload subdirectories exist (only a candidate-level `status.json`), so those failures never become rows in `failures.csv`.

The most common failure mode (server startup crashes — exactly the mode that triggered the report-redesign work) is invisible to the callout under the current normalize pipeline. The callout shipped here is correct; the data feeding it is incomplete.

**Suggested follow-up issue:** backfill server-startup failures into `failures.csv`. Two paths:
1. `normalize.normalize_run` also scans candidate-level `status.json` files (where `status != "ready"`) and synthesizes per-workload failure rows by cross-referencing the workload grid.
2. The runner writes per-workload `status.json` stubs (status=`server_ready_timeout`, no `result.json`) when the server check fails, so the workload subdirs exist and normalize picks them up automatically.

Path 2 is closer to the existing model.
