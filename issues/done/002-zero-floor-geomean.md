## Parent PRD

`issues/prd-report-redesign.md`

## What to build

Fix the global-winner ranking so that a candidate with `relative_to_best = 0` on a workload (the result of SLO violation under `goodput_at_slo`) is multiplicatively penalized in its global score, rather than silently dropped from the geometric mean. See the Problem Statement bullet 2 and Implementation Decisions → "Geomean fix" in the parent PRD.

Concretely: change the geomean helper so that values of `0` (or below the floor) are replaced with a configurable `zero_floor` value before the geomean is computed. Expose `report.robustness.zero_floor` (default `0.05`) in the YAML schema. No other behavior changes.

End-to-end: a sweep where candidate X wins 1 of 6 workloads but violates SLO on the other 5 must rank globally below candidate Y that passes SLO on all 6 with `relative_to_best ≈ 0.8`. Currently X ranks above Y (or near it) because the 5 zeros are dropped.

## Acceptance criteria

- [ ] The geomean helper used by the global-winner ranking floors zeros at the configured `zero_floor` value before computing the geometric mean.
- [ ] `report.robustness.zero_floor` is readable from the loaded config, with default `0.05` when omitted.
- [ ] A unit test pins the case: given synthetic rankings where candidate X has scores `[1.0, 0, 0, 0, 0, 0]` and candidate Y has `[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]`, Y ranks above X globally.
- [ ] A unit test pins the case: given all-positive scores, the geomean result is unchanged from the current behavior (i.e. the floor is a no-op when no zeros are present).
- [ ] A unit test verifies that the zero_floor value is configurable (e.g. `0.5` produces a less severe penalty than `0.05`).
- [ ] Running `moe-bench report` on the existing `results/qwen3-a3b-4gpu-parallelism/` (or any prior run) does not crash and produces a `report.html` whose global ranking reflects the zero-floored geomean.

## Blocked by

None - can start immediately.

## User stories addressed

- User story 18
- User story 19
