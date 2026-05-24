## Parent PRD

`issues/prd-report-redesign.md`

## What to build

Cap the two long sections of the report so big sweeps render compactly while preserving the ability to see everything on demand. See Implementation Decisions → "Robustness leaderboard cap" and "Candidate cards cap" in the parent PRD.

Two caps, same mechanism:

1. **Robustness leaderboard** (above the fold) — render top-K rows by default with a `<details>` expander revealing the rest. K from `report.leaderboard_rows` in YAML, default `5`.
2. **Candidate detail cards** (below the fold) — render top-K cards by robustness rank with a `<details>` expander for the rest. K from `report.top_k_candidate_cards` in YAML, default `8`.

End-to-end: a 24-candidate sweep renders with 5 leaderboard rows visible by default and 8 candidate cards visible by default, each with an inline "show all" expander; a 9-candidate sweep renders normally (the expander on the leaderboard reveals 4 more rows; the cards section is unchanged because 9 < 8 means top-8 + 1 expander row).

## Acceptance criteria

- [ ] `report.leaderboard_rows` is readable from the loaded config, with default `5` when omitted.
- [ ] `report.top_k_candidate_cards` is readable from the loaded config, with default `8` when omitted.
- [ ] Robustness leaderboard renders the top-K rows visible by default, ordered by global score with winner highlighted.
- [ ] A `<details>` element below the leaderboard table reveals rows K+1 through end when expanded.
- [ ] No expander renders when total candidate count ≤ K (no empty "show 0 more" element).
- [ ] Candidate detail cards section renders the top-K cards (by robustness rank) by default.
- [ ] A `<details>` element below the visible cards reveals cards K+1 through end when expanded.
- [ ] No expander renders when total candidate count ≤ K for the cards section.
- [ ] Re-running `moe-bench report` on `results/qwen3-a3b-4gpu-parallelism/` (9 candidates) produces a compact above-the-fold and a candidate-cards section without an expander.
- [ ] Re-running on a hypothetical 24-candidate run (or `results/qwen3-a3b-4gpu-grid/` partial) produces both expanders, each revealing the remainder.

## Blocked by

- Blocked by `issues/004-decision-hero-and-dividers.md` (the leaderboard cap modifies the same Decide-block surface that 004 establishes; landing 005 first would mean implementing a cap on a section that's about to be restructured).

## User stories addressed

- User story 15
- User story 16
- User story 17
