## Parent PRD

`issues/prd-report-redesign.md`

## What to build

Expand the TTFT/TPOT constraint sliders so dragging them re-ranks the entire above-the-fold Decide block in sync, not just the constraint-leaderboard section. See the Solution paragraph on interactive verification and Implementation Decisions → "Slider scope expansion" in the parent PRD.

The existing JS already re-computes a ranking from embedded measurement JSON when the sliders move. Extend it to also drive:

1. The **Decision summary hero** — recommendation line (potentially a different winner), evidence triple (wins / margin / worst-case-workload), and the "subject to..." clause showing the live slider values.
2. The **Robustness leaderboard** — top-K rows re-sorted and re-highlighted.
3. The existing **constraint-conditional leaderboard** — already driven; verify it stays consistent.

End-to-end: a reader drags the TTFT slider from 2000 to 1000 ms; the hero may now recommend a different candidate; the SLO clause in the hero updates to `p99_ttft ≤ 1000ms, p99_tpot ≤ 100ms`; the leaderboard re-orders; everything is consistent.

## Acceptance criteria

- [ ] Dragging the TTFT slider updates the hero recommendation (candidate name, server params, wins, margin, worst-case) in real time.
- [ ] Dragging the TPOT slider does the same.
- [ ] The "subject to..." SLO clause in the hero reflects the live slider positions, not the YAML-defined values.
- [ ] The robustness leaderboard's top-K rows re-sort and the highlighted winner updates as the sliders move.
- [ ] The existing constraint-conditional leaderboard continues to work and stays consistent with the hero and robustness leaderboard.
- [ ] When sliders are at their initial positions (matching YAML SLOs), all three views show the same ranking as a page reload would produce.
- [ ] When sliders are dragged to a position where no candidate meets SLO, the hero gracefully shows the "No valid candidate met the objective" state (consistent with issue 004's degraded-state handling).
- [ ] No new server-side computation runs at view time; the re-ranking is computed in the browser from the already-embedded measurement JSON.

## Blocked by

- Blocked by `issues/004-decision-hero-and-dividers.md` (the slider needs the hero and robustness leaderboard from 004 to exist before it can drive them; the JS hooks into DOM elements that 004 introduces).

## User stories addressed

- User story 9
- User story 10
- User story 11

## Notes discovered during implementation

**Silent-failure mode in HTML report generation (out of scope; worth tracking):**

The HTML renderer catches some exceptions and writes a `HTML_REPORT_SKIPPED.txt` sentinel file in the result dir instead of crashing. During this issue's work, a Python `%`-formatting collision with a `'%'` literal in the inline JS triggered this path silently — `moe-bench report` returned exit 0 and the existing tests (which don't exercise the actual HTML output) all passed. The bug was only caught by checking whether `report.html` was actually regenerated.

Two concrete follow-ups worth a focused issue:
1. The HTML renderer should NOT swallow exceptions silently. Either let them propagate (the CLI command exit code reflects the failure) or, at minimum, log them to stderr loudly.
2. Add an end-to-end test that asserts `report.html` exists and contains expected hooks after a `generate_html_report` call on a fixture run dir. This is the minimum test that would have caught the silent failure.

**Top-K cap × slider interaction (acceptable but documented):**

When a reader drags the SLO sliders such that a candidate currently below the leaderboard top-K cap (issue 005's behavior) would become the new winner under the live SLO, the JS does NOT promote that candidate across the cap boundary. The visible top-K stays the same set; only the order within each tbody updates.

This is per the issue spec ("re-sort the leaderboard rows in place") and per 005's design intent (the cap is a stable visual contract). A future enhancement could reshuffle rows across the cap boundary, but it would need to coordinate with 005's `<details>` structure to avoid jank during live drag.
