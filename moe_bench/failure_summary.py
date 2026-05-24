"""Pure-function failure summarization for the HTML report.

`compute_failure_summary` takes the failure rows (CSV-equivalent dicts where
`valid != "true"`) plus the full candidate and workload context of the
sweep, and returns either `None` (clean run, no callout to render) or a
`FailureSummary` dataclass that the renderer turns into HTML.

No I/O, no HTML, no Plotly: all rendering happens elsewhere so this module
can be unit-tested in isolation.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class FailureSummary:
    total_cells: int
    failed_cells: int
    rate_pct: float
    by_candidate: list[str] = field(default_factory=list)
    by_workload: list[str] = field(default_factory=list)
    statuses: dict[str, int] = field(default_factory=dict)


def _candidate_label(row: dict[str, Any]) -> str:
    """Prefer the human-readable serve_config name; fall back to candidate_id."""
    name = row.get("serve_config") or row.get("candidate_id") or ""
    return str(name)


def _workload_label(row: dict[str, Any]) -> str:
    name = row.get("workload") or row.get("workload_name") or ""
    return str(name)


def _candidate_count(candidates: Iterable[Any]) -> int:
    return sum(1 for _ in candidates)


def _workload_count(workloads: Iterable[Any]) -> int:
    return sum(1 for _ in workloads)


def compute_failure_summary(
    failures: list[dict[str, Any]],
    *,
    candidates: Iterable[Any],
    workloads: Iterable[Any],
) -> FailureSummary | None:
    """Return a FailureSummary for a degraded run, or None for a clean run.

    `failures` is the CSV-equivalent list of measurement rows with
    `valid != "true"`. `candidates` and `workloads` are the totals for the
    sweep (used only to compute `total_cells` and `rate_pct`).
    """
    if not failures:
        return None

    n_candidates = _candidate_count(candidates)
    n_workloads = _workload_count(workloads)
    total_cells = n_candidates * n_workloads
    failed_cells = len(failures)
    rate_pct = (100.0 * failed_cells / total_cells) if total_cells > 0 else 0.0

    by_candidate: list[str] = []
    seen_candidates: set[str] = set()
    by_workload: list[str] = []
    seen_workloads: set[str] = set()
    statuses: dict[str, int] = {}

    for row in failures:
        cand = _candidate_label(row)
        if cand and cand not in seen_candidates:
            seen_candidates.add(cand)
            by_candidate.append(cand)
        wl = _workload_label(row)
        if wl and wl not in seen_workloads:
            seen_workloads.add(wl)
            by_workload.append(wl)
        reason = str(row.get("failure_reason") or "unknown")
        statuses[reason] = statuses.get(reason, 0) + 1

    return FailureSummary(
        total_cells=total_cells,
        failed_cells=failed_cells,
        rate_pct=rate_pct,
        by_candidate=by_candidate,
        by_workload=by_workload,
        statuses=statuses,
    )


# --------------------------------------------------------------------------
# Renderer (HTML wrapper over the pure data above)
# --------------------------------------------------------------------------

# Amber/red palette used for the callout's left border and chip backgrounds.
# Picked to be visually distinct from the existing `section.hero` accent blue.
_CALLOUT_ACCENT = "#b45309"  # amber-700
_CALLOUT_TINT = "#fef3c7"    # amber-100


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def render_failure_callout(summary: FailureSummary | None) -> str:
    """Return the HTML section for the failure callout, or '' for clean runs."""
    if summary is None:
        return ""

    cand_chips = "".join(
        f'<span class="chip" style="background:{_CALLOUT_TINT};border-color:{_CALLOUT_ACCENT}">'
        f'<code>{_esc(name)}</code></span>'
        for name in summary.by_candidate
    ) or '<em class="muted">(none)</em>'

    wl_chips = "".join(
        f'<span class="chip" style="background:{_CALLOUT_TINT};border-color:{_CALLOUT_ACCENT}">'
        f'<code>{_esc(name)}</code></span>'
        for name in summary.by_workload
    ) or '<em class="muted">(none)</em>'

    status_rows = "".join(
        f'<tr><th>{_esc(reason)}</th><td><strong>{count}</strong></td></tr>'
        for reason, count in sorted(summary.statuses.items(), key=lambda kv: (-kv[1], kv[0]))
    ) or '<tr><td colspan="2"><em class="muted">no recorded statuses</em></td></tr>'

    return f"""
    <section class="hero" style="border-left-color:{_CALLOUT_ACCENT}">
      <h2>Sweep had failures</h2>
      <p class="lead">
        <strong>{summary.failed_cells}</strong> of <strong>{summary.total_cells}</strong>
        cells failed (<strong>{summary.rate_pct:.1f}%</strong>).
        Discount the recommendation below by the coverage gap.
      </p>
      <div class="hero-grid">
        <div>
          <h4>Affected candidates ({len(summary.by_candidate)})</h4>
          <div class="chips">{cand_chips}</div>
          <h4>Affected workloads ({len(summary.by_workload)})</h4>
          <div class="chips">{wl_chips}</div>
        </div>
        <div>
          <h4>Failure statuses</h4>
          <table class="kv">{status_rows}</table>
        </div>
      </div>
    </section>
    """
