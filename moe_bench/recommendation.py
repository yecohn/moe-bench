"""Pure-function recommendation computation for the HTML report's Decision hero.

`compute_recommendation` takes the ranked CSV-equivalent rows plus the
candidate metadata, the objective name, and the SLO constraints dict, and
returns a `Recommendation` dataclass naming the winner backend + candidate,
the runner-up, the margin, the worst-case workload, and human-readable
labels for objective and SLO.

The actual global score uses the zero-floored geomean from
`html_report.compute_robustness` (issue 002): SLO-violating workloads
multiplicatively penalize the geomean instead of being silently dropped.

No I/O, no HTML, no Plotly: rendering happens in `render_decision_hero`
below, a thin wrapper over the dataclass.
"""

from __future__ import annotations

import html
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Recommendation:
    winner_backend: str | None = None
    winner_candidate_id: str | None = None
    winner_serve_config_name: str | None = None
    winner_params: dict = field(default_factory=dict)
    runner_up_candidate_id: str | None = None
    runner_up_serve_config_name: str | None = None
    wins: int = 0
    total_workloads: int = 0
    margin_pct_over_runner_up: float | None = None
    worst_case_ratio: float | None = None
    worst_case_workload_label: str | None = None
    objective_label: str = ""
    slo_label: str = ""
    is_degraded: bool = False


def _num(value: Any) -> float | None:
    if value in {None, "", "None", "nan", "NaN"}:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _workload_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("workload"), row.get("input_len"), row.get("output_len"), row.get("max_concurrency"))


def _workload_label(row: dict[str, Any]) -> str:
    name = row.get("workload") or row.get("workload_name")
    if name:
        return str(name)
    parts = []
    if row.get("input_len"):
        parts.append(f"in={row['input_len']}")
    if row.get("output_len"):
        parts.append(f"out={row['output_len']}")
    if row.get("max_concurrency"):
        parts.append(f"conc={row['max_concurrency']}")
    return ", ".join(parts) or "workload"


def _curated_params(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only the curated backend-prefixed params, prefix stripped."""
    backend = str(candidate.get("backend", ""))
    prefix = f"{backend}_"
    out: dict[str, Any] = {}
    for key, value in candidate.items():
        if not key.startswith(prefix):
            continue
        if str(value) in {"", "nan", "None"}:
            continue
        out[key.removeprefix(prefix)] = value
    return out


def _format_slo_label(slo: dict[str, float]) -> str:
    """Format the SLO constraints dict as a human-readable label.

    Returns "" when slo is empty (the renderer drops the "subject to..."
    clause entirely). Each constraint is rendered as `<metric> ≤ <value>`,
    appending "ms" for any *_ms metric. Constraint order is the insertion
    order of the dict so it round-trips the YAML.
    """
    if not slo:
        return ""
    parts = []
    for k, v in slo.items():
        # _ms suffix gets a "ms" unit; otherwise no unit.
        unit = "ms" if str(k).endswith("_ms") else ""
        # Render as int when it has no fractional part, else as float.
        if isinstance(v, (int, float)) and float(v).is_integer():
            parts.append(f"{k} ≤ {int(v)}{unit}")
        else:
            parts.append(f"{k} ≤ {v}{unit}")
    return ", ".join(parts)


def compute_recommendation(
    rankings: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    slo: dict[str, float] | None,
    objective: str,
    zero_floor: float = 0.05,
) -> Recommendation:
    """Pick the global winner and runner-up across all workloads.

    Uses the zero-floored geomean from ``compute_robustness`` so SLO
    violations (``relative_to_best == 0``) multiplicatively penalize the
    geometric mean instead of being silently dropped. Tiebreak by median
    rank ascending, then by win count descending.

    Returns a `Recommendation` with `is_degraded=True` when zero candidates
    have any valid ranking rows.
    """
    # Local import keeps this module decoupled from html_report for direct
    # `compute_recommendation` callers, but reuses the canonical scorer.
    from .html_report import compute_robustness

    slo = slo or {}
    objective_label = f"Best by {objective}"
    slo_label = _format_slo_label(slo)

    if not rankings or not candidates:
        return Recommendation(
            objective_label=objective_label,
            slo_label=slo_label,
            is_degraded=True,
        )

    robustness = compute_robustness(rankings, candidates, zero_floor=zero_floor)
    # Filter to candidates that have at least one valid measurement (geomean
    # is None for empty rows; treat those as ineligible).
    eligible = [r for r in robustness if r.get("geomean_relative_to_best") is not None]
    if not eligible:
        return Recommendation(
            objective_label=objective_label,
            slo_label=slo_label,
            is_degraded=True,
        )

    # compute_robustness already sorts by (geomean desc, median_rank asc, wins desc),
    # which is exactly our tiebreak order. The head is the winner.
    winner = eligible[0]
    runner_up = eligible[1] if len(eligible) >= 2 else None

    winner_id = str(winner["candidate_id"])
    candidates_by_id = {str(c.get("candidate_id")): c for c in candidates}
    winner_cand = candidates_by_id.get(winner_id) or {}
    winner_params = _curated_params(winner_cand)

    # Total workloads = distinct workload keys across all rankings.
    total_workloads = len({_workload_key(r) for r in rankings})
    wins = int(winner.get("workload_wins") or 0)

    # Margin: percentage by which winner's geomean exceeds runner-up's.
    margin_pct: float | None = None
    runner_up_id: str | None = None
    runner_up_name: str | None = None
    if runner_up is not None:
        runner_up_id = str(runner_up["candidate_id"])
        runner_up_name = str(runner_up.get("serve_config") or runner_up_id)
        w = winner["geomean_relative_to_best"]
        r = runner_up.get("geomean_relative_to_best")
        if w is not None and r and r > 0:
            margin_pct = 100.0 * (w - r) / r

    # Worst-case workload: the winner's lowest relative_to_best across
    # workloads. Carries the workload label for the hero's evidence line.
    winner_rows = [r for r in rankings if str(r.get("candidate_id")) == winner_id]
    worst_case_ratio: float | None = None
    worst_case_label: str | None = None
    for r in winner_rows:
        rel = _num(r.get("relative_to_best"))
        if rel is None:
            continue
        if worst_case_ratio is None or rel < worst_case_ratio:
            worst_case_ratio = rel
            worst_case_label = _workload_label(r)

    return Recommendation(
        winner_backend=str(winner.get("backend") or ""),
        winner_candidate_id=winner_id,
        winner_serve_config_name=str(winner.get("serve_config") or winner_id),
        winner_params=winner_params,
        runner_up_candidate_id=runner_up_id,
        runner_up_serve_config_name=runner_up_name,
        wins=wins,
        total_workloads=total_workloads,
        margin_pct_over_runner_up=margin_pct,
        worst_case_ratio=worst_case_ratio,
        worst_case_workload_label=worst_case_label,
        objective_label=objective_label,
        slo_label=slo_label,
        is_degraded=False,
    )


# --------------------------------------------------------------------------
# Renderer (HTML wrapper over the pure data above)
# --------------------------------------------------------------------------

_BACKEND_COLORS = {"vllm": "#1f77b4", "sglang": "#ff7f0e"}
_DEFAULT_COLOR = "#0f766e"
# Threshold for the server-params expander: ≤6 params open by default, more collapsed.
_PARAMS_OPEN_THRESHOLD = 6


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _kv_block(params: dict[str, Any]) -> str:
    rows = []
    for k in sorted(params):
        rows.append(
            f'<tr><th>{_esc(k)}</th><td><code>{_esc(params[k])}</code></td></tr>'
        )
    return f'<table class="kv">{"".join(rows)}</table>'


def render_decision_hero(rec: Recommendation) -> str:
    """Render the merged Decision-summary hero. Thin wrapper over `Recommendation`."""
    if rec.is_degraded:
        slo_clause = f" subject to <code>{_esc(rec.slo_label)}</code>" if rec.slo_label else ""
        return f"""
        <section class="hero" style="border-left-color:{_DEFAULT_COLOR}">
          <h2>Decision summary</h2>
          <p class="lead">No valid candidate met the objective.</p>
          <p class="muted">{_esc(rec.objective_label)}{slo_clause}.</p>
        </section>
        """

    color = _BACKEND_COLORS.get(rec.winner_backend or "", _DEFAULT_COLOR)

    # Recommendation line: backend + candidate name, both highlighted.
    # The whole "winner" phrase is wrapped in [data-rec-winner] so the slider JS
    # can swap it out without re-rendering the rest of the hero.
    rec_line = (
        f"Deploy <span data-rec-winner>"
        f"<strong style='color:{color}'>{_esc(rec.winner_backend)}</strong> "
        f"with config <strong>{_esc(rec.winner_serve_config_name)}</strong>"
        f"</span>."
    )

    # Objective + SLO line.
    # The SLO label is wrapped in [data-rec-slo-label] so the slider JS can
    # update it live with the dragged values (instead of the YAML-baked ones).
    if rec.slo_label:
        obj_line = (
            f"{_esc(rec.objective_label)} subject to "
            f"<code data-rec-slo-label>{_esc(rec.slo_label)}</code>."
        )
    else:
        # Still emit the (empty) hook so JS can populate it if the user enables SLOs
        # via sliders on a no-SLO run; harmless on degraded paths.
        obj_line = (
            f"{_esc(rec.objective_label)}"
            f"<span data-rec-slo-label hidden></span>."
        )

    # Evidence line: wins / margin / worst-case. Each evidence number gets its
    # own hook so JS can update individual pieces.
    evidence_parts = [
        f"<strong>Wins</strong> "
        f"<span data-rec-evidence-wins>{rec.wins} of {rec.total_workloads}</span>"
        f" workloads"
    ]
    if rec.margin_pct_over_runner_up is not None and rec.runner_up_serve_config_name:
        evidence_parts.append(
            f"<strong>Margin</strong> "
            f"<span data-rec-evidence-margin>{rec.margin_pct_over_runner_up:+.1f}%</span>"
            f" over <code>{_esc(rec.runner_up_serve_config_name)}</code>"
        )
    else:
        # Empty hook present so JS can populate after a slider drag introduces
        # a runner-up where the static page had none.
        evidence_parts.append(
            f"<span data-rec-evidence-margin hidden></span>"
        )
    if rec.worst_case_ratio is not None and rec.worst_case_workload_label:
        evidence_parts.append(
            f"<strong>Worst case</strong> "
            f"<span data-rec-evidence-worst>"
            f"{rec.worst_case_ratio:.2f}× on <code>{_esc(rec.worst_case_workload_label)}</code>"
            f"</span>"
        )
    else:
        evidence_parts.append(
            f"<span data-rec-evidence-worst hidden></span>"
        )
    evidence_line = " &middot; ".join(evidence_parts)

    # Server params block: expanded by default for ≤6 params, collapsed otherwise.
    params_open = " open" if len(rec.winner_params) <= _PARAMS_OPEN_THRESHOLD else ""
    params_block = _kv_block(rec.winner_params) if rec.winner_params else "<p class='muted'><em>(no curated params)</em></p>"

    return f"""
    <section class="hero" style="border-left-color:{color}" data-decision-hero>
      <h2>Decision summary</h2>
      <p class="lead">{rec_line}</p>
      <p>{obj_line}</p>
      <p>{evidence_line}.</p>
      <details{params_open}>
        <summary>Server parameters ({len(rec.winner_params)})</summary>
        {params_block}
      </details>
    </section>
    """
