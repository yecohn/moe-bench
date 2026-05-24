"""Interactive single-file HTML report.

The Markdown report is the auditable artifact for PRs and Slack. The HTML
report is the exploration tool the user actually opens to answer:

  - which candidate should I deploy globally?
  - which is best subject to my latency SLO?
  - what does the throughput / latency tradeoff look like across backends?

Everything is rendered into one self-contained `report.html` next to
`report.md`. Plotly JS is inlined once; data is embedded as JSON so a small
amount of vanilla JS can drive the constraint-conditional leaderboard
without a server.
"""
from __future__ import annotations

import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .failure_summary import compute_failure_summary, render_failure_callout
from .recommendation import compute_recommendation, render_decision_hero

BACKEND_COLORS = {"vllm": "#1f77b4", "sglang": "#ff7f0e"}
DEFAULT_COLOR = "#888888"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import csv
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _is_valid(row: dict[str, Any]) -> bool:
    return str(row.get("valid", "")).lower() in {"true", "1"}


def _workload_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("workload"), row.get("input_len"), row.get("output_len"), row.get("max_concurrency"))


def _workload_label(row: dict[str, Any]) -> str:
    parts = []
    if row.get("input_len"):
        parts.append(f"in={row['input_len']}")
    if row.get("output_len"):
        parts.append(f"out={row['output_len']}")
    if row.get("max_concurrency"):
        parts.append(f"conc={row['max_concurrency']}")
    return ", ".join(parts) or str(row.get("workload") or "workload")


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


def _geomean(values: list[float], zero_floor: float = 0.05) -> float | None:
    """Geometric mean with zero values floored to ``zero_floor``.

    Under the default ``goodput_at_slo`` objective, an SLO-violating workload
    produces ``relative_to_best = 0``. Filtering those zeros out (the previous
    behavior) silently drops the bad workload, making a one-shot lucky candidate
    look more robust than it is. Instead we floor each non-positive value at
    ``zero_floor`` (default 0.05, ~ "the worst non-zero score a candidate can
    achieve") so SLO violations multiplicatively penalize the geomean.

    ``None`` inputs are still dropped (missing measurement, not a violation).
    Returns ``None`` only when no values remain after dropping ``None``s.
    """
    floored: list[float] = []
    for v in values:
        if v is None:
            continue
        floored.append(v if v > zero_floor else zero_floor)
    if not floored:
        return None
    return math.exp(sum(math.log(v) for v in floored) / len(floored))


# --------------------------------------------------------------------------
# Robustness metrics
# --------------------------------------------------------------------------

def compute_robustness(
    rankings: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    zero_floor: float = 0.05,
) -> list[dict[str, Any]]:
    """One row per candidate with cross-workload summary metrics.

    ``zero_floor`` (default 0.05) is the value that ``_geomean`` substitutes
    for ``relative_to_best`` scores at or below the floor — typically the
    result of an SLO violation under ``goodput_at_slo``. Configurable via
    ``report.robustness.zero_floor`` in the YAML.

    Metrics:
      - geomean_relative_to_best: how close to the workload winner, on average.
        1.0 means always the winner. Geometric mean penalizes large gaps more
        symmetrically than arithmetic mean.
      - workload_wins: count of workloads where rank=1.
      - workloads_evaluated: count of workloads the candidate has any valid measurement on.
      - worst_relative_to_best: min relative_to_best across workloads (tail behavior).
      - median_rank: median of per-workload ranks; low = robustly near the top.
      - pareto_appearances: count of workloads where the candidate is on the
        throughput/p99-TTFT Pareto frontier (per workload, per backend not needed).
    """
    by_workload: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rankings:
        by_workload[_workload_key(r)].append(r)

    pareto_membership: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for wkey, rows in by_workload.items():
        points = []
        for r in rows:
            cid = r.get("candidate_id")
            tput = _num(r.get("median_output_tok_s_per_gpu") or r.get("objective_value"))
            ttft = _num(r.get("median_p99_ttft_ms"))
            if cid and tput is not None and ttft is not None:
                points.append((cid, tput, ttft))
        points.sort(key=lambda p: p[2])  # by TTFT ascending
        best_tput = -math.inf
        for cid, tput, _ttft in points:
            if tput > best_tput:
                pareto_membership[str(cid)].add(wkey)
                best_tput = tput

    by_cand: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rankings:
        cid = str(r.get("candidate_id") or "")
        if cid:
            by_cand[cid].append(r)

    out: list[dict[str, Any]] = []
    for cand in candidates:
        cid = str(cand.get("candidate_id") or "")
        rows = by_cand.get(cid, [])
        relatives = [_num(r.get("relative_to_best")) for r in rows]
        relatives = [v for v in relatives if v is not None]
        ranks = [_num(r.get("rank")) for r in rows]
        ranks = [v for v in ranks if v is not None]
        wins = sum(1 for v in ranks if int(v) == 1)
        out.append({
            "candidate_id": cid,
            "backend": cand.get("backend"),
            "serve_config": cand.get("serve_config"),
            "geomean_relative_to_best": _geomean(relatives, zero_floor=zero_floor),
            "worst_relative_to_best": min(relatives) if relatives else None,
            "median_rank": statistics.median(ranks) if ranks else None,
            "workload_wins": wins,
            "workloads_evaluated": len(rows),
            "pareto_appearances": len(pareto_membership.get(cid, set())),
        })
    # Sort by geomean desc, then by median_rank asc, then by wins desc.
    out.sort(key=lambda r: (
        -(r["geomean_relative_to_best"] or 0),
        r["median_rank"] if r["median_rank"] is not None else 1e9,
        -r["workload_wins"],
    ))
    return out


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: Any, digits: int = 2) -> str:
    n = _num(value)
    if n is None:
        return "—"
    if abs(n) >= 100:
        return f"{n:,.0f}"
    if abs(n) >= 10:
        return f"{n:,.1f}"
    return f"{n:,.{digits}f}"


def _anchor(cid: str) -> str:
    return f"cand-{_esc(cid)}"


def _kv_block(params: dict[str, Any]) -> str:
    rows = []
    for k in sorted(params):
        rows.append(
            f'<tr><th>{_esc(k)}</th><td><code>{_esc(params[k])}</code></td></tr>'
        )
    return f'<table class="kv">{"".join(rows)}</table>'


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _section_header(manifest: dict[str, Any], measurements: list[dict[str, Any]], candidates: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    backends = manifest.get("backends") or {}
    backend_chips = []
    for name, info in backends.items():
        version = (info or {}).get("version") or "unknown"
        color = BACKEND_COLORS.get(name, DEFAULT_COLOR)
        backend_chips.append(
            f'<span class="chip" style="background:{color}1a;border-color:{color}">'
            f'<strong>{_esc(name)}</strong> {_esc(version)}</span>'
        )
    workloads = {_workload_key(r) for r in measurements}
    valid = sum(1 for r in measurements if _is_valid(r))
    return f"""
    <header>
      <h1>{_esc(manifest.get("run_id") or "moe-bench run")}</h1>
      <div class="meta">
        <span>Model: <code>{_esc(manifest.get("model") or "—")}</code></span>
        <span>GPUs: <code>{_esc(manifest.get("gpus") or "—")}</code></span>
        <span>Started: <code>{_esc(manifest.get("started_at") or "—")}</code></span>
      </div>
      <div class="chips">{"".join(backend_chips) or "<em>no backend metadata</em>"}</div>
      <div class="counts">
        <span><strong>{len(candidates)}</strong> candidates</span>
        <span><strong>{len(workloads)}</strong> workloads</span>
        <span><strong>{valid}</strong>/{len(measurements)} valid measurements</span>
        <span><strong>{len(failures)}</strong> failures</span>
      </div>
    </header>
    """


def _section_robustness_leaderboard(robustness: list[dict[str, Any]], leaderboard_rows: int = 5) -> str:
    """Render the leaderboard with a top-K visible cap.

    The first ``leaderboard_rows`` rows are visible by default. Any remaining
    rows are wrapped in a ``<details>`` expander so big sweeps stay compact
    above the fold. When the total candidate count is at or below the cap,
    no expander is rendered (no empty "show 0 more" element).
    """
    def _row_html(idx: int, row: dict[str, Any]) -> str:
        cid = str(row["candidate_id"])
        # Issue 006: stable per-row hook so the slider JS can re-sort rows in
        # place and re-highlight the new winner. The first row gets the
        # winner-highlight class by default; JS swaps it as sliders move.
        cls = ' class="winner-row"' if idx == 1 else ""
        return (
            f'<tr data-candidate-id="{_esc(cid)}" data-rank-rank="{idx}"{cls}>'
            f"<td data-col=\"rank\">{idx}</td>"
            f'<td><a href="#{_anchor(cid)}"><code>{_esc(cid)}</code></a></td>'
            f"<td>{_esc(row.get('backend'))}</td>"
            f"<td>{_esc(row.get('serve_config'))}</td>"
            f"<td data-col=\"geomean\">{_fmt(row['geomean_relative_to_best'], 3)}</td>"
            f"<td>{_fmt(row['worst_relative_to_best'], 3)}</td>"
            f"<td>{_fmt(row['median_rank'], 1)}</td>"
            f"<td data-col=\"wins\">{row['workload_wins']}/{row['workloads_evaluated']}</td>"
            f"<td>{row['pareto_appearances']}</td>"
            "</tr>"
        )

    total = len(robustness)
    visible_rows = [_row_html(i + 1, row) for i, row in enumerate(robustness[:leaderboard_rows])]
    hidden_rows = [
        _row_html(i + 1, row)
        for i, row in enumerate(robustness[leaderboard_rows:], start=leaderboard_rows)
    ]

    header_html = (
        "<tr>"
        "<th>#</th><th>candidate</th><th>backend</th><th>serve_config</th>"
        '<th title="Geometric mean of relative_to_best across workloads">geomean rel-to-best</th>'
        '<th title="Worst per-workload relative_to_best">worst rel-to-best</th>'
        '<th title="Median per-workload rank">median rank</th>'
        '<th title="Workloads won / workloads evaluated">wins</th>'
        '<th title="Workloads where this candidate is on the throughput/TTFT Pareto frontier">Pareto</th>'
        "</tr>"
    )

    details_block = ""
    if hidden_rows:
        details_block = (
            f'<details><summary>Show all {total} candidates</summary>'
            f'<table class="data" id="robustness-table-overflow">'
            f'<thead>{header_html}</thead>'
            f'<tbody id="robustness-body-overflow">{"".join(hidden_rows)}</tbody>'
            '</table></details>'
        )

    return f"""
    <section>
      <h2>Robustness leaderboard</h2>
      <p>Ranked by geometric mean of <code>relative_to_best</code> across all workloads.
         Click a <code>candidate_id</code> to drill into its detail card below.</p>
      <table class="data" id="robustness-table">
        <thead>
          {header_html}
        </thead>
        <tbody id="robustness-body">{"".join(visible_rows)}</tbody>
      </table>
      {details_block}
    </section>
    """


def _section_constraint_leaderboard(rankings: list[dict[str, Any]], workloads: list[tuple[Any, ...]]) -> str:
    """Static HTML + embedded JSON; vanilla JS does the live filtering."""
    workload_options = ['<option value="__all__">All workloads (aggregate)</option>']
    for wkey in workloads:
        label = ", ".join(str(x) for x in wkey if x not in {None, ""})
        workload_options.append(f'<option value="{_esc(json.dumps(list(wkey)))}">{_esc(label)}</option>')
    return f"""
    <section>
      <h2>Constraint-conditional leaderboard</h2>
      <p>Filter by latency SLO and workload to find the best config under your constraints.
         Throughput is shown as median <code>output_tok_s_per_gpu</code>.</p>
      <div class="controls">
        <label>Workload <select id="cc-workload">{"".join(workload_options)}</select></label>
        <label>Backend
          <span class="chip-row">
            <label><input type="checkbox" class="cc-backend" value="vllm" checked> vllm</label>
            <label><input type="checkbox" class="cc-backend" value="sglang" checked> sglang</label>
          </span>
        </label>
        <label>Max p99 TTFT (ms)
          <input type="range" id="cc-ttft" min="0" max="10000" step="10" value="10000">
          <output id="cc-ttft-out">∞</output>
        </label>
        <label>Max p99 TPOT (ms)
          <input type="range" id="cc-tpot" min="0" max="1000" step="1" value="1000">
          <output id="cc-tpot-out">∞</output>
        </label>
      </div>
      <table class="data" id="cc-table">
        <thead>
          <tr>
            <th>#</th><th>candidate</th><th>backend</th><th>serve_config</th>
            <th>workload</th><th>throughput</th><th>p99 TTFT (ms)</th><th>p99 TPOT (ms)</th>
          </tr>
        </thead>
        <tbody id="cc-body"></tbody>
      </table>
    </section>
    """


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-x, higher-y Pareto frontier. Returns sorted frontier points."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    best_y = -math.inf
    for x, y in sorted_pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def _best_config_per_backend(robustness: list[dict[str, Any]]) -> dict[str, str]:
    """Pick the canonical 'best config' per backend for the scientific plots.

    Uses geomean of relative_to_best across workloads (the same ranking the
    global-winner section uses). Returns {backend: candidate_id}.
    """
    out: dict[str, str] = {}
    for row in robustness:
        backend = str(row.get("backend") or "")
        if not backend or backend in out:
            continue
        out[backend] = str(row.get("candidate_id") or "")
    return out


def _median_by_cell(measurements: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, tuple[Any, ...]], float]:
    """Median of `metric` across repeats per (backend, candidate, workload_key)."""
    groups: dict[tuple[str, str, tuple[Any, ...]], list[float]] = defaultdict(list)
    for row in measurements:
        if not _is_valid(row):
            continue
        v = _num(row.get(metric))
        if v is None:
            continue
        key = (str(row.get("backend") or ""), str(row.get("candidate_id") or ""), _workload_key(row))
        groups[key].append(v)
    out: dict[tuple[str, str, tuple[Any, ...]], float] = {}
    for key, vals in groups.items():
        vals.sort()
        out[key] = vals[len(vals) // 2]
    return out


def _aggregate_for_pareto(measurements: list[dict[str, Any]], x_metric: str) -> dict[str, list[dict[str, Any]]]:
    """Median across repeats per (backend, candidate, workload). Pareto needs one point per cell, not per repeat."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in measurements:
        if not _is_valid(row):
            continue
        x = _num(row.get(x_metric))
        y = _num(row.get("output_tok_s_per_gpu"))
        if x is None or y is None:
            continue
        key = (str(row.get("backend") or ""), str(row.get("candidate_id") or ""), _workload_key(row))
        groups[key].append({"x": x, "y": y, "row": row})
    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (backend, cid, wkey), points in groups.items():
        xs = sorted(p["x"] for p in points)
        ys = sorted(p["y"] for p in points)
        med_x = xs[len(xs) // 2]
        med_y = ys[len(ys) // 2]
        by_backend[backend].append({
            "x": med_x, "y": med_y,
            "candidate_id": cid,
            "row": points[0]["row"],
            "repeats": len(points),
        })
    return by_backend


def _hover_card(serve_config: str, backend: str, workload: str, x_label: str, x: float, y: float, repeats: int, params: dict[str, Any]) -> str:
    """One structured hover block per point. Tight key=value rows; user can scan vertically."""
    param_rows = "<br>".join(f"&nbsp;&nbsp;<b>{_esc(k)}</b>: {_esc(v)}" for k, v in sorted(params.items()))
    return (
        f"<b>{_esc(serve_config)}</b><br>"
        f"<span style='color:#888'>{_esc(backend)} &middot; {_esc(workload)} &middot; n={repeats}</span><br>"
        f"<b>throughput</b>: {y:.1f} tok/s/GPU<br>"
        f"<b>{x_label}</b>: {x:.1f} ms<br>"
        f"<br>{param_rows}"
    )


def _figure_pareto(measurements: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]], x_metric: str, title: str, x_label: str) -> Any:
    import plotly.graph_objects as go
    fig = go.Figure()
    by_backend = _aggregate_for_pareto(measurements, x_metric)

    # Per backend: split points into non-frontier (circles) and frontier (stars on dotted line).
    # Plotly's legend can only display one marker symbol per trace, so we use two traces per
    # backend so each legend entry visually matches what's on the chart.
    annotations = []
    for backend, points in sorted(by_backend.items()):
        color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        cids = [p["candidate_id"] for p in points]
        texts = []
        for p in points:
            cand = candidates_by_id.get(p["candidate_id"], {})
            params = _curated_params(cand)
            texts.append(_hover_card(
                cand.get("serve_config") or p["candidate_id"],
                backend,
                _workload_label(p["row"]),
                x_label,
                p["x"], p["y"], p["repeats"],
                params,
            ))
        frontier_set = set(_pareto_frontier(list(zip(xs, ys))))
        nf_idx = [i for i in range(len(points)) if (xs[i], ys[i]) not in frontier_set]
        fr_idx = sorted((i for i in range(len(points)) if (xs[i], ys[i]) in frontier_set), key=lambda i: xs[i])

        if nf_idx:
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in nf_idx],
                y=[ys[i] for i in nf_idx],
                mode="markers", name=backend, legendgroup=backend,
                marker=dict(size=9, color=color, opacity=0.7, symbol="circle", line=dict(width=1, color="white")),
                text=[texts[i] for i in nf_idx], hoverinfo="text",
                customdata=[cids[i] for i in nf_idx],
            ))
        if fr_idx:
            fig.add_trace(go.Scatter(
                x=[xs[i] for i in fr_idx],
                y=[ys[i] for i in fr_idx],
                mode="lines+markers" if len(fr_idx) >= 2 else "markers",
                name=f"{backend} Pareto frontier ★", legendgroup=backend,
                marker=dict(size=14, color=color, opacity=0.95, symbol="star", line=dict(width=1, color="white")),
                line=dict(color=color, width=2, dash="dot"),
                text=[texts[i] for i in fr_idx], hoverinfo="text",
                customdata=[cids[i] for i in fr_idx],
            ))
        # Annotate top throughput per backend (most decision-relevant single point).
        # axref/ayref must be "pixel" explicitly — without that, Plotly interprets
        # ax/ay in data coords on log axes and blows up the autorange.
        if points:
            best = max(points, key=lambda p: p["y"])
            cand = candidates_by_id.get(best["candidate_id"], {})
            label = cand.get("serve_config") or best["candidate_id"]
            annotations.append(dict(
                xref="x", yref="y", axref="pixel", ayref="pixel",
                x=best["x"], y=best["y"],
                text=f"<b>{_esc(label)}</b>",
                showarrow=True, arrowhead=0, arrowcolor=color, ax=30, ay=-30,
                font=dict(size=11, color=color),
                bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, borderpad=3,
            ))

    # Compute explicit log-space x-range from the actual data so annotations can't
    # blow up the autoranger. Plotly's autorange on log axes with pixel-offset
    # annotations is fragile; explicit range guarantees a tight, readable chart.
    all_xs = [p["x"] for pts in by_backend.values() for p in pts if p["x"] is not None and p["x"] > 0]
    all_ys = [p["y"] for pts in by_backend.values() for p in pts if p["y"] is not None and p["y"] > 0]
    xaxis_kwargs: dict[str, Any] = {"title": f"{x_label} (ms, log scale, lower is better)", "type": "log"}
    if all_xs:
        lo, hi = min(all_xs), max(all_xs)
        xaxis_kwargs["range"] = [math.log10(lo) - 0.3, math.log10(hi) + 0.5]
    yaxis_kwargs: dict[str, Any] = {"title": "output tok/s/GPU (higher is better)"}
    if all_ys:
        lo, hi = min(all_ys), max(all_ys)
        pad = max(5.0, (hi - lo) * 0.10)
        yaxis_kwargs["range"] = [max(0, lo - pad), hi + pad]

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        xaxis=xaxis_kwargs,
        yaxis=yaxis_kwargs,
        hovermode="closest",
        template="plotly_white",
        height=540,
        legend=dict(orientation="h", y=-0.18),
        annotations=annotations,
        margin=dict(l=60, r=40, t=60, b=80),
    )
    return fig


def _figure_parcoords(rankings: list[dict[str, Any]], candidates: list[dict[str, Any]], backend: str) -> Any | None:
    """Parallel coordinates over the varying curated params + objective, per backend.

    Dimensions are ordered by impact (|Pearson r| with objective) so the eye reads the
    high-leverage axes first and lands on the objective axis on the right.
    """
    import plotly.graph_objects as go
    backend_cands = [c for c in candidates if str(c.get("backend")) == backend]
    if len(backend_cands) < 2:
        return None
    prefix = f"{backend}_"
    param_keys: list[str] = []
    for key in sorted({k for c in backend_cands for k in c if k.startswith(prefix)}):
        values = {str(c.get(key)) for c in backend_cands if str(c.get(key, "")) not in {"", "nan", "None"}}
        if len(values) > 1:
            param_keys.append(key)
    if not param_keys:
        return None

    objective_by_cand: dict[str, float] = {}
    rank_by_cand: dict[str, float] = {}
    for r in rankings:
        cid = str(r.get("candidate_id") or "")
        obj = _num(r.get("median_output_tok_s_per_gpu") or r.get("objective_value"))
        rank = _num(r.get("rank"))
        if obj is None:
            continue
        objective_by_cand.setdefault(cid, 0)
        objective_by_cand[cid] = max(objective_by_cand[cid], obj)
        if rank is not None:
            rank_by_cand[cid] = min(rank_by_cand.get(cid, 1e9), rank)

    # Build per-candidate (numeric or categorical-index) column for each param.
    cand_ids = [str(c.get("candidate_id") or "") for c in backend_cands]
    objective_vals = [objective_by_cand.get(cid, 0) for cid in cand_ids]

    def _column_for(key: str) -> tuple[list[float], dict[str, Any]]:
        raw = [c.get(key) for c in backend_cands]
        is_numeric = True
        numeric: list[float] = []
        for v in raw:
            num = _num(v)
            if num is None and v not in {None, "", "nan", "None"}:
                is_numeric = False
                break
            numeric.append(num if num is not None else 0.0)
        label = key.removeprefix(prefix)
        if is_numeric and any(_num(x) is not None for x in raw):
            return numeric, {"label": label, "values": numeric}
        categories = sorted({str(v) for v in raw if str(v) not in {"", "nan", "None"}})
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        vals = [float(cat_to_idx.get(str(v), 0)) for v in raw]
        return vals, {
            "label": label,
            "values": vals,
            "tickvals": list(range(len(categories))),
            "ticktext": categories,
        }

    columns: list[tuple[str, list[float], dict[str, Any]]] = []
    for key in param_keys:
        vals, dim = _column_for(key)
        columns.append((key, vals, dim))

    # Rank each axis by |Pearson r| with the objective; ties broken by variance.
    def _corr(xs: list[float], ys: list[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denom = (sxx * syy) ** 0.5
        return abs(sxy / denom) if denom else 0.0

    def _impact(xs: list[float]) -> float:
        if not xs:
            return 0.0
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / n
        return _corr(xs, objective_vals) * (1.0 + math.log1p(var))

    columns.sort(key=lambda c: -_impact(c[1]))
    dimensions = [c[2] for c in columns]
    # Objective is always the rightmost axis (parcoords convention).
    dimensions.append({"label": "tok/s/GPU (max)", "values": objective_vals})

    color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)
    fig = go.Figure(go.Parcoords(
        line=dict(
            color=objective_vals,
            colorscale=[[0, "#e5e7eb"], [0.5, "#9ca3af"], [1, color]],
            showscale=True,
            colorbar=dict(title=dict(text="tok/s/GPU"), thickness=14),
            cmin=min(objective_vals) if objective_vals else 0,
            cmax=max(objective_vals) if objective_vals else 1,
        ),
        dimensions=dimensions,
        labelangle=-15,
        labelside="bottom",
    ))
    best_cid = max(rank_by_cand, key=lambda c: -rank_by_cand[c]) if rank_by_cand else None
    subtitle = ""
    if best_cid:
        cand = next((c for c in backend_cands if str(c.get("candidate_id")) == best_cid), None)
        if cand:
            subtitle = f"  &nbsp;<span style='color:#888;font-weight:normal'>best per-workload: <b>{_esc(cand.get('serve_config') or best_cid)}</b></span>"
    fig.update_layout(
        title=dict(text=f"<b>{backend}</b>: parameter space (axes ordered by |corr| with throughput; objective on right){subtitle}", x=0.0, xanchor="left", font=dict(size=14)),
        template="plotly_white",
        height=480,
        margin=dict(l=80, r=80, t=70, b=50),
    )
    return fig


def _figure_backend_ratio(measurements: list[dict[str, Any]]) -> Any | None:
    """Heatmap of best_sglang / best_vllm throughput per (input_len, max_concurrency) per output_len.

    Log color scale (ratios are multiplicative) symmetric around 1.0. Missing cells
    are drawn with an explicit '—' rather than blank, so absent measurements don't
    look like dead-zero ratios.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    by_backend: dict[str, dict[tuple[Any, Any, Any], float]] = defaultdict(dict)
    for row in measurements:
        if not _is_valid(row):
            continue
        tput = _num(row.get("output_tok_s_per_gpu"))
        if tput is None:
            continue
        key = (row.get("input_len"), row.get("output_len"), row.get("max_concurrency"))
        backend = str(row.get("backend") or "")
        cur = by_backend[backend].get(key, -math.inf)
        if tput > cur:
            by_backend[backend][key] = tput
    if "vllm" not in by_backend or "sglang" not in by_backend:
        return None

    keys = set(by_backend["vllm"]) | set(by_backend["sglang"])  # take union so missing cells show
    output_lens = sorted({k[1] for k in keys}, key=lambda v: (v is None, v))
    if not output_lens:
        return None

    # Symmetric log range so 2× and 0.5× are visually equivalent distance from 1.0.
    log_ratios = []
    for k in keys:
        s = by_backend["sglang"].get(k)
        v = by_backend["vllm"].get(k)
        if s is None or v is None or v <= 0:
            continue
        log_ratios.append(math.log(s / v))
    if log_ratios:
        zmax = max(0.05, max(abs(min(log_ratios)), abs(max(log_ratios))))
    else:
        zmax = 0.5

    fig = make_subplots(
        rows=1, cols=len(output_lens),
        subplot_titles=[f"<b>output_len = {o}</b>" for o in output_lens],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    for i, out_len in enumerate(output_lens, start=1):
        sub_keys = [k for k in keys if k[1] == out_len]
        input_lens = sorted({k[0] for k in sub_keys}, key=lambda v: (v is None, v))
        concs = sorted({k[2] for k in sub_keys}, key=lambda v: (v is None, v))
        z, text, hover = [], [], []
        for c in concs:
            z_row, t_row, h_row = [], [], []
            for il in input_lens:
                key = (il, out_len, c)
                s = by_backend["sglang"].get(key)
                v = by_backend["vllm"].get(key)
                if s is not None and v is not None and v > 0:
                    ratio = s / v
                    z_row.append(math.log(ratio))
                    t_row.append(f"<b>{ratio:.2f}×</b>")
                    h_row.append(
                        f"input_len={il}  max_concurrency={c}<br>"
                        f"sglang best: {s:.1f}<br>vllm best: {v:.1f}<br>"
                        f"<b>ratio: {ratio:.2f}× ({'SGLang' if ratio > 1 else 'vLLM'} faster)</b>"
                    )
                else:
                    z_row.append(None)
                    t_row.append("<span style='color:#bbb'>—</span>")
                    missing = []
                    if s is None: missing.append("sglang")
                    if v is None or (v is not None and v <= 0): missing.append("vllm")
                    h_row.append(f"input_len={il}  max_concurrency={c}<br>missing: {', '.join(missing)}")
            z.append(z_row)
            text.append(t_row)
            hover.append(h_row)
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[str(v) for v in input_lens],
                y=[str(v) for v in concs],
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=13),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                colorscale="RdBu",
                zmid=0.0,
                zmin=-zmax,
                zmax=zmax,
                colorbar=dict(
                    title=dict(text="log(sglang / vllm)"),
                    tickvals=[-zmax, -zmax/2, 0, zmax/2, zmax],
                    ticktext=[f"{math.exp(-zmax):.2f}×", f"{math.exp(-zmax/2):.2f}×", "1.0×", f"{math.exp(zmax/2):.2f}×", f"{math.exp(zmax):.2f}×"],
                    x=1.02,
                ) if i == len(output_lens) else None,
                showscale=(i == len(output_lens)),
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="input_len", row=1, col=i, type="category")
        if i == 1:
            fig.update_yaxes(title_text="max_concurrency", row=1, col=i, type="category")
    fig.update_layout(
        title=dict(text="Best SGLang ÷ best vLLM throughput per workload cell (log color scale; blue = SGLang faster)", x=0.0, xanchor="left", font=dict(size=14)),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=120, t=70, b=60),
    )
    return fig


# --------------------------------------------------------------------------
# Scientific decision plots
# --------------------------------------------------------------------------

def _load_zero_floor(result_dir: Path, default: float = 0.05) -> float:
    """Read ``report.robustness.zero_floor`` from ``<result_dir>/config.yaml``.

    Returns ``default`` (0.05) when the config file is absent, the key is
    missing, or the value is non-numeric. See ``_geomean`` for the rationale
    behind the default.
    """
    import yaml
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return default
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return default
    raw = ((cfg.get("report") or {}).get("robustness") or {}).get("zero_floor")
    floored = _num(raw)
    return floored if floored is not None else default


def _load_leaderboard_rows(result_dir: Path, default: int = 5) -> int:
    """Read ``report.leaderboard_rows`` from ``<result_dir>/config.yaml``.

    Returns ``default`` (5) when the config file is absent, the key is
    missing, the value is non-integer, or the value is non-positive.
    Drives the top-K cap on the robustness leaderboard.
    """
    import yaml
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return default
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return default
    raw = (cfg.get("report") or {}).get("leaderboard_rows")
    # bool is a subclass of int in Python; explicitly reject it.
    if isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return raw if raw > 0 else default


def _load_top_k_candidate_cards(result_dir: Path, default: int = 8) -> int:
    """Read ``report.top_k_candidate_cards`` from ``<result_dir>/config.yaml``.

    Returns ``default`` (8) when the config file is absent, the key is
    missing, the value is non-integer, or the value is non-positive.
    Drives the top-K cap on the candidate detail cards section.
    """
    import yaml
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return default
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return default
    raw = (cfg.get("report") or {}).get("top_k_candidate_cards")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return raw if raw > 0 else default


def _load_slo(result_dir: Path) -> dict[str, float]:
    """Read objective.constraints from config.yaml so plots can draw SLO lines."""
    import yaml
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}
    raw = (cfg.get("objective") or {}).get("constraints") or {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        f = _num(v)
        if f is not None:
            out[k] = f
    return out


def _load_objective_maximize(result_dir: Path, default: str = "goodput_at_slo") -> str:
    """Read ``objective.maximize`` from ``<result_dir>/config.yaml``.

    Returns ``default`` when the config file is absent or the key is missing.
    Used by the decision hero to label what "best" means.
    """
    import yaml
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return default
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return default
    raw = (cfg.get("objective") or {}).get("maximize")
    return str(raw) if raw else default


def _figure_operating_curve(measurements: list[dict[str, Any]], best_by_backend: dict[str, str], slo: dict[str, float]) -> Any | None:
    """Throughput vs latency along the concurrency axis. The canonical serving graph.

    For each backend's best config, plot the median throughput vs median p99 TTFT
    (and p99 TPOT in a second panel). Points are connected in order of increasing
    max_concurrency, so the line traces the operating curve from low-load to saturation.

    The SLO is drawn as a horizontal line; the intersection of the curve with the
    SLO is annotated as "max throughput at SLO" per backend.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not best_by_backend:
        return None
    rows_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in measurements:
        if not _is_valid(r):
            continue
        backend = str(r.get("backend") or "")
        cid = str(r.get("candidate_id") or "")
        if best_by_backend.get(backend) != cid:
            continue
        rows_by_backend[backend].append(r)
    if not any(rows_by_backend.values()):
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=["<b>p99 TTFT</b> vs throughput", "<b>p99 TPOT</b> vs throughput"], horizontal_spacing=0.10)

    def _medians(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        """Median across repeats per (input_len, output_len, max_concurrency)."""
        groups: dict[tuple[Any, Any, Any], list[dict[str, float]]] = defaultdict(list)
        for r in rows:
            key = (r.get("input_len"), r.get("output_len"), r.get("max_concurrency"))
            groups[key].append({
                "tput": _num(r.get("output_tok_s_per_gpu")),
                "ttft": _num(r.get("p99_ttft_ms")),
                "tpot": _num(r.get("p99_tpot_ms")),
                "conc": _num(r.get("max_concurrency")) or 0,
                "input_len": _num(r.get("input_len")) or 0,
                "output_len": _num(r.get("output_len")) or 0,
            })
        agg = []
        for key, points in groups.items():
            def med(field):
                vs = sorted(p[field] for p in points if p[field] is not None)
                return vs[len(vs) // 2] if vs else None
            agg.append({
                "input_len": points[0]["input_len"],
                "output_len": points[0]["output_len"],
                "conc": points[0]["conc"],
                "tput": med("tput"),
                "ttft": med("ttft"),
                "tpot": med("tpot"),
            })
        # Sort by concurrency so the line traces the operating curve direction.
        agg.sort(key=lambda r: (r["input_len"], r["output_len"], r["conc"]))
        return agg

    slo_ttft = slo.get("p99_ttft_ms")
    slo_tpot = slo.get("p99_tpot_ms")
    annotations = []
    for backend, rows in sorted(rows_by_backend.items()):
        color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)
        agg = _medians(rows)
        # Group by (input_len, output_len) so each prompt-shape gets its own line.
        by_shape: dict[tuple[Any, Any], list[dict[str, float]]] = defaultdict(list)
        for r in agg:
            by_shape[(r["input_len"], r["output_len"])].append(r)
        for (in_len, out_len), pts in by_shape.items():
            label = f"{backend} (in={int(in_len)}, out={int(out_len)})"
            hover = [
                f"<b>{label}</b><br>conc={int(p['conc'])}<br>throughput={p['tput']:.1f}<br>p99 TTFT={p['ttft']:.1f}<br>p99 TPOT={p['tpot']:.2f}"
                for p in pts
            ]
            xs_ttft = [p["tput"] for p in pts if p["tput"] is not None and p["ttft"] is not None]
            ys_ttft = [p["ttft"] for p in pts if p["tput"] is not None and p["ttft"] is not None]
            xs_tpot = [p["tput"] for p in pts if p["tput"] is not None and p["tpot"] is not None]
            ys_tpot = [p["tpot"] for p in pts if p["tput"] is not None and p["tpot"] is not None]
            if xs_ttft:
                fig.add_trace(go.Scatter(
                    x=xs_ttft, y=ys_ttft, mode="lines+markers", name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=9, color=color, line=dict(width=1, color="white")),
                    text=hover, hoverinfo="text", legendgroup=backend,
                ), row=1, col=1)
            if xs_tpot:
                fig.add_trace(go.Scatter(
                    x=xs_tpot, y=ys_tpot, mode="lines+markers", name=label,
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=9, color=color, symbol="diamond", line=dict(width=1, color="white")),
                    text=hover, hoverinfo="text", legendgroup=backend, showlegend=False,
                ), row=1, col=2)
        # Annotate the highest-throughput point where the SLO is still met.
        # axref/ayref must be "pixel" — see _figure_pareto note about log-axis autorange.
        if slo_ttft is not None:
            ok = [p for p in agg if p["tput"] is not None and p["ttft"] is not None and p["ttft"] <= slo_ttft]
            if ok:
                pmax = max(ok, key=lambda p: p["tput"])
                annotations.append(dict(
                    xref="x1", yref="y1", axref="pixel", ayref="pixel",
                    x=pmax["tput"], y=pmax["ttft"],
                    text=f"<b>{backend} @SLO: {pmax['tput']:.0f} tok/s/GPU</b>",
                    showarrow=True, arrowhead=2, arrowcolor=color, ax=0, ay=-30,
                    font=dict(size=11, color=color),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor=color, borderwidth=1, borderpad=3,
                ))
    if slo_ttft is not None:
        fig.add_hline(y=slo_ttft, line=dict(color="#888", dash="dash", width=1), row=1, col=1, annotation=dict(text=f"SLO {slo_ttft:.0f} ms", font=dict(size=10, color="#888"), x=1, xanchor="right"))
    if slo_tpot is not None:
        fig.add_hline(y=slo_tpot, line=dict(color="#888", dash="dash", width=1), row=1, col=2, annotation=dict(text=f"SLO {slo_tpot:.0f} ms", font=dict(size=10, color="#888"), x=1, xanchor="right"))

    fig.update_xaxes(title_text="output tok/s/GPU", row=1, col=1)
    fig.update_xaxes(title_text="output tok/s/GPU", row=1, col=2)
    fig.update_yaxes(title_text="p99 TTFT (ms, log)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="p99 TPOT (ms, log)", type="log", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=470,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=60, r=30, t=60, b=80),
        annotations=list(fig.layout.annotations) + annotations,
    )
    return fig


def _figure_latency_vs_promptlen(measurements: list[dict[str, Any]], best_by_backend: dict[str, str], metric: str, y_label: str) -> Any | None:
    """Best-config line per backend with min/max band across all configs of that backend.

    For each input_len, plot the best-config's median latency as a line, and shade
    [min, max] across all configs of that backend at that input_len. The width of
    the band shows how much tuning matters.
    """
    import plotly.graph_objects as go
    if not best_by_backend:
        return None
    rows_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in measurements:
        if not _is_valid(r):
            continue
        backend = str(r.get("backend") or "")
        if backend:
            rows_by_backend[backend].append(r)
    if not rows_by_backend:
        return None

    fig = go.Figure()
    for backend, rows in sorted(rows_by_backend.items()):
        color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)
        # Take median across (output_len, max_concurrency, repeat) at each input_len.
        # Conditioning on max_concurrency=1 makes the curve about pure prefill cost
        # without scheduler queueing; fall back to all if no conc=1 data.
        conc1 = [r for r in rows if _num(r.get("max_concurrency")) == 1]
        pool = conc1 if conc1 else rows
        best_id = best_by_backend.get(backend)
        by_input: dict[float, list[float]] = defaultdict(list)
        by_input_best: dict[float, list[float]] = defaultdict(list)
        for r in pool:
            il = _num(r.get("input_len"))
            v = _num(r.get(metric))
            if il is None or v is None:
                continue
            by_input[il].append(v)
            if str(r.get("candidate_id")) == best_id:
                by_input_best[il].append(v)
        if not by_input:
            continue
        input_lens = sorted(by_input)
        med_band_min = []
        med_band_max = []
        med_best = []
        for il in input_lens:
            vals = sorted(by_input[il])
            med_band_min.append(vals[0])
            med_band_max.append(vals[-1])
            best_vals = sorted(by_input_best.get(il, []))
            med_best.append(best_vals[len(best_vals) // 2] if best_vals else (vals[len(vals) // 2]))
        # Band shaded between best and worst config of that backend at each input_len.
        fig.add_trace(go.Scatter(
            x=input_lens + input_lens[::-1],
            y=med_band_max + med_band_min[::-1],
            fill="toself", fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name=f"{backend} config range",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=input_lens, y=med_best,
            mode="lines+markers", name=f"{backend} best",
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(width=1, color="white")),
            hovertemplate=f"<b>{backend}</b><br>input_len=%{{x}}<br>{y_label}=%{{y:.2f}} ms<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        height=380,
        xaxis=dict(title="input_len (log)", type="log"),
        yaxis=dict(title=f"{y_label} (ms, log)", type="log"),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=60, r=30, t=20, b=70),
    )
    return fig


def _figure_decision_map(measurements: list[dict[str, Any]]) -> Any | None:
    """Per workload cell: who wins, and by how much.

    Heatmap of (input_len x max_concurrency), faceted by output_len. Cell color
    = log-ratio of winning backend over loser (diverging, centered at 0). Cell
    text = "<winner> 1.27x". Cells where only one backend ran are still shown
    (with annotation), so missing measurements are visible.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    by_backend: dict[str, dict[tuple[Any, Any, Any], float]] = defaultdict(dict)
    for row in measurements:
        if not _is_valid(row):
            continue
        tput = _num(row.get("output_tok_s_per_gpu"))
        if tput is None:
            continue
        key = (row.get("input_len"), row.get("output_len"), row.get("max_concurrency"))
        backend = str(row.get("backend") or "")
        cur = by_backend[backend].get(key, -math.inf)
        if tput > cur:
            by_backend[backend][key] = tput
    backends = sorted(by_backend.keys())
    if len(backends) < 2:
        return None

    all_keys = set().union(*[set(d.keys()) for d in by_backend.values()])
    if not all_keys:
        return None
    output_lens = sorted({k[1] for k in all_keys}, key=lambda v: (v is None, v))

    log_ratios = []
    for k in all_keys:
        vals = [by_backend[b].get(k) for b in backends]
        if all(v is not None and v > 0 for v in vals):
            log_ratios.append(math.log(max(vals) / min(vals)))
    zmax = max(0.05, max(log_ratios) if log_ratios else 0.05)

    fig = make_subplots(rows=1, cols=len(output_lens),
                       subplot_titles=[f"<b>output_len = {o}</b>" for o in output_lens],
                       shared_yaxes=True, horizontal_spacing=0.05)
    for i, out_len in enumerate(output_lens, start=1):
        sub_keys = [k for k in all_keys if k[1] == out_len]
        input_lens = sorted({k[0] for k in sub_keys}, key=lambda v: (v is None, v))
        concs = sorted({k[2] for k in sub_keys}, key=lambda v: (v is None, v))
        z, text, hover = [], [], []
        for c in concs:
            z_row, t_row, h_row = [], [], []
            for il in input_lens:
                key = (il, out_len, c)
                vals = {b: by_backend[b].get(key) for b in backends}
                present = {b: v for b, v in vals.items() if v is not None and v > 0}
                if len(present) >= 2:
                    winner = max(present, key=present.get)
                    loser = min(present, key=present.get)
                    ratio = present[winner] / present[loser]
                    # Signed log-ratio so vLLM-wins is negative, SGLang-wins is positive when those are the two.
                    sign = 1.0 if winner == "sglang" else (-1.0 if winner == "vllm" else (1.0 if winner > loser else -1.0))
                    z_row.append(sign * math.log(ratio))
                    color = BACKEND_COLORS.get(winner, "#444")
                    t_row.append(f"<b style='color:{color}'>{winner}</b><br>{ratio:.2f}×")
                    h_row.append(
                        f"input_len={il}, max_concurrency={c}<br>"
                        + "<br>".join(f"{b}: {v:.1f} tok/s/GPU" for b, v in present.items())
                        + f"<br><b>{winner} wins by {ratio:.2f}×</b>"
                    )
                elif len(present) == 1:
                    only = next(iter(present))
                    z_row.append(None)
                    t_row.append(f"<span style='color:#aaa'>only<br>{only}</span>")
                    h_row.append(f"input_len={il}, max_concurrency={c}<br>only {only} ran")
                else:
                    z_row.append(None)
                    t_row.append("<span style='color:#bbb'>—</span>")
                    h_row.append(f"input_len={il}, max_concurrency={c}<br>no data")
            z.append(z_row)
            text.append(t_row)
            hover.append(h_row)
        fig.add_trace(go.Heatmap(
            z=z,
            x=[str(v) for v in input_lens],
            y=[str(v) for v in concs],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="RdBu",
            zmid=0.0, zmin=-zmax, zmax=zmax,
            colorbar=dict(
                title=dict(text="winner margin<br>(log ratio)"),
                tickvals=[-zmax, 0, zmax],
                ticktext=[f"vllm {math.exp(zmax):.2f}×", "tie", f"sglang {math.exp(zmax):.2f}×"],
                x=1.02,
            ) if i == len(output_lens) else None,
            showscale=(i == len(output_lens)),
        ), row=1, col=i)
        fig.update_xaxes(title_text="input_len", row=1, col=i, type="category")
        if i == 1:
            fig.update_yaxes(title_text="max_concurrency", row=1, col=i, type="category")
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=140, t=70, b=60),
    )
    return fig


def _figure_tuning_sensitivity(measurements: list[dict[str, Any]], rankings: list[dict[str, Any]]) -> Any | None:
    """How much does config choice matter, per backend per workload?

    Boxplot of median output_tok_s_per_gpu across all configs of that backend
    at each workload. Tight + high box = backend is easy to tune well. Wide
    box = tuning matters a lot.
    """
    import plotly.graph_objects as go
    # Use rankings (already aggregated per candidate × workload) so we get medians.
    if not rankings:
        return None
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rankings:
        backend = str(r.get("backend") or "")
        workload = _workload_label(r)
        v = _num(r.get("median_output_tok_s_per_gpu") or r.get("objective_value"))
        if v is None or v <= 0:
            continue
        by_cell[(backend, workload)].append(v)
    if not by_cell:
        return None
    workloads_sorted = sorted({wl for (_, wl) in by_cell.keys()})
    backends_sorted = sorted({b for (b, _) in by_cell.keys()})
    fig = go.Figure()
    for backend in backends_sorted:
        ys, xs = [], []
        for wl in workloads_sorted:
            vals = by_cell.get((backend, wl), [])
            for v in vals:
                xs.append(wl)
                ys.append(v)
        if ys:
            fig.add_trace(go.Box(
                x=xs, y=ys, name=backend,
                marker_color=BACKEND_COLORS.get(backend, DEFAULT_COLOR),
                boxmean=True, boxpoints="all", jitter=0.4, pointpos=0,
                line=dict(width=1.5),
            ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        boxmode="group",
        yaxis=dict(title="output tok/s/GPU"),
        xaxis=dict(title="workload", tickangle=-30),
        legend=dict(orientation="h", y=-0.30),
        margin=dict(l=60, r=30, t=20, b=120),
    )
    return fig


def _figure_candidate_throughput(measurements: list[dict[str, Any]], candidate_id: str, backend: str) -> Any | None:
    """Stacked subplots for one candidate: throughput bars (top), latency lines on log y (bottom)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Median across repeats per workload, so each x position is one cell.
    by_wl: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in measurements:
        if str(r.get("candidate_id")) != candidate_id or not _is_valid(r):
            continue
        by_wl[_workload_key(r)].append(r)
    if not by_wl:
        return None
    items = []
    for wkey, group in by_wl.items():
        def _median_of(metric: str) -> float | None:
            vals = sorted(v for v in (_num(r.get(metric)) for r in group) if v is not None)
            if not vals:
                return None
            return vals[len(vals) // 2]
        sample = group[0]
        items.append({
            "label": _workload_label(sample),
            "sort_key": (_num(sample.get("input_len")) or 0, _num(sample.get("output_len")) or 0, _num(sample.get("max_concurrency")) or 0),
            "tput": _median_of("output_tok_s_per_gpu"),
            "ttft": _median_of("p99_ttft_ms"),
            "tpot": _median_of("p99_tpot_ms"),
            "n": len(group),
        })
    items.sort(key=lambda r: r["sort_key"])
    labels = [r["label"] for r in items]
    tput = [r["tput"] for r in items]
    ttft = [r["ttft"] for r in items]
    tpot = [r["tpot"] for r in items]
    color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45])
    fig.add_trace(go.Bar(
        x=labels, y=tput, name="output tok/s/GPU",
        marker_color=color, marker_line_color="white", marker_line_width=1,
        text=[f"{v:.0f}" if v is not None else "" for v in tput],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>throughput: %{y:.1f} tok/s/GPU<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=labels, y=ttft, name="p99 TTFT", mode="lines+markers",
        line=dict(color="#d62728", width=2), marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>p99 TTFT: %{y:.1f} ms<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=labels, y=tpot, name="p99 TPOT", mode="lines+markers",
        line=dict(color="#9467bd", width=2, dash="dot"), marker=dict(size=8, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>p99 TPOT: %{y:.2f} ms<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="tok/s/GPU", row=1, col=1)
    fig.update_yaxes(title_text="ms (log)", type="log", row=2, col=1)
    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=60, r=30, t=20, b=80),
        bargap=0.25,
    )
    return fig


def _embed_figure(fig: Any, include_js: bool = False, div_id: str | None = None) -> str:
    if fig is None:
        return ""
    return fig.to_html(
        include_plotlyjs="inline" if include_js else False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _section_pareto(measurements: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]]) -> tuple[str, bool]:
    fig_ttft = _figure_pareto(measurements, candidates_by_id, "p99_ttft_ms", "Throughput vs p99 TTFT", "p99 TTFT")
    fig_tpot = _figure_pareto(measurements, candidates_by_id, "p99_tpot_ms", "Throughput vs p99 TPOT", "p99 TPOT")
    body = (
        '<section><h2>Pareto explorer (all candidates)</h2>'
        '<p>One point per (candidate × workload), median across repeats. '
        '<b>Star markers</b> are on the per-backend Pareto frontier; the dotted line connects them. '
        'The labeled point is the highest-throughput candidate per backend. Hover for full params.</p>'
        f'<div class="figure">{_embed_figure(fig_ttft, div_id="pareto-ttft")}</div>'
        f'<div class="figure">{_embed_figure(fig_tpot, div_id="pareto-tpot")}</div>'
        '</section>'
    )
    return body, False


def _section_parcoords(rankings: list[dict[str, Any]], candidates: list[dict[str, Any]], js_already_included: bool) -> str:
    parts = ['<section><h2>Parameter sweep</h2>',
             '<p>Parallel coordinates over the varying server parameters per backend. '
             'Color encodes peak <code>output_tok_s_per_gpu</code> — darker lines are stronger candidates.</p>']
    any_drawn = False
    for backend in sorted({str(c.get("backend")) for c in candidates}):
        fig = _figure_parcoords(rankings, candidates, backend)
        if fig is None:
            continue
        parts.append(f'<div class="figure">{_embed_figure(fig, include_js=not js_already_included and not any_drawn)}</div>')
        any_drawn = True
    if not any_drawn:
        parts.append('<p><em>No varying scalar parameters across candidates.</em></p>')
    parts.append("</section>")
    return "".join(parts)


def _section_backend_ratio(measurements: list[dict[str, Any]]) -> str:
    fig = _figure_backend_ratio(measurements)
    if fig is None:
        return ('<section><h2>Backend head-to-head</h2>'
                '<p><em>Need both backends with overlapping workloads to draw the ratio heatmap.</em></p></section>')
    return ('<section><h2>Backend head-to-head</h2>'
            '<p>Ratio of the best SGLang throughput to the best vLLM throughput for each workload cell. '
            'Cells above 1.0 (blue) favor SGLang; below 1.0 (red) favor vLLM.</p>'
            f'<div class="figure">{_embed_figure(fig)}</div></section>')


def _section_per_workload_drilldown(rankings: list[dict[str, Any]]) -> str:
    by_workload: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rankings:
        by_workload[_workload_key(r)].append(r)
    parts = ['<section><h2>Per-workload drill-down</h2>',
             '<p>Top 5 candidates for each individual workload constraint. '
             'Click a <code>candidate_id</code> to jump to its detail card.</p>']
    if not by_workload:
        parts.append('<p><em>No rankings.</em></p></section>')
        return "".join(parts)
    for wkey in sorted(by_workload.keys(), key=lambda k: tuple(("",) if v is None else (str(v),) for v in k)):
        rows = sorted(by_workload[wkey], key=lambda r: _num(r.get("rank")) or 1e9)[:5]
        if not rows:
            continue
        label = _workload_label(rows[0])
        body_rows = []
        for r in rows:
            body_rows.append(
                "<tr>"
                f"<td>{_esc(r.get('rank'))}</td>"
                f'<td><a href="#{_anchor(str(r.get("candidate_id")))}"><code>{_esc(r.get("candidate_id"))}</code></a></td>'
                f"<td>{_esc(r.get('backend'))}</td>"
                f"<td>{_esc(r.get('serve_config'))}</td>"
                f"<td>{_fmt(r.get('median_output_tok_s_per_gpu') or r.get('objective_value'))}</td>"
                f"<td>{_fmt(r.get('relative_to_best'), 3)}</td>"
                f"<td>{_fmt(r.get('median_p99_ttft_ms'))}</td>"
                f"<td>{_fmt(r.get('median_p99_tpot_ms'))}</td>"
                "</tr>"
            )
        parts.append(f'<h3>{_esc(label)}</h3>'
                     '<table class="data"><thead><tr>'
                     '<th>rank</th><th>candidate</th><th>backend</th><th>serve_config</th>'
                     '<th>tok/s/GPU</th><th>rel-to-best</th><th>p99 TTFT</th><th>p99 TPOT</th>'
                     '</tr></thead><tbody>'
                     f'{"".join(body_rows)}</tbody></table>')
    parts.append("</section>")
    return "".join(parts)


def _section_candidate_cards(
    measurements: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    robustness_by_cid: dict[str, dict[str, Any]],
    top_k: int = 8,
) -> str:
    """Render the per-candidate detail cards with a top-K visible cap.

    Cards are sorted by robustness rank (geomean rel-to-best, desc). The
    first ``top_k`` cards render visibly; any remaining cards are wrapped
    in a ``<details>`` expander so big sweeps stay manageable. When the
    total candidate count is at or below the cap, no expander is rendered.
    """
    ranks_by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rankings:
        ranks_by_cid[str(r.get("candidate_id"))].append(r)

    sorted_candidates = sorted(
        candidates,
        key=lambda c: -(robustness_by_cid.get(str(c.get("candidate_id")), {}).get("geomean_relative_to_best") or 0),
    )
    # Drop candidates with no id; preserve the rest in robustness-rank order.
    sorted_candidates = [c for c in sorted_candidates if str(c.get("candidate_id") or "")]

    def _card_html(cand: dict[str, Any]) -> str:
        cid = str(cand.get("candidate_id") or "")
        backend = str(cand.get("backend") or "")
        color = BACKEND_COLORS.get(backend, DEFAULT_COLOR)
        rob = robustness_by_cid.get(cid, {})
        params = _curated_params(cand)
        ranks = sorted(ranks_by_cid.get(cid, []), key=lambda r: (str(r.get("workload"))))
        rank_rows = []
        for r in ranks:
            rank_rows.append(
                "<tr>"
                f"<td>{_esc(_workload_label(r))}</td>"
                f"<td>{_esc(r.get('rank'))}</td>"
                f"<td>{_fmt(r.get('relative_to_best'), 3)}</td>"
                f"<td>{_fmt(r.get('median_output_tok_s_per_gpu') or r.get('objective_value'))}</td>"
                f"<td>{_fmt(r.get('median_p99_ttft_ms'))}</td>"
                f"<td>{_fmt(r.get('median_p99_tpot_ms'))}</td>"
                "</tr>"
            )
        fig = _figure_candidate_throughput(measurements, cid, backend)
        fig_html = f'<div class="figure">{_embed_figure(fig)}</div>' if fig is not None else ""
        return f"""
        <article class="candidate" id="{_anchor(cid)}" style="border-left-color:{color}">
          <h3>{_esc(cand.get("serve_config") or cid)} <span class="muted">/ {_esc(backend)}</span></h3>
          <p class="candidate-id">candidate_id: <code>{_esc(cid)}</code></p>
          <ul class="metrics">
            <li><strong>{_fmt(rob.get("geomean_relative_to_best"), 3)}</strong> geomean rel-to-best</li>
            <li><strong>{rob.get("workload_wins", 0)}</strong> / {rob.get("workloads_evaluated", 0)} wins</li>
            <li><strong>{_fmt(rob.get("worst_relative_to_best"), 3)}</strong> worst rel-to-best</li>
            <li><strong>{rob.get("pareto_appearances", 0)}</strong> Pareto appearances</li>
          </ul>
          <div class="candidate-grid">
            <div>
              <h4>Server parameters</h4>
              {_kv_block(params)}
            </div>
            <div>
              <h4>Per-workload performance</h4>
              {fig_html or "<p><em>No valid measurements for this candidate.</em></p>"}
            </div>
          </div>
          <details>
            <summary>Per-workload ranks</summary>
            <table class="data">
              <thead><tr>
                <th>workload</th><th>rank</th><th>rel-to-best</th><th>tok/s/GPU</th><th>p99 TTFT</th><th>p99 TPOT</th>
              </tr></thead>
              <tbody>{"".join(rank_rows) or '<tr><td colspan="6"><em>no rankings</em></td></tr>'}</tbody>
            </table>
          </details>
          <details>
            <summary>Server command</summary>
            <pre><code>{_esc(cand.get("server_cmd") or "")}</code></pre>
          </details>
        </article>
        """

    parts: list[str] = [
        '<section><h2>Candidate detail cards</h2>',
        '<p>Anchored details for each candidate. Linked from the leaderboards above.</p>',
    ]
    total = len(sorted_candidates)
    visible = sorted_candidates[:top_k]
    hidden = sorted_candidates[top_k:]
    for cand in visible:
        parts.append(_card_html(cand))
    if hidden:
        parts.append(f'<details><summary>Show all {total} candidates</summary>')
        for cand in hidden:
            parts.append(_card_html(cand))
        parts.append('</details>')
    parts.append("</section>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Live data for the constraint-conditional leaderboard
# --------------------------------------------------------------------------

def _live_data(
    measurements: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    slo: dict[str, float] | None = None,
    objective: str | None = None,
    zero_floor: float = 0.05,
) -> dict[str, Any]:
    measurement_rows = []
    for row in measurements:
        if not _is_valid(row):
            continue
        cid = str(row.get("candidate_id") or "")
        if not cid:
            continue
        cand = candidates_by_id.get(cid, {})
        measurement_rows.append({
            "candidate_id": cid,
            "backend": str(row.get("backend") or ""),
            "serve_config": str(cand.get("serve_config") or row.get("serve_config") or ""),
            "workload_key": list(_workload_key(row)),
            "workload_label": _workload_label(row),
            "output_tok_s_per_gpu": _num(row.get("output_tok_s_per_gpu")),
            "p99_ttft_ms": _num(row.get("p99_ttft_ms")),
            "p99_tpot_ms": _num(row.get("p99_tpot_ms")),
        })
    # Per-candidate median p99 latencies per workload — the slider JS needs
    # these to decide which workloads pass SLO when re-ranking client-side.
    # The JS recomputes recommendation from these directly so we don't depend
    # on the static rankings.csv (which is pinned to the YAML SLOs).
    cand_meta: dict[str, dict[str, Any]] = {}
    for cid, cand in candidates_by_id.items():
        cand_meta[cid] = {
            "backend": str(cand.get("backend") or ""),
            "serve_config": str(cand.get("serve_config") or cid),
        }
    return {
        "measurements": measurement_rows,
        "candidate_meta": cand_meta,
        "slo": slo or {},
        "objective": objective or "",
        "zero_floor": zero_floor,
    }


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --fg: #1d2433;
  --muted: #6b7280;
  --bg: #f8fafc;
  --card: #ffffff;
  --border: #e5e7eb;
  --accent: #2563eb;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 24px 64px; background: var(--bg); color: var(--fg); font: 14.5px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
header { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px 28px; margin-bottom: 24px; }
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { margin-top: 0; font-size: 20px; }
h3 { margin-top: 0; }
h4 { margin: 6px 0 10px; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
.meta { display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted); margin-bottom: 12px; }
.meta code { color: var(--fg); }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.chip { display: inline-flex; gap: 6px; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--border); font-size: 12.5px; }
.chip-row { display: inline-flex; gap: 10px; }
.counts { display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted); }
.counts strong { color: var(--fg); }
section { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px 26px; margin-bottom: 22px; }
section.hero { border-left: 6px solid var(--accent); }
.hero .lead { color: var(--muted); margin-top: -4px; }
.hero-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 28px; align-items: start; }
@media (max-width: 800px) { .hero-grid { grid-template-columns: 1fr; } }
.metrics { list-style: none; padding: 0; margin: 12px 0; display: flex; gap: 18px; flex-wrap: wrap; }
.metrics li { background: var(--bg); padding: 8px 14px; border-radius: 8px; }
.metrics strong { font-size: 18px; }
.candidate-id { color: var(--muted); margin-top: -4px; }
table.data, table.kv { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.data th, table.data td { padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
table.data thead th { background: var(--bg); position: sticky; top: 0; font-weight: 600; }
table.data tbody tr:hover { background: #f1f5ff; }
table.data tbody tr.winner-row { background: #fef3c7; font-weight: 600; }
table.data tbody tr.winner-row:hover { background: #fde68a; }
table.kv th { text-align: left; color: var(--muted); padding: 3px 8px 3px 0; font-weight: 500; width: 1%; white-space: nowrap; }
table.kv td { padding: 3px 0; }
code { background: var(--bg); padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }
pre { background: var(--bg); padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }
pre code { padding: 0; background: transparent; }
.muted { color: var(--muted); font-weight: normal; }
.controls { display: flex; gap: 22px; flex-wrap: wrap; align-items: end; margin-bottom: 14px; padding: 14px 16px; background: var(--bg); border-radius: 10px; }
.controls label { display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; color: var(--muted); }
.controls input[type="range"] { width: 220px; }
.controls select, .controls input { font-size: 13px; }
.controls output { color: var(--fg); font-weight: 600; }
.figure { margin: 12px 0 6px; }
.divider { margin: 36px 0 14px; padding: 0 4px; }
.divider-title { font-size: 22px; margin: 0; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
.divider-sub { color: var(--muted); margin: 4px 0 0; font-size: 13px; }
section h3 { margin: 0 0 4px; font-size: 17px; }
section h3 + p { color: var(--muted); margin-top: 0; font-size: 13.5px; }
article.candidate { background: var(--card); border: 1px solid var(--border); border-left: 5px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 14px; }
.candidate-grid { display: grid; grid-template-columns: minmax(240px, 1fr) 2fr; gap: 22px; }
@media (max-width: 800px) { .candidate-grid { grid-template-columns: 1fr; } }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--accent); font-size: 13px; }
"""

JS_TEMPLATE = """
const __DATA__ = %s;

(function() {
  const ttft = document.getElementById('cc-ttft');
  const tpot = document.getElementById('cc-tpot');
  const ttftOut = document.getElementById('cc-ttft-out');
  const tpotOut = document.getElementById('cc-tpot-out');
  const wl = document.getElementById('cc-workload');
  const body = document.getElementById('cc-body');
  const chips = Array.from(document.querySelectorAll('.cc-backend'));
  if (!ttft || !body) return;

  // Hero + robustness-leaderboard hooks (issue 006). May be missing on
  // degraded pages or no-SLO runs — guard each lookup.
  const heroWinner = document.querySelector('[data-rec-winner]');
  const heroSloLabel = document.querySelector('[data-rec-slo-label]');
  const heroWins = document.querySelector('[data-rec-evidence-wins]');
  const heroMargin = document.querySelector('[data-rec-evidence-margin]');
  const heroWorst = document.querySelector('[data-rec-evidence-worst]');
  const robBody = document.getElementById('robustness-body');
  const robBodyOverflow = document.getElementById('robustness-body-overflow');

  const ZERO_FLOOR = (typeof __DATA__.zero_floor === 'number') ? __DATA__.zero_floor : 0.05;
  const BACKEND_COLORS = {vllm: '#1f77b4', sglang: '#ff7f0e'};

  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtRel(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    return v.toFixed(3);
  }
  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
  }
  function passesSlo(p99_ttft_ms, p99_tpot_ms, maxTtft, maxTpot, ttftMax, tpotMax) {
    if (maxTtft < ttftMax) {
      if (p99_ttft_ms === null || p99_ttft_ms === undefined || p99_ttft_ms > maxTtft) return false;
    }
    if (maxTpot < tpotMax) {
      if (p99_tpot_ms === null || p99_tpot_ms === undefined || p99_tpot_ms > maxTpot) return false;
    }
    return true;
  }
  function sloLabelFor(maxTtft, maxTpot, ttftMax, tpotMax) {
    const parts = [];
    if (maxTtft < ttftMax) parts.push('p99_ttft_ms \\u2264 ' + Math.round(maxTtft) + 'ms');
    if (maxTpot < tpotMax) parts.push('p99_tpot_ms \\u2264 ' + Math.round(maxTpot) + 'ms');
    return parts.join(', ');
  }

  function computeRecommendation(maxTtft, maxTpot, ttftMax, tpotMax) {
    // Port of moe_bench/recommendation.py:compute_recommendation. We
    // aggregate per (candidate, workload) by median throughput / latencies,
    // mark each cell SLO-pass against the LIVE slider values, derive
    // relative_to_best per workload (0 on SLO violation), and compute the
    // zero-floored geomean as the global score.
    const measurements = __DATA__.measurements;
    const groups = new Map();
    measurements.forEach(r => {
      const wk = JSON.stringify(r.workload_key);
      const k = r.candidate_id + '|' + wk;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(r);
    });
    const cellByCand = new Map();
    const allWorkloads = new Set();
    function median(arr) {
      const s = arr.filter(v => v !== null && v !== undefined && !Number.isNaN(v)).sort((a,b)=>a-b);
      return s.length ? s[Math.floor(s.length / 2)] : null;
    }
    groups.forEach((rows, key) => {
      const sep = key.indexOf('|');
      const cid = key.slice(0, sep);
      const wkStr = key.slice(sep + 1);
      allWorkloads.add(wkStr);
      const medTput = median(rows.map(r => r.output_tok_s_per_gpu));
      const medTtft = median(rows.map(r => r.p99_ttft_ms));
      const medTpot = median(rows.map(r => r.p99_tpot_ms));
      const passes = passesSlo(medTtft, medTpot, maxTtft, maxTpot, ttftMax, tpotMax);
      if (!cellByCand.has(cid)) cellByCand.set(cid, {});
      cellByCand.get(cid)[wkStr] = {
        tput: medTput, passes: passes, label: rows[0].workload_label,
      };
    });
    const bestByWl = new Map();
    allWorkloads.forEach(wkStr => {
      let best = 0;
      cellByCand.forEach((cellMap) => {
        const cell = cellMap[wkStr];
        if (cell && cell.passes && cell.tput !== null && cell.tput > best) best = cell.tput;
      });
      bestByWl.set(wkStr, best);
    });
    const rows = [];
    cellByCand.forEach((cellMap, cid) => {
      const relatives = [];
      const ranks = [];
      let wins = 0;
      let worstRel = null;
      let worstLabel = null;
      const workloadKeys = Array.from(allWorkloads);
      workloadKeys.forEach(wkStr => {
        const cell = cellMap[wkStr];
        if (!cell || cell.tput === null) return;
        const best = bestByWl.get(wkStr);
        let rel = 0;
        if (cell.passes && best > 0) rel = cell.tput / best;
        relatives.push(rel);
        if (rel >= 1.0 - 1e-9) wins += 1;
        if (worstRel === null || rel < worstRel) {
          worstRel = rel;
          worstLabel = cell.label;
        }
        let r = 1;
        cellByCand.forEach((other, oCid) => {
          if (oCid === cid) return;
          const oc = other[wkStr];
          if (oc && oc.passes && oc.tput > cell.tput) r += 1;
        });
        ranks.push(r);
      });
      let geomean = null;
      if (relatives.length > 0) {
        const floored = relatives.map(v => v > ZERO_FLOOR ? v : ZERO_FLOOR);
        const logSum = floored.reduce((s, v) => s + Math.log(v), 0);
        geomean = Math.exp(logSum / floored.length);
      }
      ranks.sort((a, b) => a - b);
      const medianRank = ranks.length ? ranks[Math.floor(ranks.length / 2)] : null;
      const meta = (__DATA__.candidate_meta || {})[cid] || {};
      rows.push({
        candidate_id: cid,
        backend: meta.backend || '',
        serve_config: meta.serve_config || cid,
        geomean: geomean,
        median_rank: medianRank,
        wins: wins,
        workloads_evaluated: relatives.length,
        worst_rel: worstRel,
        worst_label: worstLabel,
      });
    });
    rows.sort((a, b) => {
      const ag = (a.geomean === null) ? -1 : a.geomean;
      const bg = (b.geomean === null) ? -1 : b.geomean;
      if (bg !== ag) return bg - ag;
      const ar = (a.median_rank === null) ? 1e9 : a.median_rank;
      const br = (b.median_rank === null) ? 1e9 : b.median_rank;
      if (ar !== br) return ar - br;
      return b.wins - a.wins;
    });
    return { rows: rows, totalWorkloads: allWorkloads.size };
  }

  function updateHero(rec, maxTtft, maxTpot, ttftMax, tpotMax) {
    if (rec.rows.length === 0) return;
    const winner = rec.rows[0];
    const runnerUp = rec.rows.length >= 2 ? rec.rows[1] : null;
    if (heroWinner) {
      const color = BACKEND_COLORS[winner.backend] || '#0f766e';
      heroWinner.innerHTML =
        '<strong style="color:' + color + '">' + escapeHtml(winner.backend) + '</strong> ' +
        'with config <strong>' + escapeHtml(winner.serve_config) + '</strong>';
    }
    if (heroSloLabel) {
      const liveLabel = sloLabelFor(maxTtft, maxTpot, ttftMax, tpotMax);
      if (liveLabel) {
        heroSloLabel.textContent = liveLabel;
        heroSloLabel.removeAttribute('hidden');
      } else {
        heroSloLabel.textContent = '';
        heroSloLabel.setAttribute('hidden', '');
      }
    }
    if (heroWins) {
      heroWins.textContent = winner.wins + ' of ' + rec.totalWorkloads;
    }
    if (heroMargin) {
      if (runnerUp && runnerUp.geomean !== null && runnerUp.geomean > 0 && winner.geomean !== null) {
        const pct = 100 * (winner.geomean - runnerUp.geomean) / runnerUp.geomean;
        const sign = pct >= 0 ? '+' : '';
        heroMargin.innerHTML =
          sign + pct.toFixed(1) + '%% over <code>' + escapeHtml(runnerUp.serve_config) + '</code>';
        heroMargin.removeAttribute('hidden');
      } else {
        heroMargin.textContent = '';
        heroMargin.setAttribute('hidden', '');
      }
    }
    if (heroWorst) {
      if (winner.worst_rel !== null && winner.worst_label) {
        heroWorst.innerHTML =
          winner.worst_rel.toFixed(2) + '\\u00d7 on <code>' + escapeHtml(winner.worst_label) + '</code>';
        heroWorst.removeAttribute('hidden');
      } else {
        heroWorst.textContent = '';
        heroWorst.setAttribute('hidden', '');
      }
    }
  }

  function resortTbody(tbodyEl, rec, startRank) {
    if (!tbodyEl) return startRank;
    const existing = Array.from(tbodyEl.querySelectorAll('tr[data-candidate-id]'));
    if (existing.length === 0) return startRank;
    const byCid = new Map();
    existing.forEach(tr => byCid.set(tr.getAttribute('data-candidate-id'), tr));
    const ordered = [];
    const placed = new Set();
    rec.rows.forEach(r => {
      const tr = byCid.get(r.candidate_id);
      if (tr && !placed.has(r.candidate_id)) {
        ordered.push({tr: tr, score: r});
        placed.add(r.candidate_id);
      }
    });
    existing.forEach(tr => {
      const cid = tr.getAttribute('data-candidate-id');
      if (!placed.has(cid)) {
        ordered.push({tr: tr, score: null});
        placed.add(cid);
      }
    });
    let rank = startRank;
    ordered.forEach(item => {
      const rankCell = item.tr.querySelector('td[data-col="rank"]');
      if (rankCell) rankCell.textContent = String(rank);
      const geomCell = item.tr.querySelector('td[data-col="geomean"]');
      if (geomCell && item.score && item.score.geomean !== null) {
        geomCell.textContent = fmtRel(item.score.geomean);
      }
      const winsCell = item.tr.querySelector('td[data-col="wins"]');
      if (winsCell && item.score) {
        winsCell.textContent = item.score.wins + '/' + item.score.workloads_evaluated;
      }
      if (rank === 1) item.tr.classList.add('winner-row');
      else item.tr.classList.remove('winner-row');
      tbodyEl.appendChild(item.tr);
      rank += 1;
    });
    return rank;
  }

  function updateRobustness(rec) {
    // Re-sort BOTH the top-K visible tbody (005) and the overflow tbody
    // behind 005's <details> expander. Each tbody is sorted independently
    // by the new global score, then rank cells continue numbering across
    // them. We don't move rows between tbodies (that would fight 005's
    // top-K invariant). The winner-row highlight always lands on rank 1
    // in the visible body.
    if (!robBody && !robBodyOverflow) return;
    const nextRank = resortTbody(robBody, rec, 1);
    resortTbody(robBodyOverflow, rec, nextRank);
  }

  function renderConstraintLeaderboard(maxTtft, maxTpot, ttftMax, tpotMax) {
    const enabledBackends = new Set(chips.filter(c => c.checked).map(c => c.value));
    const wlValue = wl.value;
    let rows = __DATA__.measurements.filter(r => {
      if (!enabledBackends.has(r.backend)) return false;
      if (maxTtft < ttftMax && r.p99_ttft_ms !== null && r.p99_ttft_ms > maxTtft) return false;
      if (maxTtft < ttftMax && r.p99_ttft_ms === null) return false;
      if (maxTpot < tpotMax && r.p99_tpot_ms !== null && r.p99_tpot_ms > maxTpot) return false;
      if (maxTpot < tpotMax && r.p99_tpot_ms === null) return false;
      if (wlValue !== '__all__') {
        const wkey = JSON.parse(wlValue);
        if (JSON.stringify(r.workload_key) !== JSON.stringify(wkey)) return false;
      }
      return true;
    });
    if (wlValue === '__all__') {
      const byCand = new Map();
      rows.forEach(r => {
        const cur = byCand.get(r.candidate_id);
        if (!cur || (r.output_tok_s_per_gpu || 0) > (cur.output_tok_s_per_gpu || 0)) {
          byCand.set(r.candidate_id, r);
        }
      });
      rows = Array.from(byCand.values());
    }
    rows.sort((a, b) => (b.output_tok_s_per_gpu || 0) - (a.output_tok_s_per_gpu || 0));
    body.innerHTML = rows.map((r, i) =>
      '<tr>' +
        '<td>' + (i + 1) + '</td>' +
        '<td><a href="#cand-' + r.candidate_id + '"><code>' + r.candidate_id + '</code></a></td>' +
        '<td>' + r.backend + '</td>' +
        '<td>' + r.serve_config + '</td>' +
        '<td>' + r.workload_label + '</td>' +
        '<td><strong>' + fmt(r.output_tok_s_per_gpu) + '</strong></td>' +
        '<td>' + fmt(r.p99_ttft_ms) + '</td>' +
        '<td>' + fmt(r.p99_tpot_ms) + '</td>' +
      '</tr>'
    ).join('') || '<tr><td colspan="8"><em>No candidates match these constraints.</em></td></tr>';
  }

  function render() {
    const ttftMax = +ttft.max;
    const tpotMax = +tpot.max;
    const maxTtft = +ttft.value;
    const maxTpot = +tpot.value;
    ttftOut.textContent = maxTtft >= ttftMax ? '\\u221e' : maxTtft.toString();
    tpotOut.textContent = maxTpot >= tpotMax ? '\\u221e' : maxTpot.toString();
    renderConstraintLeaderboard(maxTtft, maxTpot, ttftMax, tpotMax);
    const rec = computeRecommendation(maxTtft, maxTpot, ttftMax, tpotMax);
    updateHero(rec, maxTtft, maxTpot, ttftMax, tpotMax);
    updateRobustness(rec);
  }

  // Calibrate slider maxes to the data so the sliders feel useful.
  const ttftDataMax = Math.max(...__DATA__.measurements.map(r => r.p99_ttft_ms || 0), 100);
  const tpotDataMax = Math.max(...__DATA__.measurements.map(r => r.p99_tpot_ms || 0), 50);
  ttft.max = Math.ceil(ttftDataMax * 1.1);
  ttft.value = ttft.max;
  tpot.max = Math.ceil(tpotDataMax * 1.1);
  tpot.value = tpot.max;

  [ttft, tpot, wl].forEach(el => el.addEventListener('input', render));
  chips.forEach(c => c.addEventListener('change', render));
  render();
})();
"""


def _embed_scientific_figure(fig: Any, div_id: str | None = None, include_js: bool = False) -> str:
    if fig is None:
        return ""
    return _embed_figure(fig, include_js=include_js, div_id=div_id)


def _section_scientific(measurements: list[dict[str, Any]], rankings: list[dict[str, Any]], robustness: list[dict[str, Any]], slo: dict[str, float], include_js: bool) -> str:
    """The 'Decide' block: decision-supporting graphs at the top of the report.

    The decision hero itself is rendered separately (above the divider) by
    ``render_decision_hero``; this function only emits the divider and the
    supporting graphs that justify the recommendation.
    """
    best_by_backend = _best_config_per_backend(robustness)
    op_curve = _figure_operating_curve(measurements, best_by_backend, slo)
    lat_ttft = _figure_latency_vs_promptlen(measurements, best_by_backend, "p99_ttft_ms", "p99 TTFT")
    lat_tpot = _figure_latency_vs_promptlen(measurements, best_by_backend, "p99_tpot_ms", "p99 TPOT")
    decision_map = _figure_decision_map(measurements)
    sensitivity = _figure_tuning_sensitivity(measurements, rankings)

    chunks = ['<div class="divider"><h2 class="divider-title">Decide</h2><p class="divider-sub">The recommendation and how to verify it.</p></div>']

    if op_curve is not None:
        chunks.append(
            '<section><h3>Operating curve: throughput vs latency</h3>'
            '<p>Each line traces one backend\'s best config across increasing concurrency. The knee of the curve is where latency degrades faster than throughput grows. The dashed line is the SLO; the labeled point is the highest throughput each backend sustains while still meeting it.</p>'
            f'<div class="figure">{_embed_scientific_figure(op_curve, include_js=include_js, div_id="op-curve")}</div></section>'
        )
        include_js = False  # Plotly bundle now inlined

    if lat_ttft is not None or lat_tpot is not None:
        chunks.append('<section><h3>Latency scaling with prompt length</h3>'
                     '<p>Per-backend best config (line) with shaded min/max across all that backend\'s configs (band width = how much tuning matters). Conditioned on max_concurrency=1 to isolate prefill cost.</p>')
        if lat_ttft is not None:
            chunks.append(f'<div class="figure">{_embed_scientific_figure(lat_ttft, include_js=include_js)}</div>')
            include_js = False
        if lat_tpot is not None:
            chunks.append(f'<div class="figure">{_embed_scientific_figure(lat_tpot)}</div>')
        chunks.append('</section>')

    if decision_map is not None:
        chunks.append(
            '<section><h3>Decision map: who wins each workload</h3>'
            '<p>For every workload cell where both backends ran, the winner and the ratio over the loser. Cells with only one backend present are labeled as such; missing measurements are shown as <code>—</code>.</p>'
            f'<div class="figure">{_embed_scientific_figure(decision_map, include_js=include_js)}</div></section>'
        )
        include_js = False

    if sensitivity is not None:
        chunks.append(
            '<section><h3>Tuning sensitivity</h3>'
            '<p>One box per (backend × workload), populated with the throughput of every config of that backend at that workload. Tight + high box = backend is easy to tune well. Wide box = config choice dominates.</p>'
            f'<div class="figure">{_embed_scientific_figure(sensitivity, include_js=include_js)}</div></section>'
        )

    chunks.append('<div class="divider"><h2 class="divider-title">Explore</h2><p class="divider-sub">Design-space tools and per-candidate detail.</p></div>')
    return "".join(chunks)


def generate_html_report(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    measurements = _read_csv(result_dir / ("measurements.csv" if (result_dir / "measurements.csv").exists() else "normalized.csv"))
    rankings = _read_csv(result_dir / "rankings.csv")
    candidates = _read_csv(result_dir / "candidates.csv")
    failures = _read_csv(result_dir / "failures.csv")
    manifest: dict[str, Any] = {}
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}

    candidates_by_id = {str(c.get("candidate_id")): c for c in candidates}
    zero_floor = _load_zero_floor(result_dir)
    robustness = compute_robustness(rankings, candidates, zero_floor=zero_floor)
    robustness_by_cid = {r["candidate_id"]: r for r in robustness}
    workloads = sorted({_workload_key(r) for r in measurements}, key=lambda k: tuple("" if v is None else str(v) for v in k))
    slo = _load_slo(result_dir)
    objective = _load_objective_maximize(result_dir)
    # Top-K caps for the two long sections (issue 005). Both fall back to
    # sensible defaults when the YAML key is missing.
    leaderboard_rows = _load_leaderboard_rows(result_dir)
    top_k_cards = _load_top_k_candidate_cards(result_dir)

    pareto_html, pareto_includes_js = _section_pareto(measurements, candidates_by_id)
    # Conditional failure callout: only renders for degraded runs (returns "" for clean).
    # Anchored above the Decide divider so a reader sees the coverage gap before
    # acting on the recommendation. See moe_bench/failure_summary.py for the pure logic.
    failure_summary = compute_failure_summary(failures, candidates=candidates, workloads=workloads)
    # Merged Decision-summary hero: replaces the old backend-only `_section_decision_summary`
    # and the candidate-level `_section_global_winner`. The hero names both the backend
    # and the candidate to deploy and quantifies the win (wins / margin / worst-case).
    recommendation = compute_recommendation(
        rankings, candidates, slo=slo, objective=objective, zero_floor=zero_floor,
    )
    # Constraint-slider section is hidden when objective.constraints is empty
    # (no SLO to slide). See PRD "Degraded-state behavior".
    constraint_section = (
        _section_constraint_leaderboard(rankings, workloads) if slo else ""
    )
    sections = [
        _section_header(manifest, measurements, candidates, failures),
        render_failure_callout(failure_summary),
        render_decision_hero(recommendation),
        # "Decide" divider + supporting graphs that justify the recommendation.
        # The Plotly bundle is inlined inside this block on the first figure that successfully renders.
        _section_scientific(measurements, rankings, robustness, slo, include_js=True),
        _section_robustness_leaderboard(robustness, leaderboard_rows=leaderboard_rows),
        constraint_section,
        pareto_html,
        _section_parcoords(rankings, candidates, js_already_included=True),
        _section_per_workload_drilldown(rankings),
        _section_candidate_cards(
            measurements, rankings, candidates, robustness_by_cid, top_k=top_k_cards,
        ),
    ]

    live_json = json.dumps(
        _live_data(
            measurements, rankings, candidates_by_id,
            slo=slo, objective=objective, zero_floor=zero_floor,
        ),
        default=str,
    )
    js = JS_TEMPLATE % live_json

    title = f"moe-bench report — {manifest.get('run_id') or result_dir.name}"
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
{''.join(sections)}
</main>
<script>{js}</script>
</body>
</html>
"""
    out = result_dir / "report.html"
    out.write_text(html_doc, encoding="utf-8")
    return out
