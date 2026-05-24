from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _num(v: Any) -> float | None:
    if v in {None, "", "None", "nan"}:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _bool(v: Any) -> bool:
    return str(v).lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if len(values) == 1 else None


def load_objective(result_dir: Path) -> dict[str, Any]:
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return {"maximize": "output_tok_s_per_gpu", "constraints": {}}
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("objective") or {"maximize": "output_tok_s_per_gpu", "constraints": {}}


def passes_constraints(row: dict[str, Any], constraints: dict[str, Any]) -> tuple[bool, str]:
    for metric, limit in (constraints or {}).items():
        if limit is None or limit == "":
            continue
        val = _num(row.get(metric) or row.get(f"median_{metric}"))
        if val is None:
            return False, f"missing_constraint_metric:{metric}"
        if val > float(limit):
            return False, f"constraint_failed:{metric}>{limit}"
    return True, ""


def compute_goodput_at_slo(row: dict[str, Any], constraints: dict[str, Any]) -> float | None:
    """Throughput conditional on meeting all p99 SLOs.

    Differs from passes_constraints + filtering: candidates that violate an SLO
    get goodput=0 instead of disappearing from the table. Lets the user see
    "fast but tail-broken" candidates instead of silently dropping them.

    Uses median_output_tok_s_per_gpu as the underlying throughput. Returns None
    if throughput itself is missing (genuinely no measurement, not an SLO miss).
    """
    tput = _num(row.get("median_output_tok_s_per_gpu"))
    if tput is None:
        return None
    for metric, limit in (constraints or {}).items():
        if limit is None or limit == "":
            continue
        val = _num(row.get(metric) or row.get(f"median_{metric}"))
        if val is None or val > float(limit):
            return 0.0
    return tput


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("candidate_id"), row.get("server_state_sha"), row.get("backend"), row.get("serve_config"), row.get("serve_params_json"), row.get("workload"), row.get("input_len"), row.get("output_len"), row.get("max_concurrency"), row.get("request_rate"))
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    metrics = ["output_tok_s_per_gpu", "total_tok_s_per_gpu", "request_throughput", "p99_ttft_ms", "p99_tpot_ms", "mean_ttft_ms", "mean_tpot_ms"]
    for key, group in groups.items():
        valid = [r for r in group if _bool(r.get("valid"))]
        base = {
            "candidate_id": key[0], "server_state_sha": key[1], "backend": key[2], "serve_config": key[3], "serve_params_json": key[4], "workload": key[5], "input_len": key[6], "output_len": key[7], "max_concurrency": key[8], "request_rate": key[9],
            "valid_repeats": len(valid), "failed_repeats": len(group) - len(valid), "total_repeats": len(group),
        }
        for m in metrics:
            vals = [_num(r.get(m)) for r in valid]
            vals = [v for v in vals if v is not None]
            base[f"median_{m}"] = median(vals)
            base[f"mean_{m}"] = mean(vals)
            base[f"std_{m}"] = stdev(vals)
            if m == "output_tok_s_per_gpu" and mean(vals):
                base["cv_output_tok_s_per_gpu"] = (stdev(vals) or 0.0) / (mean(vals) or 1.0)
        out.append(base)
    return out


def rank_run(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    rows = read_csv(result_dir / ("measurements.csv" if (result_dir / "measurements.csv").exists() else "normalized.csv"))
    objective = load_objective(result_dir)
    maximize = objective.get("maximize", "output_tok_s_per_gpu")
    constraints = objective.get("constraints") or {}
    agg = aggregate_rows(rows)
    # goodput_at_slo is a derived per-row metric. Compute eagerly so it's
    # available both as an objective and as a sortable column in reports.
    for row in agg:
        row["goodput_at_slo"] = compute_goodput_at_slo(row, constraints)
        row["median_goodput_at_slo"] = row["goodput_at_slo"]

    # When the objective IS goodput_at_slo, the constraints are already baked
    # into the metric value (violators get 0), so we keep them in the table
    # instead of pre-filtering them out. Otherwise filter as before.
    pre_filter = (maximize != "goodput_at_slo")
    by_workload: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in agg:
        ok, reason = passes_constraints(row, constraints)
        row["passes_constraints"] = ok
        row["constraint_reason"] = reason
        if not row["valid_repeats"]:
            continue
        if pre_filter and not ok:
            continue
        by_workload[(row.get("workload"), row.get("input_len"), row.get("output_len"), row.get("max_concurrency"), row.get("request_rate"))].append(row)

    ranked: list[dict[str, Any]] = []
    metric_col = f"median_{maximize}" if maximize != "goodput_at_slo" else "median_goodput_at_slo"
    for workload_key, candidates in sorted(by_workload.items(), key=lambda kv: str(kv[0])):
        candidates.sort(key=lambda r: _num(r.get(metric_col)) if _num(r.get(metric_col)) is not None else float("-inf"), reverse=True)
        best = _num(candidates[0].get(metric_col)) if candidates else None
        for idx, row in enumerate(candidates, start=1):
            item = dict(row)
            item["rank"] = idx
            item["objective_metric"] = maximize
            item["objective_value"] = row.get(metric_col)
            val = _num(row.get(metric_col))
            item["relative_to_best"] = (val / best) if val is not None and best else None
            ranked.append(item)

    fieldnames = [
        "rank", "candidate_id", "server_state_sha", "backend", "serve_config", "serve_params_json", "workload", "input_len", "output_len", "max_concurrency", "request_rate",
        "objective_metric", "objective_value", "relative_to_best", "valid_repeats", "failed_repeats", "total_repeats", "passes_constraints", "constraint_reason",
        "median_goodput_at_slo",
        "median_output_tok_s_per_gpu", "mean_output_tok_s_per_gpu", "std_output_tok_s_per_gpu", "cv_output_tok_s_per_gpu",
        "median_p99_ttft_ms", "mean_p99_ttft_ms", "median_p99_tpot_ms", "mean_p99_tpot_ms", "median_request_throughput",
    ]
    with (result_dir / "rankings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in ranked:
            w.writerow(row)
    return result_dir / "rankings.csv"
