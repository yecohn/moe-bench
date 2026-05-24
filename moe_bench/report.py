from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .html_report import generate_html_report
from .plots import make_plots


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int = 20) -> str:
    rows = rows[:limit]
    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, str):
                vals.append(v[:80])
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def load_config(result_dir: Path) -> dict[str, Any]:
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compact_json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


def serve_config_rows_from_config(cfg: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for serve in cfg.get("serve_configs", []) or []:
        name = str(serve.get("name", "unknown"))
        for backend in [k for k in serve.keys() if k != "name" and not k.startswith("_")]:
            rows.append({
                "serve_config": name,
                "backend": backend,
                "params": compact_json(serve.get(backend)),
            })
    return rows


def serve_config_rows_from_normalized(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("candidate_id"), row.get("serve_config"), row.get("backend"), row.get("serve_params_json"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"candidate_id": str(row.get("candidate_id", "")), "serve_config": str(row.get("serve_config", "")), "backend": str(row.get("backend", "")), "params": str(row.get("serve_params_json", ""))})
    return out


def serve_config_rows_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for c in candidates:
        backend = str(c.get("backend", ""))
        prefix = f"{backend}_"
        params = {
            k.removeprefix(prefix): v
            for k, v in c.items()
            if k.startswith(prefix) and str(v) not in {"", "nan", "None"}
        }
        rows.append({
            "candidate_id": str(c.get("candidate_id", "")),
            "server_state_sha": str(c.get("server_state_sha", "")),
            "serve_config": str(c.get("serve_config", "")),
            "backend": backend,
            "params": json.dumps(params, sort_keys=True, default=str),
        })
    return rows


def first_server_cmd(rows: list[dict[str, Any]], serve_config: str, backend: str) -> str:
    for row in rows:
        if row.get("serve_config") == serve_config and row.get("backend") == backend and row.get("server_cmd"):
            return str(row.get("server_cmd"))
    return ""


def append_serve_config_section(md: list[str], serve_rows: list[dict[str, str]], normalized: list[dict[str, Any]]) -> None:
    md += ["## Optimization candidates / server parameters", ""]
    md += ["Each candidate is one backend-specific `server_params_json` vector. Workload fields like input length, output length, and concurrency are constraints measured against these candidates.", ""]
    if not serve_rows:
        md.append("_No server parameter config found._")
        md.append("")
        return
    md.append(md_table([{ "candidate_id": r.get("candidate_id", ""), "server_state_sha": r.get("server_state_sha", ""), "serve_config": r["serve_config"], "backend": r["backend"] } for r in serve_rows], ["candidate_id", "server_state_sha", "serve_config", "backend"], limit=200))
    md.append("")
    for row in serve_rows:
        serve_config = row["serve_config"]
        backend = row["backend"]
        params = row.get("params") or "{}"
        try:
            params = json.dumps(json.loads(params), indent=2, sort_keys=True)
        except Exception:
            pass
        cid = row.get("candidate_id") or ""
        title = f"`{cid}` / `{serve_config}` / `{backend}`" if cid else f"`{serve_config}` / `{backend}`"
        md += [f"### {title}", "", "```json", params, "```", ""]
        cmd = first_server_cmd(normalized, serve_config, backend)
        if cmd:
            md += ["Server command example from run:", "", "```bash", cmd, "```", ""]


def _float(value: Any) -> float | None:
    try:
        if value in {None, "", "None", "nan"}:
            return None
        return float(value)
    except Exception:
        return None


def parameter_sensitivity_rows(normalized: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate = {c.get("candidate_id"): c for c in candidates}
    rows: list[dict[str, Any]] = []
    backends = sorted({str(c.get("backend")) for c in candidates})
    for backend in backends:
        backend_candidates = [c for c in candidates if str(c.get("backend")) == backend]
        prefix = f"{backend}_"
        param_keys = sorted({k for c in backend_candidates for k in c if k.startswith(prefix)})
        for key in param_keys:
            # Sensitivity should compare values within one backend. Missing values
            # usually mean "parameter does not exist for this backend", not a real value.
            values = {str(c.get(key)) for c in backend_candidates if str(c.get(key, "")) != ""}
            if len(values) <= 1:
                continue
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in normalized:
                cand = by_candidate.get(row.get("candidate_id"), {})
                if str(cand.get("backend")) != backend or str(cand.get(key, "")) == "":
                    continue
                value = str(cand.get(key))
                grouped.setdefault(value, []).append(row)
            for value, group in grouped.items():
                valid = [r for r in group if str(r.get("valid")).lower() in {"true", "1"}]
                vals = [_float(r.get("output_tok_s_per_gpu")) for r in valid]
                vals = [v for v in vals if v is not None]
                ttft = [_float(r.get("p99_ttft_ms")) for r in valid]
                ttft = [v for v in ttft if v is not None]
                tpot = [_float(r.get("p99_tpot_ms")) for r in valid]
                tpot = [v for v in tpot if v is not None]
                rows.append({
                    "backend": backend,
                    "parameter": key.removeprefix(prefix),
                    "value": value,
                    "rows": len(group),
                    "valid_rate": round(len(valid) / len(group), 3) if group else 0,
                    "avg_output_tok_s_per_gpu": round(sum(vals) / len(vals), 3) if vals else "",
                    "avg_p99_ttft_ms": round(sum(ttft) / len(ttft), 3) if ttft else "",
                    "avg_p99_tpot_ms": round(sum(tpot) / len(tpot), 3) if tpot else "",
                })
    return rows


def generate_report(result_dir: str | Path, make_plot_files: bool = True, make_html: bool = True) -> Path:
    result_dir = Path(result_dir)
    normalized = read_csv(result_dir / ("measurements.csv" if (result_dir / "measurements.csv").exists() else "normalized.csv"))
    rankings = read_csv(result_dir / "rankings.csv")
    failures = read_csv(result_dir / "failures.csv")
    candidates = read_csv(result_dir / "candidates.csv")
    manifest = {}
    if (result_dir / "manifest.json").exists():
        manifest = json.loads((result_dir / "manifest.json").read_text())
    cfg = load_config(result_dir)
    plots = make_plots(result_dir) if make_plot_files else []
    html_path: Path | None = None
    if make_html:
        try:
            html_path = generate_html_report(result_dir)
        except Exception as exc:
            (result_dir / "HTML_REPORT_SKIPPED.txt").write_text(f"HTML report generation failed: {exc}\n")

    valid = [r for r in normalized if str(r.get("valid")).lower() in {"true", "1"}]
    rank1 = [r for r in rankings if str(r.get("rank")) == "1"]
    winner_counts = Counter(f"{r.get('candidate_id')}/{r.get('backend')}/{r.get('serve_config')}" for r in rank1)
    backend_counts = Counter(r.get("backend") for r in rank1)

    md: list[str] = []
    md += ["# MoE benchmark report", ""]
    md += [f"Run ID: `{manifest.get('run_id', result_dir.name)}`"]
    if manifest.get("model"):
        md += [f"Model: `{manifest.get('model')}`"]
    if manifest.get("gpus"):
        md += [f"GPUs: `{manifest.get('gpus')}`"]
    md += [""]

    md += ["## Summary", ""]
    md += [f"- Total rows: {len(normalized)}"]
    md += [f"- Valid rows: {len(valid)}"]
    md += [f"- Failed/invalid rows: {len(failures)}"]
    md += [f"- Ranked workload winners: {len(rank1)}"]
    md += [""]

    serve_rows = serve_config_rows_from_candidates(candidates) or serve_config_rows_from_config(cfg) or serve_config_rows_from_normalized(normalized)
    append_serve_config_section(md, serve_rows, normalized)

    md += ["## Overall winners", ""]
    if winner_counts:
        rows = [{"backend_config": k, "wins": v} for k, v in winner_counts.most_common()]
        md.append(md_table(rows, ["backend_config", "wins"], limit=50))
    else:
        md.append("_No winners. Run `moe-bench rank` first or check failures._")
    md += [""]

    if backend_counts:
        md += ["## Backend win counts", ""]
        md.append(md_table([{"backend": k, "wins": v} for k, v in backend_counts.most_common()], ["backend", "wins"]))
        md += [""]

    md += ["## Best candidate per workload constraint", ""]
    md.append(md_table(rank1, ["workload", "input_len", "output_len", "max_concurrency", "candidate_id", "backend", "serve_config", "objective_value", "median_p99_ttft_ms", "median_p99_tpot_ms", "valid_repeats", "failed_repeats"], limit=100))
    md += ["", "The full backend-specific server parameters for each `candidate_id` are listed in the **Optimization candidates / server parameters** section above."]
    md += [""]

    md += ["## Candidate detail files", "", "- `candidates.csv`: curated flat table with the top tunable server parameters for each backend, default-filled where omitted, plus `server_cmd`.", "- `measurements.csv`: every candidate × workload measurement, including `candidate_id` and `server_state_sha`.", ""]

    md += ["## Top-K candidates per workload", ""]
    md.append(md_table(rankings, ["rank", "workload", "candidate_id", "backend", "serve_config", "objective_value", "relative_to_best", "median_p99_ttft_ms", "median_p99_tpot_ms"], limit=80))
    md += [""]

    md += ["## Parameter sensitivity", ""]
    sens = parameter_sensitivity_rows(normalized, candidates)
    if sens:
        md.append(md_table(sens, ["backend", "parameter", "value", "rows", "valid_rate", "avg_output_tok_s_per_gpu", "avg_p99_ttft_ms", "avg_p99_tpot_ms"], limit=200))
    else:
        md.append("_No varying scalar server parameters found across candidates._")
    md += [""]

    md += ["## Failure analysis", ""]
    if failures:
        reasons = Counter(r.get("failure_reason") or "unknown" for r in failures)
        md.append(md_table([{"failure_reason": k, "count": v} for k, v in reasons.most_common()], ["failure_reason", "count"], limit=30))
        md += ["", "### Sample failures", ""]
        md.append(md_table(failures, ["backend", "serve_config", "workload", "failure_reason", "source_path"], limit=20))
    else:
        md.append("_No failures recorded._")
    md += [""]

    if html_path is not None:
        md += ["## Interactive report", ""]
        md += [f"Open [`{html_path.name}`]({html_path.relative_to(result_dir)}) for the interactive view with constraint sliders, Pareto explorer, and per-candidate drill-down.", ""]

    md += ["## Plots", ""]
    if plots:
        for p in plots:
            rel = p.relative_to(result_dir)
            md.append(f"![{p.name}]({rel})")
            md.append("")
    else:
        md.append("_No plots generated._")

    path = result_dir / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
