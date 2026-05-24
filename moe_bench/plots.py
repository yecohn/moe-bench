from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_legend_params(result_dir: Path) -> list[str]:
    cfg_path = result_dir / "config.yaml"
    if not cfg_path.exists():
        return []
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return []
    return list((cfg.get("report") or {}).get("legend_params") or [])


def _candidate_label_map(result_dir: Path) -> dict[str, str]:
    try:
        import pandas as pd
        candidates = pd.read_csv(result_dir / "candidates.csv")
    except Exception:
        return {}
    legend_params = _load_legend_params(result_dir)
    labels: dict[str, str] = {}
    for _, row in candidates.iterrows():
        cid = str(row.get("candidate_id"))
        parts = [str(row.get("backend", ""))]
        if legend_params:
            backend = str(row.get("backend", ""))
            for p in legend_params:
                candidates = [p, f"{backend}_{p}", f"param_{p}"]
                col = next((c for c in candidates if c in row), None)
                val = row.get(col, "") if col else ""
                if str(val) not in {"", "nan", "None"}:
                    parts.append(f"{p}={val}")
        else:
            parts.append(str(row.get("serve_config", cid)))
        labels[cid] = " ".join(parts)
    return labels


def _add_plot_label(df: Any, labels: dict[str, str]) -> Any:
    if "candidate_id" in df:
        df["plot_label"] = df["candidate_id"].astype(str).map(labels)
    if "plot_label" not in df or df["plot_label"].isna().all():
        df["plot_label"] = df.get("serve_config", df.get("backend", "candidate"))
    df["plot_label"] = df["plot_label"].fillna(df.get("serve_config", "candidate"))
    return df


def _numeric(df: Any, cols: list[str]) -> Any:
    import pandas as pd
    for col in cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _save(fig: Any, path: Path, written: list[Path]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.clf()
    written.append(path)


def make_plots(result_dir: str | Path) -> list[Path]:
    """Create decision-oriented plots.

    The goal is not to visualize every raw metric. The plots should answer:
    1. Which candidate wins each workload constraint?
    2. What are the throughput/latency tradeoffs?
    3. Which candidates/backends fail?
    """
    result_dir = Path(result_dir)
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception as exc:
        (plots_dir / "PLOTS_SKIPPED.txt").write_text(f"Plot dependencies unavailable: {exc}\n")
        return []

    measurements_path = result_dir / ("measurements.csv" if (result_dir / "measurements.csv").exists() else "normalized.csv")
    rankings_path = result_dir / "rankings.csv"
    if not measurements_path.exists():
        return []

    meas = pd.read_csv(measurements_path)
    ranks = pd.read_csv(rankings_path) if rankings_path.exists() else pd.DataFrame()
    if meas.empty:
        return []

    labels = _candidate_label_map(result_dir)
    meas = _add_plot_label(meas, labels)
    if not ranks.empty:
        ranks = _add_plot_label(ranks, labels)

    metric_cols = [
        "input_len", "output_len", "max_concurrency", "rank", "objective_value",
        "output_tok_s_per_gpu", "p99_ttft_ms", "p99_tpot_ms",
        "median_output_tok_s_per_gpu", "median_p99_ttft_ms", "median_p99_tpot_ms",
    ]
    meas = _numeric(meas, metric_cols)
    if not ranks.empty:
        ranks = _numeric(ranks, metric_cols)

    valid = meas[meas.get("valid", False).astype(str).str.lower().isin(["true", "1"])] if "valid" in meas else meas
    sns.set_theme(style="whitegrid")
    written: list[Path] = []

    # 1. Top-K candidate bars per workload: the primary decision plot.
    if not ranks.empty and {"rank", "workload", "objective_value", "plot_label"}.issubset(ranks.columns):
        top = ranks[ranks["rank"] <= 5].copy()
        if not top.empty:
            fig = plt.figure(figsize=(max(11, top["workload"].nunique() * 1.0), 6))
            ax = sns.barplot(data=top, x="workload", y="objective_value", hue="plot_label")
            ax.set_title("Top-K server candidates by objective for each workload")
            ax.set_ylabel("Objective value (usually output tok/s/GPU)")
            ax.set_xlabel("Workload constraint")
            ax.tick_params(axis="x", rotation=60)
            _save(fig, plots_dir / "topk_candidates_by_workload.png", written)

    # 2. Winner throughput by workload, colored by backend.
    if not ranks.empty and {"rank", "workload", "objective_value", "backend"}.issubset(ranks.columns):
        best = ranks[ranks["rank"] == 1].copy()
        if not best.empty:
            fig = plt.figure(figsize=(max(10, best["workload"].nunique() * 0.8), 5.5))
            ax = sns.barplot(data=best, x="workload", y="objective_value", hue="backend")
            ax.set_title("Winning candidate throughput per workload")
            ax.set_ylabel("Best output tok/s/GPU")
            ax.set_xlabel("Workload constraint")
            ax.tick_params(axis="x", rotation=60)
            _save(fig, plots_dir / "winning_throughput_by_workload.png", written)

    # 3. Latency of the selected winner: catches high-throughput but bad-latency winners.
    if not ranks.empty and {"rank", "workload", "median_p99_ttft_ms", "median_p99_tpot_ms"}.issubset(ranks.columns):
        best = ranks[ranks["rank"] == 1].copy()
        if not best.empty:
            melt = best.melt(
                id_vars=["workload"],
                value_vars=["median_p99_ttft_ms", "median_p99_tpot_ms"],
                var_name="latency_metric",
                value_name="ms",
            ).dropna(subset=["ms"])
            if not melt.empty:
                fig = plt.figure(figsize=(max(10, best["workload"].nunique() * 0.8), 5.5))
                ax = sns.barplot(data=melt, x="workload", y="ms", hue="latency_metric")
                ax.set_title("p99 latency of winning candidate per workload")
                ax.set_ylabel("milliseconds")
                ax.set_xlabel("Workload constraint")
                ax.tick_params(axis="x", rotation=60)
                _save(fig, plots_dir / "winner_latency_by_workload.png", written)

    # 4. Pareto tradeoff plots across all valid measurements.
    if not valid.empty and {"output_tok_s_per_gpu", "p99_ttft_ms", "plot_label", "backend"}.issubset(valid.columns):
        fig = plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(data=valid, x="p99_ttft_ms", y="output_tok_s_per_gpu", hue="plot_label", style="backend", size="max_concurrency" if "max_concurrency" in valid else None, sizes=(50, 220))
        ax.set_title("Pareto tradeoff: throughput vs p99 TTFT")
        ax.set_xlabel("p99 TTFT (ms), lower is better")
        ax.set_ylabel("Output tok/s/GPU, higher is better")
        _save(fig, plots_dir / "pareto_throughput_vs_ttft.png", written)

    if not valid.empty and {"output_tok_s_per_gpu", "p99_tpot_ms", "plot_label", "backend"}.issubset(valid.columns):
        fig = plt.figure(figsize=(10, 7))
        ax = sns.scatterplot(data=valid, x="p99_tpot_ms", y="output_tok_s_per_gpu", hue="plot_label", style="backend", size="max_concurrency" if "max_concurrency" in valid else None, sizes=(50, 220))
        ax.set_title("Pareto tradeoff: throughput vs p99 TPOT")
        ax.set_xlabel("p99 TPOT (ms), lower is better")
        ax.set_ylabel("Output tok/s/GPU, higher is better")
        _save(fig, plots_dir / "pareto_throughput_vs_tpot.png", written)

    # 5. Workload-grid heatmaps for the selected winner when the workload is a grid.
    if not ranks.empty and {"rank", "input_len", "max_concurrency", "output_len"}.issubset(ranks.columns):
        best = ranks[ranks["rank"] == 1].copy()
        for output_len, sub in best.groupby("output_len", dropna=False):
            if sub["input_len"].notna().sum() and sub["max_concurrency"].notna().sum():
                for metric, title, filename in [
                    ("objective_value", "Best output tok/s/GPU", "best_throughput_heatmap"),
                    ("median_p99_ttft_ms", "Winner p99 TTFT (ms)", "winner_p99_ttft_heatmap"),
                    ("median_p99_tpot_ms", "Winner p99 TPOT (ms)", "winner_p99_tpot_heatmap"),
                ]:
                    if metric not in sub or sub[metric].dropna().empty:
                        continue
                    pivot = sub.pivot_table(index="max_concurrency", columns="input_len", values=metric, aggfunc="max")
                    if pivot.empty:
                        continue
                    fig = plt.figure(figsize=(max(6, len(pivot.columns) * 1.1), max(4, len(pivot.index) * 0.8)))
                    ax = sns.heatmap(pivot, annot=True, fmt=".1f", cbar=True)
                    ax.set_title(f"{title} (output_len={output_len})")
                    ax.set_xlabel("input_len")
                    ax.set_ylabel("max_concurrency")
                    safe_out = str(output_len).replace(".", "_")
                    _save(fig, plots_dir / f"{filename}_out{safe_out}.png", written)

    # 6. Backend win counts: useful when many workload constraints exist.
    if not ranks.empty and {"rank", "backend"}.issubset(ranks.columns):
        best = ranks[ranks["rank"] == 1]
        if not best.empty:
            fig = plt.figure(figsize=(7, 4.5))
            ax = sns.countplot(data=best, x="backend")
            ax.set_title("Backend win count across workload constraints")
            ax.set_ylabel("# workload constraints won")
            _save(fig, plots_dir / "backend_win_counts.png", written)

    # 7. Coverage/failure heatmap by candidate/workload.
    if {"candidate_id", "workload", "valid"}.issubset(meas.columns):
        meas["coverage_valid"] = meas["valid"].astype(str).str.lower().isin(["true", "1"])
        cov = meas.pivot_table(index=["backend", "candidate_id"], columns="workload", values="coverage_valid", aggfunc=lambda s: int(sum(bool(x) for x in s)), fill_value=0)
        if not cov.empty:
            fig = plt.figure(figsize=(max(10, len(cov.columns) * 0.35), max(4, len(cov.index) * 0.4)))
            ax = sns.heatmap(cov, annot=False, cbar=True)
            ax.set_title("Valid measurement coverage by candidate/workload")
            _save(fig, plots_dir / "coverage_failure_heatmap.png", written)

    # 8. Backend ratio: best SGLang candidate vs best vLLM candidate per workload.
    if {"backend", "workload", "output_tok_s_per_gpu"}.issubset(valid.columns) and valid["backend"].nunique() >= 2:
        best_backend = valid.sort_values("output_tok_s_per_gpu", ascending=False).groupby(["backend", "workload"], as_index=False).first()
        pivot = best_backend.pivot(index="workload", columns="backend", values="output_tok_s_per_gpu")
        if {"vllm", "sglang"}.issubset(set(pivot.columns)):
            ratio = (pivot["sglang"] / pivot["vllm"]).dropna().reset_index(name="sglang_over_vllm")
            if not ratio.empty:
                fig = plt.figure(figsize=(max(10, len(ratio) * 0.5), 5))
                ax = sns.barplot(data=ratio, x="workload", y="sglang_over_vllm")
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
                ax.set_title("Best SGLang throughput / best vLLM throughput by workload")
                ax.set_ylabel("ratio >1 means SGLang higher throughput")
                ax.tick_params(axis="x", rotation=60)
                _save(fig, plots_dir / "backend_best_throughput_ratio.png", written)

    return written
