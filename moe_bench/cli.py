from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from .config import expand_serve_configs, load_config
from .normalize import import_legacy, normalize_run
from .rank import rank_run
from .report import generate_report
from .runner import run_sweep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moe-bench", description="Small vLLM/SGLang inference benchmark platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a configured sweep")
    run.add_argument("config", type=Path)
    run.add_argument("--out", default="results", help="results root directory")
    run.add_argument("--backends", default=None, help="comma-separated backend names")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--report", action="store_true", help="normalize, rank, and report after run")
    run.add_argument("--plots", action="store_true", help="generate plot PNGs with --report (off by default)")
    run.add_argument("--no-html", action="store_true", help="skip interactive HTML report (on by default with --report)")
    run.add_argument("--model", default=None, help="override experiment.model")
    run.add_argument("--run-id", default=None, help="override experiment.run_id and experiment.name")
    run.add_argument("--served-model-name", default=None, help="override backends.vllm.served_model_name")
    run.add_argument("--dtype", default=None, help="override experiment.dtype and every per-candidate dtype field")

    norm = sub.add_parser("normalize", help="raw results -> normalized.csv/failures.csv")
    norm.add_argument("result_dir", type=Path)
    norm.add_argument("--gpus", type=int, default=None)

    rank = sub.add_parser("rank", help="normalized.csv -> rankings.csv")
    rank.add_argument("result_dir", type=Path)

    short = sub.add_parser("shortlist", help="write a new config with top-K ranked server candidates")
    short.add_argument("result_dir", type=Path)
    short.add_argument("--top-k", type=int, default=5)
    short.add_argument("--out", required=True, type=Path)
    short.add_argument("--repeats", type=int, default=None, help="override execution.repeats in the output config")

    rep = sub.add_parser("report", help="generate report.md and report.html; PNG plots are optional")
    rep.add_argument("result_dir", type=Path)
    rep.add_argument("--plots", action="store_true", help="generate PNG plots")
    rep.add_argument("--no-plots", action="store_true", help="deprecated; PNG plots are off by default")
    rep.add_argument("--no-html", action="store_true", help="skip interactive HTML report")

    gen = sub.add_parser("generate", help="expand search_space into explicit serve_configs")
    gen.add_argument("config", type=Path)
    gen.add_argument("--out", required=True, type=Path)

    imp = sub.add_parser("import-legacy", help="import historical vLLM/SGLang sweep outputs into normalized.csv")
    imp.add_argument("--vllm", default=None, help="path to vLLM moe_sweep results")
    imp.add_argument("--sglang", default=None, help="path to SGLang moe_sweep results")
    imp.add_argument("--out", required=True, type=Path)
    imp.add_argument("--gpus", type=int, default=4)
    imp.add_argument("--report", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "run":
        result_dir = run_sweep(
            args.config,
            args.out,
            only_backends=args.backends,
            dry=args.dry_run,
            model=args.model,
            run_id=args.run_id,
            served_model_name=args.served_model_name,
            dtype=args.dtype,
        )
        if args.dry_run:
            return 0
        if args.report:
            normalize_run(result_dir)
            rank_run(result_dir)
            report_path = generate_report(result_dir, make_plot_files=args.plots, make_html=not args.no_html)
            print(report_path)
        return 0
    if args.cmd == "normalize":
        print(normalize_run(args.result_dir, gpus=args.gpus))
        return 0
    if args.cmd == "rank":
        print(rank_run(args.result_dir))
        return 0
    if args.cmd == "shortlist":
        cfg = load_config(args.result_dir / "config.yaml")
        candidates = {}
        with (args.result_dir / "candidates.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                candidates[row["candidate_id"]] = row
        chosen = []
        seen = set()
        with (args.result_dir / "rankings.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(float(row.get("rank") or 0)) > args.top_k:
                    continue
                cid = row.get("candidate_id")
                if not cid or cid in seen or cid not in candidates:
                    continue
                seen.add(cid)
                cand = candidates[cid]
                backend = cand["backend"]
                prefix = f"{backend}_"
                params = {
                    k.removeprefix(prefix): v
                    for k, v in cand.items()
                    if k.startswith(prefix)
                    and str(v) not in {"", "nan", "None", "default", "auto", "none"}
                }
                # Convert common scalar strings back to useful YAML types.
                for k, v in list(params.items()):
                    if v in {"True", "False"}:
                        params[k] = v == "True"
                    else:
                        try:
                            params[k] = int(v) if str(v).isdigit() else float(v)
                        except Exception:
                            params[k] = v
                chosen.append({"name": cand.get("serve_config") or f"candidate_{cid}", backend: params})
        cfg = dict(cfg)
        cfg["serve_configs"] = chosen
        cfg.pop("search_space", None)
        if args.repeats is not None:
            cfg.setdefault("execution", {})["repeats"] = args.repeats
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(args.out)
        return 0
    if args.cmd == "report":
        print(generate_report(args.result_dir, make_plot_files=args.plots and not args.no_plots, make_html=not args.no_html))
        return 0
    if args.cmd == "generate":
        cfg = load_config(args.config)
        expanded = dict(cfg)
        expanded["serve_configs"] = expand_serve_configs(cfg)
        expanded.pop("search_space", None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(expanded, sort_keys=False), encoding="utf-8")
        print(args.out)
        return 0
    if args.cmd == "import-legacy":
        out = import_legacy(args.vllm, args.sglang, args.out, gpus=args.gpus)
        print(out / "normalized.csv")
        if args.report:
            rank_run(out)
            print(generate_report(out))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
