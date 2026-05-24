# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`moe-bench` is a small benchmark platform that sweeps server-parameter candidates for vLLM and SGLang against a grid of workloads (`input_len × output_len × concurrency × repeats`) and ranks them by an objective (default: `output_tok_s_per_gpu` per workload). It does not implement inference; it spawns the real backend servers and the backends' own benchmark clients.

See `README.md` for the user-facing tour and `BACKENDS.md` for backend-env setup details.

## Environments — two-tier model

The platform env and the backend envs are deliberately separate:

- **Platform env** runs `moe-bench` itself. Only needs `PyYAML`, `pandas`, `matplotlib`, `seaborn` (see `pyproject.toml`). Typically the repo's `.venv` or an existing project venv (e.g. `/mnt/projects/AI/josh/aiq/.venv`).
- **Backend envs** live under `.backends/vllm/` and `.backends/sglang/` (built by `scripts/create_backend_venvs.sh`), or point at existing repo venvs like `/mnt/projects/AI/josh/vllm/.venv/`. Each backend's Python interpreter is selected per-config via `backends.<name>.python` in YAML.

When debugging "module not found" / version errors, always check which interpreter the config's `python:` field points at — `moe_bench/runner.py` shells out to that exact path. Don't assume the platform env has vLLM/SGLang installed; it shouldn't.

## Pipeline (cli → runner → normalize → rank → report)

`moe_bench/cli.py` is the single entry point (`moe-bench` console script). Subcommands map to modules:

- `run` → `runner.run_sweep`: copies config, writes `manifest.json`, then loops `backend × serve_config × workload`. For each `(backend, serve_config)`:
  1. `backends.make_backend` builds the server command from the YAML by kebab-casing keys (e.g. `tensor_parallel_size` → `--tensor-parallel-size`). Bools become `--flag` or `--no-flag`. `extra_args` (also accepted: `args`, `raw_args`, `server_args`) is appended verbatim.
  2. Server is launched as a process group via `subprocess.Popen(..., start_new_session=True)`, `utils.wait_ready` polls `/v1/models` (vLLM) or the backend's ready endpoint until `server_ready_timeout_sec`.
  3. For each workload, the backend's own bench client runs (`vllm bench serve` for vLLM, `sglang.bench_serving` for SGLang) up to `bench_timeout_sec`. SGLang writes JSONL; `SglangBackend.parse_result_path` materializes `result.json` from the last line.
  4. `status.json` per workload encodes one of: `ready`, `server_ready_timeout`, `bench_timeout`, `oom` (detected via `utils.detect_oom` log scan), `bench_failed`, `missing_result`. With `execution.resume: true` (default), existing `status.json` + `result.json` cause the cell to be skipped — this is how mid-sweep restarts work.
  5. `utils.terminate_tree` SIGTERMs the whole process group, then SIGKILLs after `grace_sec`. Always relies on `start_new_session=True` to kill children too. `cool_down_sec` waits between candidates.
- `normalize` → `normalize.normalize_run`: walks `raw/<backend>/<serve_config>/<workload>/result.json`, maps backend-specific keys to `CANONICAL_FIELDS`, writes `measurements.csv` (with `normalized.csv` as a compatibility alias) and `candidates.csv` (one row per candidate, curated tuning params with backend-prefixed columns).
- `rank` → `rank.rank_run`: groups by `(candidate_id, workload_key)`, aggregates median/mean/std/cv across repeats, applies `objective.constraints` (e.g. `p99_ttft_ms`), and ranks per workload. Output: `rankings.csv`.
- `report` → `report.generate_report`: writes `report.md` plus an interactive `report.html` (built by `html_report.py`). HTML is on by default — pass `--no-html` to skip. PNG plots (`plots.py`) are off by default — pass `--plots`. The HTML report is self-contained (inlines Plotly + embeds measurement JSON for the constraint-slider leaderboard); just open it in a browser.
- `shortlist`: reads `rankings.csv` + `candidates.csv`, rebuilds typed YAML `serve_configs` for the top-K candidates (drops empty/`default`/`auto` columns, re-types bools/ints/floats) and writes a validation config. Strips `search_space` from the output.
- `generate`: pre-expands `search_space` grids into explicit `serve_configs` for inspection without running.
- `import-legacy`: ingests historical vLLM/SGLang sweep dirs and writes a `normalized.csv` so old data can use the same `rank`/`report` pipeline.

## Config model (`moe_bench/config.py`)

Required keys: `experiment`, `backends`, `workloads`, and either `serve_configs` or `search_space`. Validation happens in `validate_config` on load.

- **`serve_configs`** is a list. Each item has `name:` plus a backend sub-dict (e.g. `vllm: {...}`, `sglang: {...}`). A candidate runs against backend `X` only if it contains a `X:` sub-dict — that's how the same list mixes vLLM-only, SGLang-only, and shared candidates.
- **`search_space`** is shorthand: `search_space.<backend>: { defaults: {...}, grid: {param: [values, ...]} }`. `_expand_backend_search_space` produces one generated `serve_config` per Cartesian-product point; generated names are `<backend>_<candidate_id>` so they're collision-free.
- **`candidate_id`** is `sha1(json({backend, params}))[:12]`. It's the stable identity of a server-parameter vector across runs and is the join key in CSVs (`measurements.csv`, `rankings.csv`, `candidates.csv`).
- **`workloads`** must include `input_lens`, `output_lens`, `concurrencies`. `expand_workloads` multiplies by `seeds` and `execution.repeats` and produces workload names like `prompt256_out128_conc1_seed0_rep0` that `normalize.workload_from_name` can parse back.

## Results directory (per run)

```
results/<run_id>/
  config.yaml manifest.json
  raw/<backend>/<serve_config>/<workload>/{result.json, server.log, bench.log, command.json, status.json}
  candidates.csv measurements.csv normalized.csv failures.csv rankings.csv report.md
  plots/   # only if --plots
```

`run_id` comes from `experiment.run_id` if set, else `<experiment.name>-<UTC stamp>`. Re-running with the same `run_id` and `resume: true` continues where it stopped — see the smart-sweep script for how a single `run_id` is shared across separate per-backend `moe-bench run` invocations and then a final combined `normalize`/`rank`/`report` is run over the merged directory.

## Common commands

```bash
# Plan a sweep without executing
moe-bench run configs/smoke.yaml --dry-run

# Full sweep + post-processing
CUDA_VISIBLE_DEVICES=0,1,2,3 moe-bench run configs/qwen3_a3b.yaml --report

# Run only one backend in this invocation (used to serialize GPU use between backends)
moe-bench run configs/<cfg>.yaml --backends vllm --report

# Re-post-process an existing run dir (idempotent)
moe-bench normalize results/<run_id>
moe-bench rank      results/<run_id>
moe-bench report    results/<run_id> [--plots]

# Expand search_space grids to explicit serve_configs for inspection
moe-bench generate configs/<sweep>.yaml --out configs/<sweep>_expanded.yaml

# Build a follow-up validation config from top-K winners
moe-bench shortlist results/<run_id> --top-k 5 --repeats 5 --out configs/validation.yaml

# End-to-end driver that cleans GPUs between backends, runs each, then combines
./scripts/run_qwen3_a3b_grid.sh

# Backend envs
./scripts/create_backend_venvs.sh all     # or `vllm` / `sglang`
./scripts/check_backend_envs.sh
```

There is no test suite, lint config, or formatter wired up in this repo. Validate changes by running a dry-run plus a small real config (`configs/smoke.yaml`) end-to-end and inspecting `report.md`.

## Gotchas

- **vLLM and SGLang can't share GPUs at the same time.** `scripts/run_qwen3_a3b_grid.sh` runs `--backends vllm` and `--backends sglang` as separate invocations with `cleanup_gpu_processes` (a `pkill` + `nvidia-smi`-driven escalation to SIGKILL) between them. Don't try to run both backends concurrently from a single `moe-bench run`.

- **`objective.maximize: goodput_at_slo`** is a derived metric defined in `rank.py:compute_goodput_at_slo`: it's `median_output_tok_s_per_gpu` when both `p99_ttft_ms` and `p99_tpot_ms` are within `objective.constraints`, else `0`. Unlike the plain throughput objective, this keeps SLO-violators in `rankings.csv` (with goodput=0) instead of pre-filtering them — useful when you want to see "fast but tail-broken" candidates rather than have them disappear. The default config (`configs/qwen3_a3b_4gpu_grid.yaml`) uses this objective.
- **YAML key conversion is mechanical.** Anything in the backend sub-dict that isn't `extra_args`/`args`/`raw_args`/`server_args` or `_`-prefixed gets converted to a CLI flag by `_` → `-`. Bools flip to `--no-foo` when false. Unknown CLI flags belong in `extra_args:` (a list).
- **SGLang result parsing is a special case.** SGLang's bench client emits JSONL; `SglangBackend.parse_result_path` writes `result.json` from the last JSONL line. If you add a new backend with a different output format, you'll likely need to override `parse_result_path` similarly.
- **`workloads.request_rate` defaults to `"inf"` (string).** `to_float` in `normalize.py` handles `"inf"`/`"infinity"`. Keep the string form when round-tripping YAML.
- **Constraint metrics in `objective.constraints` are looked up as either `<metric>` or `median_<metric>`** (see `rank.passes_constraints`). Setting a constraint to `null` disables it.
