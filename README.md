# moe-bench

`moe-bench` is a small benchmark platform for finding good LLM/MoE inference serving configs across vLLM and SGLang.

It runs configurable optimization sweeps:

```text
backend server-parameter candidates × workload constraints × repeats
```

`serve_configs` / generated `search_space` candidates are the degrees of freedom. Workload fields like input tokens, output tokens, and concurrency are the constraints. The platform produces raw logs/results, `candidates.csv`, `measurements.csv`, objective rankings, a Markdown summary (`report.md`), and a self-contained interactive HTML report (`report.html`) for exploring tradeoffs in a browser. PNG plots are available as opt-in via `--plots`.

## Install

```bash
cd moe-bench
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Or use the existing project env:

```bash
source /mnt/projects/AI/josh/aiq/.venv/bin/activate
cd /mnt/projects/AI/josh/moe-bench
pip install -e .
```

## Backend environments

The platform is separate from the serving engines. Build backend-specific envs from prebuilt wheels with:

```bash
./scripts/create_backend_venvs.sh all
./scripts/check_backend_envs.sh
```

This creates:

```text
.backends/vllm/bin/python
.backends/sglang/bin/python
```

using `pip install --only-binary=:all: vllm sglang` style installs to avoid local compilation. See [BACKENDS.md](BACKENDS.md) for version pins, custom indexes, editable source builds, and using existing repo venvs.

### Docker

A `Dockerfile` is included that bundles the platform plus pre-built vLLM and SGLang backend venvs:

```bash
docker build -t moe-bench .
docker run --rm -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/moe-bench/results \
  moe-bench
# inside the container:
moe-bench run configs/moe_parallelism_4gpu.yaml --report
```

See [BACKENDS.md](BACKENDS.md) for build-arg overrides (`VLLM_PACKAGE`, `SGLANG_PACKAGE`, `PYTHON_VERSION`).

## Config style

Backend serve settings are written as structured YAML for both vLLM and SGLang. Keys are converted to CLI flags by replacing `_` with `-`:

```yaml
vllm:
  tensor_parallel_size: 4
  enable_expert_parallel: true

sglang:
  tp: 4
  context_length: 8192
  enable_dp_attention: true
```

This becomes flags like `--tensor-parallel-size 4`, `--enable-expert-parallel`, `--context-length 8192`, etc. For rare unsupported/new flags, use `extra_args`:

```yaml
sglang:
  tp: 4
  extra_args: [--some-new-flag, value]
```

## Search-space optimization

You can either list explicit candidates in `serve_configs`, or generate candidates from a grid:

```yaml
search_space:
  vllm:
    defaults:
      tensor_parallel_size: 1
      data_parallel_size: 1
    grid:
      max_num_seqs: [8, 16]
      max_num_batched_tokens: [512, 1024]
  sglang:
    defaults:
      tp: 1
      context_length: 1024
    grid:
      max_running_requests: [8, 16]
      chunked_prefill_size: [512, 1024]
```

Expand for inspection:

```bash
moe-bench generate configs/search.yaml --out configs/search_expanded.yaml
```

## Commands

Dry-run a sweep plan:

```bash
moe-bench run configs/smoke.yaml --dry-run
```

Run a sweep:

```bash
moe-bench run configs/qwen3_a3b.yaml --report
```

`--report` triggers `normalize → rank → report` after the run and emits both `report.md` and `report.html`. The HTML report is on by default; disable it with `--no-html`. PNG plots are off by default; enable them with `--plots`.

Override config fields from the CLI without editing YAML (useful for reusing one config across models):

```bash
moe-bench run configs/moe_parallelism_4gpu.yaml \
  --model Qwen/Qwen3-30B-A3B \
  --run-id my-run \
  --served-model-name moe-bench-my-run \
  --dtype bfloat16 \
  --report
```

Post-process an existing run (idempotent — safe to re-run):

```bash
moe-bench normalize results/<run_id>
moe-bench rank      results/<run_id>
moe-bench report    results/<run_id>           # writes report.md + report.html
moe-bench report    results/<run_id> --plots   # also emit PNG plots
```

Driver scripts that handle GPU cleanup between backends (vLLM and SGLang can't share GPUs simultaneously) and run a full sweep end-to-end:

```bash
./scripts/run_qwen3_a3b_grid.sh        # full grid sweep
./scripts/run_qwen3_a3b_quick.sh       # smaller quick sweep
./scripts/run_parallelism_sweep.sh     # parallelism-focused sweep (TP/DP/EP)
```

The parallelism driver is parameterized via env vars: `MODEL`, `RUN_ID`, `SERVED_MODEL_NAME`, `DTYPE`, `CONFIG`, `CUDA_VISIBLE_DEVICES`.

### Interactive HTML report

`report.html` is a single self-contained file (Plotly JS inlined, measurement JSON embedded) — open it directly in a browser, no server required. Sections:

- **Scientific summary** — decision summary, operating curve, latency vs prompt length, decision map, tuning sensitivity
- **Drill into candidates** — global winner, robustness leaderboard, constraint-slider leaderboard (filter live on `p99_ttft_ms` / `p99_tpot_ms`), Pareto front, parameter parcoords, per-workload drilldown, per-candidate cards

### Optional PNG plots

Pass `--plots` to additionally emit decision-oriented PNGs under `results/<run_id>/plots/`: top-k by workload, winning throughput, Pareto, throughput/latency heatmaps, backend win counts, coverage/failure heatmap. Control legend density without hiding candidate details:

```yaml
report:
  legend_params: [max_num_seqs, max_running_requests, max_num_batched_tokens]
```

The plots use only those parameters in labels; `candidates.csv` carries the full curated tuning parameters for deep dives.

Create a validation config from the top ranked server-parameter candidates:

```bash
moe-bench shortlist results/<run_id> --top-k 5 --repeats 5 --out configs/validation.yaml
```

Import current historical vLLM/SGLang sweep results:

```bash
moe-bench import-legacy \
  --vllm ../vllm/benchmarks/moe_sweep/results \
  --sglang ../sglang/benchmark/moe_sweep/results \
  --out results/legacy-qwen3-a3b \
  --gpus 4
```

## Result layout

```text
results/<run_id>/
  config.yaml
  manifest.json
  raw/<backend>/<serve_config>/<workload>/
    result.json
    server.log
    bench.log
    command.json
    status.json
  candidates.csv       # one row per candidate; curated top vLLM/SGLang tuning params, default-filled, no JSON blob
  measurements.csv     # one row per candidate × workload measurement, including server_state_sha
  normalized.csv       # compatibility alias for measurements.csv
  failures.csv
  rankings.csv
  report.md            # includes a Server parameter configs section
  report.html          # interactive single-file report (default; --no-html to skip)
  plots/               # only if --plots
```

## Objective

Default ranking is simple and auditable:

1. treat each `candidate_id` as one server-parameter vector
2. keep only valid measurements
3. apply latency constraints from `objective.constraints` (e.g. `p99_ttft_ms`, `p99_tpot_ms`)
4. rank by the configured objective descending per workload constraint

Two objectives are supported via `objective.maximize`:

- `output_tok_s_per_gpu` — raw throughput per GPU; SLO violators are filtered out before ranking.
- `goodput_at_slo` (default in the 4-GPU configs) — `median_output_tok_s_per_gpu` when all `objective.constraints` are met, else `0`. Keeps SLO violators visible in `rankings.csv` (with goodput=0) rather than dropping them, so "fast but tail-broken" candidates still surface in the report.

Global-winner ranking across workloads is the geomean of `relative_to_best` per workload, with median rank as tiebreaker.

Stage/exploration/validation are not hardcoded concepts. They are just different YAML configs with different grids and repeat counts.
