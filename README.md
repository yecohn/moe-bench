# moe-bench

`moe-bench` is a small benchmark platform for finding good LLM/MoE inference serving configs across vLLM and SGLang.

It runs configurable optimization sweeps:

```text
backend server-parameter candidates × workload constraints × repeats
```

`serve_configs` / generated `search_space` candidates are the degrees of freedom. Workload fields like input tokens, output tokens, and concurrency are the constraints. The platform produces raw logs/results, `candidates.csv`, `measurements.csv`, objective rankings, Markdown reports, and plots.

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

Post-process an existing run:

```bash
moe-bench normalize results/<run_id>
moe-bench rank results/<run_id>
moe-bench report results/<run_id>
```

Control plot legend detail without hiding candidate details:

```yaml
report:
  legend_params: [max_num_seqs, max_running_requests, max_num_batched_tokens]
```

The plots use only those parameters in labels; `candidates.csv` contains the curated top tuning parameters for deep dives without dumping every backend default.

Current plots are decision-oriented:

- `topk_candidates_by_workload.png` — top server candidates per workload constraint
- `winning_throughput_by_workload.png` — winning throughput per workload
- `winner_latency_by_workload.png` — TTFT/TPOT latency of the selected winner
- `pareto_throughput_vs_ttft.png` and `pareto_throughput_vs_tpot.png` — throughput/latency tradeoff
- `best_throughput_heatmap_out*.png` plus winner latency heatmaps — workload grid view
- `backend_win_counts.png` and `backend_best_throughput_ratio.png` — backend-level comparison
- `coverage_failure_heatmap.png` — failures/coverage by candidate and workload

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
  plots/
```

## Objective

Default ranking is simple and auditable:

1. treat each `candidate_id` as one server-parameter vector
2. keep only valid measurements
3. apply latency constraints if configured
4. rank by `output_tok_s_per_gpu` descending per workload constraint

Stage/exploration/validation are not hardcoded concepts. They are just different YAML configs with different grids and repeat counts.
