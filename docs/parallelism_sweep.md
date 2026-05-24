# Parallelism sweep

A turnkey sweep that compares vLLM vs SGLang across the **TP / DP / EP** axis on 4 GPUs and produces an interactive HTML report ranking the candidates.

The defaults target Qwen3-30B-A3B on A100-PCIE-40GB, but the same config runs against any MoE model that fits the per-GPU memory budget under TP=4 or TP=2 DP=2.

## Run it

Defaults (Qwen3-30B-A3B, 4 GPUs, bfloat16, ~1h wall-clock):

```bash
scripts/run_parallelism_sweep.sh
```

A different model:

```bash
MODEL="meta-llama/Llama-3-8B" \
RUN_ID="llama-3-8b-parallelism" \
DTYPE="float16" \
  scripts/run_parallelism_sweep.sh
```

The script serializes the two backends (vLLM and SGLang can't share GPUs at the same time), cleans up between phases, and calls `moe-bench normalize / rank / report` at the end. Output lands in `results/<RUN_ID>/`.

If you'd rather drive it yourself:

```bash
moe-bench run configs/moe_parallelism_4gpu.yaml \
  --model meta-llama/Llama-3-8B \
  --run-id llama-3-8b-parallelism \
  --served-model-name moe-bench-llama-3-8b \
  --dtype float16 \
  --report
```

`--dry-run` is supported and prints the resolved model, run id, and per-candidate workload count without launching anything.

### Run via Docker

A `Dockerfile` at the repo root packages moe-bench plus pre-built vLLM and SGLang backend venvs into a single image. After `docker build -t moe-bench .`:

```bash
docker run --rm -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/moe-bench/results \
  moe-bench

# inside the container:
moe-bench run configs/moe_parallelism_4gpu.yaml --report
```

See [`../BACKENDS.md`](../BACKENDS.md#10-docker-image) (section 10) for prerequisites, version-pin overrides, and the full reference.

## Configure it

Most of the sweep is fixed in `configs/moe_parallelism_4gpu.yaml`. Identity fields are intended to be set from the CLI; everything else lives in the YAML.

To author a brand-new sweep config from scratch, see [writing_a_sweep_config.md](writing_a_sweep_config.md).

### Identity — CLI overrides only

| Flag | Replaces | Default in script |
|---|---|---|
| `--model` | `experiment.model` | `Qwen/Qwen3-30B-A3B` |
| `--run-id` | `experiment.run_id` and `experiment.name` | `qwen3-a3b-4gpu-parallelism` |
| `--served-model-name` | `backends.vllm.served_model_name` | `moe-bench-${RUN_ID}` |
| `--dtype` | `experiment.dtype` plus every per-candidate `dtype` field | `bfloat16` |

The same flags are read from `MODEL` / `RUN_ID` / `SERVED_MODEL_NAME` / `DTYPE` env vars by the wrapper script.

### What's in the YAML

- **`serve_configs`** — the parallelism candidates. 4 vLLM (TP=4 DP=1 ± EP, TP=2 DP=2 ± EP) and 5 SGLang (TP=4 baseline, +EP, +DP-attention, +DP-attention+EP, +mixed). Add or drop entries here to change the candidate list. Each candidate's name is the join key in the report.
- **`workloads`** — `input_lens × output_lens × concurrencies` define the grid. `num_prompts` is the per-cell request count; `num_warmups` is the warmup count.
- **`execution`** — `repeats`, `resume`, `fail_fast`, `server_ready_timeout_sec`, `bench_timeout_sec`, `cool_down_sec`. `resume: true` is the default and is how mid-sweep restarts work: cells with both `status.json` and `result.json` are skipped.
- **`objective`** — what to maximize and which constraints define "valid". Default is `goodput_at_slo` (throughput conditional on `p99_ttft_ms` and `p99_tpot_ms` constraints; SLO violators get goodput=0 and remain visible). See [optimization.md](optimization.md) for what these SLOs mean and how ranking works.
- **`gpu_memory_utilization` (vLLM) / `mem_fraction_static` (SGLang)** — 0.85 by default. Lower if your model is bigger or your GPUs run hot at startup; vLLM's per-rank free-memory check at startup includes NCCL P2P buffers, so 0.90 is risky on 40 GiB cards.

The candidates assume the model fits under TP=4 or TP=2 DP=2. For a small model that fits at TP=1, you can add pure-DP layouts to `serve_configs`. For a model that doesn't fit at TP=2 DP=2, drop those candidates and add higher-TP variants if your hardware supports them.

## The report

After the run completes, open `results/<RUN_ID>/report.html` in a browser. The HTML is self-contained — Plotly is inlined and the measurement JSON is embedded for the interactive constraint slider.

The report has two main sections.

### Scientific summary — start here

- **Decision summary** — the recommended backend + candidate for the current SLO settings, with a plain-English justification.
- **Operating curve** — throughput vs concurrency per backend, with the Pareto-best candidate per concurrency annotated.
- **Latency vs prompt length** — TTFT and TPOT side-by-side, so you can see which candidate handles short vs long prompts well.
- **Decision map** — a heatmap of "which candidate wins" across the workload grid. Reveals workload regimes where the global winner doesn't apply.
- **Tuning sensitivity** — how much performance varies across the candidate space. Tells you whether tuning matters for this model on this hardware.

### Drill into candidates

- **Global winner** — single best candidate by geomean of `relative_to_best` across workloads, with median-rank and win-count tiebreakers.
- **Robustness leaderboard** — candidates ranked by worst-case performance across workloads.
- **Constraint leaderboard** — interactive: drag the TTFT and TPOT sliders to see how the ranking changes under different SLOs.
- **Pareto** — throughput vs latency scatter with the frontier highlighted per backend.
- **Parameter sweep parcoords** — parallel-coordinates plot of every tuning param against the objective.
- **Per-workload drilldown** — same plot family but filtered to a single workload.
- **Candidate cards** — full parameter dump + measurement table per candidate.

The raw data behind everything is in `results/<RUN_ID>/{measurements,rankings,candidates}.csv` if you want to do your own analysis.

## When something fails

Per-cell failures are recorded in `results/<RUN_ID>/failures.csv` with one of these statuses: `server_ready_timeout`, `bench_timeout`, `oom`, `bench_failed`, `missing_result`.

For a candidate that never came up, check `results/<RUN_ID>/raw/<backend>/<candidate>/server.log` — the most common failure on memory-constrained GPUs is vLLM's startup check `Free memory on device cuda:N (X GiB) … less than desired (...)`. Lower `gpu_memory_utilization` by 0.05 and retry.

With `execution.resume: true` and the same `--run-id`, re-running the script picks up where it stopped without redoing successful cells.
