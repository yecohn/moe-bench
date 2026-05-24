# Writing a sweep config

A walkthrough for authoring your own `config.yaml` from scratch. The platform runs **backend × serve_config × workload** for every cell, ranks the cells by your objective, and writes a report. The config is where you decide what each of those dimensions contains.

If you just want to run the existing parallelism sweep against a different model, see [parallelism_sweep.md](parallelism_sweep.md) — you don't need to author anything.

## Required keys

A valid config has these top-level keys:

- `experiment` — identity and hardware
- `execution` — control-loop knobs (timeouts, repeats, resume)
- `backends` — which engines to launch and where their Python interpreters live
- `serve_configs` **or** `search_space` — the candidates you want to compare
- `workloads` — the request-pattern grid
- `objective` — what to maximize and which SLOs to enforce

Validation happens on load; missing required keys produce a clear error before anything launches.

## Step 1 — Pick a starting point

Don't write from scratch. Copy the closest existing config and edit it:

```bash
cp configs/smoke.yaml configs/my_sweep.yaml
```

`smoke.yaml` is the smallest complete example (~60 lines, one candidate, one workload). Good base for any new sweep. `moe_parallelism_4gpu.yaml` is a fuller example with 9 candidates and 6 workloads.

## Step 2 — `experiment`: identity + hardware

```yaml
experiment:
  name: my-sweep              # human label
  run_id: my-sweep            # results land in results/<run_id>/
  model: Qwen/Qwen3-30B-A3B   # HuggingFace ID or local path
  gpus: 4                     # how many GPUs you'll actually use
  dtype: bfloat16
  host: 127.0.0.1
```

`run_id` is the join key across runs — re-running with the same id resumes where you stopped (see Step 3). `name`, `run_id`, `model`, and `dtype` can be overridden per-invocation with `--model / --run-id / --dtype` so the same YAML serves multiple models.

## Step 3 — `execution`: timeouts and resume

```yaml
execution:
  repeats: 1                    # repeat each cell N times for variance estimates
  resume: true                  # skip cells with status.json + result.json
  fail_fast: false              # stop on first failure if true; else collect failures
  server_ready_timeout_sec: 1200
  bench_timeout_sec: 600
  cool_down_sec: 10             # gap between candidates
```

The defaults work for most cases. Bump `bench_timeout_sec` when you use long outputs or low concurrency (a single conc=1 / output_lens=1024 cell can run 40+ minutes). Leave `resume: true` so a killed sweep can be picked up.

## Step 4 — `backends`: where the engines live

```yaml
backends:
  vllm:
    enabled: true
    python: /mnt/projects/AI/josh/vllm/.venv/bin/python
    port: 19100
    served_model_name: moe-bench-my-sweep
  sglang:
    enabled: true
    python: /mnt/projects/AI/josh/sglang/python/.venv/bin/python
    port: 19101
```

The `python:` path is which interpreter the runner shells out to. The platform env never imports vLLM or SGLang itself — only the backend env does. Set `enabled: false` to skip a backend without removing it.

Ports just need to be free; the runner doesn't care what numbers you pick.

## Step 5 — `serve_configs`: define the candidates

This is the heart of the sweep. Each entry under `serve_configs` is **one server configuration** to compare. Two ways to author it.

### Pattern A — explicit list (hand-picked candidates)

Use when you have a specific set of configurations in mind, e.g. parallelism layouts:

```yaml
serve_configs:
  - name: vllm_tp4dp1
    vllm: &vllm_common                # YAML anchor for shared params
      dtype: bfloat16
      gpu_memory_utilization: 0.85
      max_num_seqs: 128
      max_num_batched_tokens: 8192
      tensor_parallel_size: 4
      data_parallel_size: 1
      enable_expert_parallel: false

  - name: vllm_tp4dp1_ep
    vllm:
      <<: *vllm_common                # inherit; override only what differs
      enable_expert_parallel: true

  - name: sglang_tp4
    sglang:                           # SGLang-only candidate (no vllm: key)
      tp: 4
      mem_fraction_static: 0.85
      max_running_requests: 128
      chunked_prefill_size: 8192
```

**How keys become CLI flags.** Any key under `vllm:` or `sglang:` is converted automatically:

- `tensor_parallel_size: 4` → `--tensor-parallel-size 4`
- `enable_prefix_caching: true` → `--enable-prefix-caching`
- `enable_prefix_caching: false` → `--no-enable-prefix-caching`

For flags the runner can't derive this way, use `extra_args: [...]`:

```yaml
vllm:
  tensor_parallel_size: 4
  extra_args: ["--swap-space", "16"]
```

**Mixed-backend candidates.** Put both `vllm:` and `sglang:` sub-dicts in one entry to compare the same params across engines:

```yaml
- name: tp4_baseline
  vllm: { tensor_parallel_size: 4, ... }
  sglang: { tp: 4, ... }
```

The candidate runs once per backend (so this expands into 2 candidates internally).

### Pattern B — search_space (Cartesian grid)

Use when you want every combination of a few parameter values, e.g. sweeping scheduler capacity:

```yaml
search_space:
  vllm:
    defaults:
      dtype: bfloat16
      gpu_memory_utilization: 0.85
      tensor_parallel_size: 4
    grid:
      max_num_seqs:           [64, 128, 256]
      max_num_batched_tokens: [4096, 8192]
      enable_prefix_caching:  [true, false]
```

That expands to 3 × 2 × 2 = 12 vLLM candidates with auto-generated names. `defaults` are applied to every generated candidate; `grid` is the Cartesian product axes.

Mix `serve_configs` and `search_space` freely — the runner concatenates them.

To preview what a `search_space` expands to without running:

```bash
moe-bench generate configs/my_sweep.yaml --out configs/my_sweep_expanded.yaml
```

## Step 6 — `workloads`: the request grid

```yaml
workloads:
  input_lens:    [256, 4096]      # prompt token counts
  output_lens:   [128]            # decode token counts
  concurrencies: [4, 16, 64]      # simultaneous requests per cell
  num_prompts:   32               # total requests per cell
  num_warmups:   4                # warmup requests excluded from metrics
  seeds:         [0]
  random_range_ratio: "0.0"       # prompt-length jitter; "0.0" = exact
```

Cell count per candidate = `len(input_lens) × len(output_lens) × len(concurrencies) × len(seeds) × execution.repeats`. The example above is 2 × 1 × 3 × 1 × 1 = **6 workloads** per candidate.

**How to choose values:**

- **`input_lens` / `output_lens`** — bracket your real traffic. Short prompt + long output is decode-heavy; long prompt + short output is prefill-heavy. Pick at least one of each if you care about both regimes.
- **`concurrencies`** — sweep from "single user" (1 or 4) to "loaded" (64 or 128) to find the throughput-vs-latency knee.
- **`num_prompts`** — larger means more stable percentile metrics but longer cell time. 32 is fine for exploration; 256+ for validation runs.

## Step 7 — `objective`: how to rank

```yaml
objective:
  maximize: goodput_at_slo
  constraints:
    p99_ttft_ms: 2000          # SLO: 2s p99 first-token latency
    p99_tpot_ms: 100           # SLO: 100ms p99 per-output-token
```

Pick `maximize` based on what question you're asking:

- **`goodput_at_slo`** (recommended) — throughput conditional on both SLOs being met. Cells that violate an SLO get goodput=0 and remain visible in the report (rather than being filtered out).
- **`median_output_tok_s_per_gpu`** — raw decode throughput per GPU, SLO-blind.
- Any column in `measurements.csv` (or its `median_` variant) is a valid objective.

Constraints are looked up as either `<metric>` or `median_<metric>`. Set any constraint to `null` to disable it. With no constraints, only the `maximize` metric matters for ranking.

For a deeper explanation of SLOs, the objective semantics, and how the per-workload + global ranking is computed, see [`optimization.md`](optimization.md).

## Step 8 — Dry-run to validate

```bash
moe-bench run configs/my_sweep.yaml --dry-run
```

Prints the resolved model, run id, dtype, total cell count, and per-candidate workload count — without launching any servers. Catches YAML mistakes (missing keys, wrong types) and confirms you got the cell count you expected.

If cell count is much higher than you intended, you have an unintended Cartesian explosion in `search_space` or `workloads`.

## Step 9 — Run

```bash
moe-bench run configs/my_sweep.yaml --report
```

`--report` runs `normalize` + `rank` + `report` automatically after the sweep finishes. Output lands in `results/<run_id>/report.html` (plus `report.md` and the CSVs).

If you want the safety net of forced cleanup between vLLM and SGLang phases (recommended on shared GPUs), use the wrapper script:

```bash
CONFIG=configs/my_sweep.yaml RUN_ID=my-sweep scripts/run_parallelism_sweep.sh
```

The script forces a `pkill` + `nvidia-smi`-driven cleanup before each backend so leaked workers from a prior crash can't poison the next phase.

## A complete minimal example

This is a working end-to-end config in ~60 lines, equivalent to `configs/smoke.yaml`:

```yaml
experiment:
  name: smoke
  model: Qwen/Qwen3-30B-A3B
  gpus: 4
  dtype: bfloat16
  host: 127.0.0.1

execution:
  repeats: 1
  resume: true
  fail_fast: false
  server_ready_timeout_sec: 900
  bench_timeout_sec: 1800
  cool_down_sec: 10

backends:
  vllm:
    enabled: true
    python: .backends/vllm/bin/python
    port: 8000
    served_model_name: moe-bench
  sglang:
    enabled: true
    python: .backends/sglang/bin/python
    port: 30000

serve_configs:
  - name: tp4_baseline
    vllm:
      tensor_parallel_size: 4
      max_num_seqs: 128
      max_num_batched_tokens: 8192
      gpu_memory_utilization: 0.85
      enable_prefix_caching: false
    sglang:
      tp: 4
      context_length: 8192
      mem_fraction_static: 0.85
      max_running_requests: 128
      chunked_prefill_size: 8192
      disable_radix_cache: true

workloads:
  input_lens: [256]
  output_lens: [128]
  concurrencies: [1]
  num_prompts: 16
  num_warmups: 2
  seeds: [0]

objective:
  maximize: output_tok_s_per_gpu
  constraints:
    p99_ttft_ms: null
    p99_tpot_ms: null
```

One candidate × two backends × one workload = 2 cells. Useful as a sanity check before launching a real sweep.

## Common patterns

- **Compare backends at fixed parallelism.** One candidate with both `vllm:` and `sglang:` sub-dicts. See `configs/smoke.yaml`.
- **Sweep the parallelism axis on one backend.** Explicit list of candidates varying TP / DP / EP. See `configs/moe_parallelism_4gpu.yaml`.
- **Sweep scheduler capacity.** `search_space` over `max_num_seqs` × `max_num_batched_tokens` × `enable_prefix_caching`. Fewer than ~30 cells / hour, so cap the grid.
- **Validate top-K winners from a prior sweep with more repeats.** Run `moe-bench shortlist results/<prior-run> --top-k 5 --repeats 5 --out configs/validation.yaml` to auto-generate a config with just the winners.

## Tips

- **Cell count drives wall-clock.** A cell is one full bench run. Typical 1–10 min each; with conc=1 + long output, 40+ min. Multiply by total cells before launching.
- **vLLM and SGLang can't share GPUs.** The runner serializes them automatically within a single `moe-bench run`. The wrapper script does so across invocations with cleanup in between.
- **`gpu_memory_utilization` / `mem_fraction_static`** of 0.85 is calibrated for 30B-class MoE on A100-40GB (NCCL P2P eats ~5 GiB during multi-rank init). Use 0.90 on bigger GPUs or smaller models; 0.80 if the first candidate fails its startup check.
- **The exact server command** the runner launched for each candidate is saved to `results/<run_id>/raw/<backend>/<candidate>/server_command.json`. Useful when a candidate misbehaves and you want to reproduce by hand.

## Next steps

- [optimization.md](optimization.md) — what SLOs mean and how the sweep ranks candidates
- [parallelism_sweep.md](parallelism_sweep.md) — how to run the bundled parallelism sweep and what to expect in the report
- [`../CLAUDE.md`](../CLAUDE.md) — pipeline internals (cli → runner → normalize → rank → report) for when you need to debug deeper
- [`../BACKENDS.md`](../BACKENDS.md) — building / pointing at backend Python envs
