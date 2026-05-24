# Session context — moe-bench HTML report + parallelism sweep

Summary of the work done in this session so a future session can pick up cleanly.
Pairs with `CLAUDE.md` (architecture) and the configs in `configs/`.

> **Update (since this file was written):**
> - The parallelism config was renamed from `configs/qwen3_a3b_4gpu_parallelism.yaml`
>   to `configs/moe_parallelism_4gpu.yaml` (model-agnostic; defaults still target
>   Qwen3-30B-A3B). All path references below have been updated in place.
> - The driver script was renamed from `scripts/run_qwen3_a3b_parallelism.sh` to
>   `scripts/run_parallelism_sweep.sh` and parameterized via env vars
>   (`MODEL`, `RUN_ID`, `SERVED_MODEL_NAME`, `DTYPE`) with the qwen3-a3b values
>   as defaults.
> - `moe-bench run` now accepts `--model`, `--run-id`, `--served-model-name`,
>   `--dtype` flags that override the corresponding `experiment.*` /
>   `backends.vllm.served_model_name` fields. `apply_cli_overrides` in
>   `moe_bench/config.py` is the pure function behind it; first tests in the
>   repo at `tests/test_config_overrides.py`.
> - The parallelism-sweep root cause was rediagnosed: the first vLLM candidate's
>   failure at GMU=0.90 was caused by NCCL P2P buffers reducing free memory on
>   cuda:3 below the per-rank check threshold, NOT by a pre-existing orphan
>   process. The orphan write-up below (and the "Working memory: GPU / process
>   state" section) is stale on that point — see `issues/prd.md` for the
>   corrected analysis.

## Goal of the project

`moe-bench` compares **vLLM vs SGLang** on a model (currently Qwen3-30B-A3B on
4× A100-PCIE-40GB) by sweeping serve-parameter candidates against a workload
grid and ranking them. The platform is also an **investigation tool**: the
HTML report should let a user explore tradeoffs and decide which backend +
config to deploy.

## Architecture decisions made this session

- **Single-file interactive HTML report** (`moe_bench/html_report.py`),
  generated alongside `report.md`. Plotly JS inlined once; measurement JSON
  embedded for the constraint-slider leaderboard (vanilla JS does the filter).
  Section order:
  1. Header
  2. **Scientific summary** divider — `decision_summary`, `operating_curve`,
     `latency_vs_promptlen` (TTFT + TPOT side-by-side), `decision_map`,
     `tuning_sensitivity`
  3. **Drill into candidates** divider — `global_winner`,
     `robustness_leaderboard`, `constraint_leaderboard`, `pareto`,
     `parameter_sweep` (parcoords), `per_workload_drilldown`, `candidate_cards`
- **`goodput_at_slo` objective** (`rank.py:compute_goodput_at_slo`): throughput
  conditional on meeting all `objective.constraints`; SLO violators get
  goodput=0 instead of being pre-filtered out, so they remain visible in
  reports. Used as default in the grid / quick / parallelism configs.
- **Global-winner ranking** = geomean of `relative_to_best` across workloads,
  tiebreak by median rank, then wins. *Known caveat:* `_geomean` filters zeros,
  which means a candidate with `relative_to_best=0` on most workloads (due to
  SLO violation under `goodput_at_slo`) has those workloads dropped from the
  geomean rather than dragging it down. Discussed but not fixed; options are
  (a) floor zeros at ~0.05, (b) use worst-case as primary, (c) median rank as
  primary. Worth fixing as `report.robustness.zero_floor: 0.05`.
- **Two-tier env model** unchanged: platform env (aiq venv with plotly added)
  runs `moe-bench`; backend envs at `/mnt/projects/AI/josh/vllm/.venv/` and
  `/mnt/projects/AI/josh/sglang/python/.venv/` run the engines themselves.

## Files added / modified

| Path | Change |
|---|---|
| `moe_bench/html_report.py` | **new** — full interactive HTML report |
| `moe_bench/rank.py` | added `compute_goodput_at_slo`, `median_goodput_at_slo` column |
| `moe_bench/report.py` | wired `generate_html_report` into `generate_report` |
| `moe_bench/cli.py` | added `--no-html` flag (HTML on by default with `--report`) |
| `pyproject.toml` | added `plotly>=5.18` dep |
| `CLAUDE.md` | documented HTML report + `goodput_at_slo` |
| `configs/qwen3_a3b_4gpu_grid.yaml` | **new** — 24 candidates × 24 workloads × 1 = 576 cells |
| `configs/qwen3_a3b_4gpu_quick.yaml` | **new** — 24 candidates × 6 workloads = 144 cells |
| `configs/moe_parallelism_4gpu.yaml` | **new** — 11 candidates focused on TP/DP/EP axis × 6 workloads |
| `scripts/run_qwen3_a3b_grid.sh` | renamed from `run_qwen3_a3b_smart_sweep.sh` |
| `scripts/run_qwen3_a3b_quick.sh` | **new** |
| `scripts/run_parallelism_sweep.sh` | **new** |
| (removed) `configs/qwen3_a3b_4gpu_smart_sweep.yaml` | deleted |
| (removed) `configs/qwen3_a3b_4gpu_param_grid.yaml` | deleted |
| `results/stack-smoke-2026-05-22T114430Z/report.html` | sample of the interactive HTML report (4 candidates × 1 workload, opt-125m smoke) |

## Plot tuning lessons learned

- **Pareto legend bug**: a single trace with mixed marker symbols
  (circle + star) displays only one symbol in the legend. Fix: split into two
  traces per backend (non-frontier circles + frontier stars-on-line) so the
  legend marker matches what's on the chart. Linked by `legendgroup=backend`.
- **Plotly annotation autorange bug**: `ax`/`ay` arrow-tail offsets default to
  pixel units *only when* `axref`/`ayref` are explicit. Without them, on a log
  axis Plotly interprets `ax=30` as "30 units in log-data space" — which made
  the Pareto x-axis blow up to 10^35. Fix: always set
  `axref="pixel", ayref="pixel"` on annotations. Also defensively compute
  explicit log-space `xaxis.range` and `yaxis.range` from the data so the
  chart can't blow up regardless of Plotly behavior. See `_figure_pareto` and
  `_figure_operating_curve`.

## Sweep runs and what we learned

### Smoke test (passed cleanly)
- `configs/stack_smoke.yaml` on `facebook/opt-125m`, GPU 0 only.
- 4 candidates × 1 workload, finished in minutes.
- Used to validate the HTML report end-to-end.

### Grid sweep (killed after 25h, 169/576 cells)
- `configs/qwen3_a3b_4gpu_grid.yaml`, all 4 GPUs.
- Ran ~25h, completed 169 cells (~29%) with **0 failures** before being killed.
- Pace turned out **~6.5 cells/h not 12** because `conc=1 × output=1024` cells
  take 40+ minutes each. Original 30h projection was 2-3× too optimistic.
- Data exists at `results/qwen3-a3b-4gpu-grid/` if you want to inspect the
  6 fully-completed TP=4 vLLM candidates. Partial.

### Quick sweep (cancelled mid-launch)
- `configs/qwen3_a3b_4gpu_quick.yaml` — dropped `conc=1`, dropped long-context,
  `num_prompts=32`. Designed for ~2h.
- Cancelled to pivot to the parallelism-focused config.

### Parallelism sweep (FAILED — see fix-up section below)
- `configs/moe_parallelism_4gpu.yaml` — 11 candidates focused on
  isolating the TP/DP/EP axis (one balanced scheduler point per layout).
- **All 11 candidates hit `server_ready_timeout`**. Root causes:
  1. **First vLLM candidate** died in seconds:
     `Free memory on device cuda:3 (34.35/39.49 GiB) on startup is less than
     desired GPU memory utilization (0.9, 35.54 GiB)`. ~5 GB residual on GPU 3
     from a prior orphan `VLLM::EngineCore` left over after the quick sweep
     cancellation; with `gpu_memory_utilization=0.9` there was no margin.
  2. **Pure-DP candidates fundamentally don't fit**: Qwen3-30B-A3B is **57 GB**
     in bfloat16 (per the load log: "Checkpoint size: 56.87 GiB"). A100-PCIE
     has 40 GB. `TP=1 DP=4` replicates the full model on each GPU = won't fit.
  3. **Cascade**: every failed candidate leaked workers; the next candidate
     saw dirty GPUs and failed memory-balance checks (`The memory capacity is
     unbalanced. Some GPUs may be occupied by other processes`). SGLang then
     died with `Rank 0 scheduler died during initialization (exit code: -3)`.

## Fix-up plan for the parallelism sweep (TO DO when next picking this up)

1. **Sudo-kill any leftover GPU processes** before relaunch. As of session end,
   GPU 3 had 23 GB used by orphan PID 2755203 (`VLLM::EngineCore`, root-owned).
   There was also an unrelated `vllm serve Qwen3-1.7B` on port 8000 (PID 2753718)
   — check whether that's intentional before killing it. Verify with:
   ```bash
   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   sudo -n kill -9 <orphan_pid>
   ```
2. **Edit `configs/moe_parallelism_4gpu.yaml`**:
   - Remove `vllm_tp1dp4` and `vllm_tp1dp4_ep` candidates (model won't fit)
   - `gpu_memory_utilization: 0.90 → 0.85` for both vLLM and SGLang
   - `server_ready_timeout_sec: 1200 → 1800`
3. **Resulting sweep**: 9 candidates × 6 workloads = **54 cells**, target ~1h.
4. **Optionally harden `scripts/run_parallelism_sweep.sh`** to verify
   `nvidia-smi` shows 0 MB on all GPUs before launching each backend phase,
   and to retry cleanup with `sudo -n` if needed.

## What the candidate list should look like after the fix

**vLLM (4 candidates)** — `(TP, DP)` ∈ {(4,1), (2,2)} × EP {off, on}:
- `vllm_tp4dp1`, `vllm_tp4dp1_ep`
- `vllm_tp2dp2`, `vllm_tp2dp2_ep`

**SGLang (5 candidates)** — TP=4 always (it can't do pure-DP across GPUs the
way vLLM can; its `dp` and `ep` are sub-divisions within TP):
- `sglang_tp4` — baseline
- `sglang_tp4_ep` — adds expert parallel
- `sglang_tp4_dpattn` — DP-attention without EP
- `sglang_tp4_dpattn_ep` — DP-attention + EP (SGLang's MoE flagship)
- `sglang_tp4_dpattn2_ep` — `dp=2` mixed DP-attention + EP

## Pending design items (discussed but not implemented)

- **Configurable `report:` block** in YAML — `sections`, `slo`,
  `leaderboard_columns`, `workload_groups`, `top_k_candidate_cards`,
  `primary_metric`. Default-ordered section list, user overrides by omission.
- **New sections** not yet built: `failure_summary` (early-warn card),
  `workload_group_winners` (per-group hero card), `slo_compliance` (stacked
  bar per workload).
- **`top_k_candidate_cards`** cap on the candidate-cards section (currently
  emits all candidates; with 24 candidates it's a long scroll).
- **Robustness zero-floor fix** described above.

## Working memory: GPU / process state at end of session

- 4× A100-PCIE-40GB on host `172.16.102.153`
- GPU 0–2: clean; GPU 3 had **23 GB** used by orphan `VLLM::EngineCore` PID
  2755203 (root-owned, needs `sudo -n kill -9`)
- Unrelated `vllm serve Qwen/Qwen3-1.7B` on port 8000 (PID 2753718) was also
  running — confirm with the user before touching it
- Platform venv: `/mnt/projects/AI/josh/aiq/.venv` (has plotly 6.5.2)
- Backend venvs: `/mnt/projects/AI/josh/vllm/.venv/`,
  `/mnt/projects/AI/josh/sglang/python/.venv/`
- Last sweep log: `results/logs/qwen3_a3b_parallelism_20260523T163718Z.log`
- Last (broken) report: `results/qwen3-a3b-4gpu-parallelism/report.html`
- Sample working report: `results/stack-smoke-2026-05-22T114430Z/report.html`
