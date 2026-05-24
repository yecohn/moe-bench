# SLOs and how the sweep ranks candidates

What "best" means in a moe-bench sweep is not a fixed property of the platform — it's set by your `objective:` block in the YAML. This doc explains the two pieces of that decision (the SLOs and the objective metric), what the defaults do, and where the ranking happens in the pipeline.

## What is an SLO?

**SLO = Service Level Objective.** A latency target you commit to for production traffic. For LLM inference, the two standard ones are:

- **p99 TTFT** (Time To First Token) — "99% of requests get their first response token within X ms." How long the user waits before *anything* appears. Maps to perceived responsiveness.
- **p99 TPOT** (Time Per Output Token) — "99% of requests stream subsequent tokens at intervals ≤ Y ms." How smoothly the response streams after the first token. Maps to perceived fluency.

`p99` is the 99th percentile across many requests. The worst 1% may violate, the rest must meet target. SLOs are about *tail* behavior, not averages — users notice the slow requests.

Typical SLO values by use case:

| Use case | p99 TTFT | p99 TPOT | Note |
|---|---|---|---|
| Autocomplete / typing assistant | 200 ms | 30 ms | Very tight; humans abandon faster |
| Chatbot / coding assistant | 2000 ms | 100 ms | The current YAML defaults |
| Long-form summarization | 5000 ms | 200 ms | Tolerant; users expect to wait |
| Batch / offline | n/a | n/a | Disable SLOs; rank by raw throughput |

Configure SLOs in `objective.constraints`:

```yaml
objective:
  constraints:
    p99_ttft_ms: 2000
    p99_tpot_ms: 100
```

Set any constraint to `null` to disable it. Other metrics from `measurements.csv` (e.g. `p95_ttft_ms`, `median_ttft_ms`) work as constraint keys too.

## What does the sweep optimize?

The `objective.maximize` field names the single metric that defines "best". The default is `goodput_at_slo`:

```yaml
objective:
  maximize: goodput_at_slo
  constraints:
    p99_ttft_ms: 2000
    p99_tpot_ms: 100
```

**`goodput_at_slo`** is computed per (candidate, workload) as:

> `goodput_at_slo = median_output_tok_s_per_gpu` if **all** `objective.constraints` are met for that cell, else `0`.

In plain terms: throughput, but counted only when the candidate also meets the latency SLOs. A candidate that violates an SLO on a workload gets a goodput score of zero on that workload — it doesn't disappear from the report, but it can't win on that cell.

### Why this objective makes sense for serving

- Models the real deployment tradeoff: a config that's fast on average but breaks latency commitments is unshippable. SLO compliance is non-negotiable.
- Single number captures both *speed* and *quality-of-service*. Easy to rank.
- TTFT and TPOT together cover the two latency dimensions users perceive.

### Caveats to be aware of

- **Cliff edge.** A cell that exceeds the SLO by 1 ms scores zero, same as one that exceeds by 100 ms. No partial credit. Calibrate SLOs to your actual production targets, not stretch goals.
- **Per-deployment SLO.** Chatbot SLOs are much tighter than batch SLOs. Re-rank with your real numbers; the YAML defaults are a starting point.
- **Cost-blind.** Goodput is normalized per GPU, but a 4-GPU and 2-GPU winner score the same on this metric. The platform doesn't price-rank.
- **Heterogeneous traffic.** A single SLO across short and long prompts may under- or over-tune for either. Pick workloads in your sweep that reflect your real traffic mix.

### Other objectives you can use

Any column in `measurements.csv` is a valid `maximize` target (or its `median_` variant):

| Objective | When to use |
|---|---|
| `goodput_at_slo` | User-facing serving with a latency SLA (default) |
| `median_output_tok_s_per_gpu` | Batch / offline; no human waiting |
| `median_request_throughput` | When you care about requests/sec, not tokens/sec |
| `output_tok_s_per_gpu` (mean) | Same as median but mean-aggregated; more outlier-sensitive |

To rank by raw throughput with no SLO gate, set `maximize: median_output_tok_s_per_gpu` and `constraints: {p99_ttft_ms: null, p99_tpot_ms: null}`.

## How the ranking works

Ranking happens **before** the report is built. The pipeline:

1. **`moe-bench run`** launches each (backend, candidate, workload) cell and writes a `result.json` per cell.
2. **`moe-bench normalize`** maps backend-specific keys to a canonical schema in `measurements.csv`. One row per (candidate, workload, repeat).
3. **`moe-bench rank`** is where the optimization happens:
   - Groups `measurements.csv` by `(candidate_id, workload_key)`.
   - Aggregates across `execution.repeats` (median / mean / std / coefficient-of-variation per metric).
   - For each cell, checks `objective.constraints`. If any constraint fails, the cell's `valid_under_slo` becomes false and `goodput_at_slo` becomes 0.
   - Ranks candidates *per workload* by `objective.maximize`, breaking ties by lower `p99_tpot_ms`.
   - Writes `rankings.csv` (one row per `candidate × workload`, with `rank`, `relative_to_best`, and the aggregated metrics).
4. **`moe-bench report`** reads `rankings.csv` and computes the *global* winner across workloads:
   - For each candidate, takes the **geometric mean** of its per-workload `relative_to_best` scores.
   - Tiebreaks by median rank, then by win count.
   - The HTML report shows this global ordering as the "global winner" and the "robustness leaderboard". The constraint-conditional leaderboard re-applies the same logic in the browser with user-adjusted SLO sliders.

**Key implication:** the report is a presentation layer. Changing what's "best" means changing `objective.maximize` or `objective.constraints` in the YAML and re-running `rank` (or the whole `--report` pipeline). You cannot make a different candidate win by editing the HTML.

## Where to look for the numbers

After a run completes, the optimization is fully realized in three files under `results/<run_id>/`:

- **`measurements.csv`** — raw per-cell metrics. The source of truth.
- **`rankings.csv`** — per-(candidate, workload) rank under the configured objective. Read this to understand why a candidate won or lost on a specific workload.
- **`report.html` / `report.md`** — human-readable presentation of the same rankings, with the global winner up top.

If you want to see the same data ranked under a *different* objective without re-running the sweep:

```bash
# Edit objective.maximize or objective.constraints in results/<run_id>/config.yaml, then:
moe-bench rank results/<run_id>
moe-bench report results/<run_id>
```

`rank` and `report` are idempotent and re-read the YAML each time. The expensive part (the bench runs) is preserved.

## See also

- [`writing_a_sweep_config.md`](writing_a_sweep_config.md) — full tutorial including the `objective:` block
- [`parallelism_sweep.md`](parallelism_sweep.md) — what the resulting HTML report shows
- [`../CLAUDE.md`](../CLAUDE.md) — pipeline architecture (cli → runner → normalize → rank → report)
