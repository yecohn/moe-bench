## Problem Statement

The parallelism-axis sweep for Qwen3-30B-A3B on 4× A100-PCIE-40GB (`configs/moe_parallelism_4gpu.yaml`) failed end-to-end on its first attempt: all 11 candidates hit `server_ready_timeout` and the produced `report.html` has no usable data. The result is that we still cannot answer the question this sweep was designed to answer — *which TP/DP/EP layout is best for this model on this hardware* — and roughly an hour of wall-clock plus operator attention was wasted.

Root-cause investigation showed two distinct problems, only one of which the original fix-up plan in `context.md` correctly diagnosed:

1. **The first vLLM candidate (`vllm_tp4dp1`) failed at startup because vLLM's per-rank memory check at `gpu_memory_utilization=0.90` is incompatible with the NCCL P2P buffers that the other ranks have already allocated on cuda:3 by the time worker 3 runs its check.** The "Free memory on device cuda:3 (34.35/39.49 GiB) … less than desired (0.9, 35.54 GiB)" error is not caused by a pre-existing orphan process — it is caused by vLLM's own multi-rank initialization. The `context.md` write-up attributed this to a leftover `VLLM::EngineCore` from a prior session, which the per-candidate `server.log` evidence contradicts.
2. **The two pure-DP vLLM candidates (`vllm_tp1dp4`, `vllm_tp1dp4_ep`) fundamentally cannot fit.** Qwen3-30B-A3B is 57 GB in bfloat16; replicating the full model on each 40 GB GPU is impossible. These candidates would always fail and should never have been in the candidate list for this hardware.

Once the first candidate failed mid-initialization, leaked workers held ~23 GiB on cuda:3 and ~6 GiB on the other GPUs, putting subsequent candidates well under the 0.90 threshold and producing a cascade of identical-looking `server_ready_timeout` failures. This made the failure pattern look uniform and obscured the actual trigger.

The user needs the parallelism sweep to run to completion so they can compare the realistic deployment layouts for this model and decide which backend + config to ship.

## Solution

A small, targeted patch to `configs/moe_parallelism_4gpu.yaml` plus a one-shot pre-flight cleanup, then re-running the existing driver script.

From the user's perspective:

- The sweep config is edited in place to (a) remove the infeasible TP=1 DP=4 candidates, and (b) drop both engines' GPU memory utilization knobs from 0.90 to 0.85 so vLLM's per-rank check passes after NCCL P2P buffers are allocated.
- Before launching, the user runs a one-line `sudo` kill against the known root-owned orphan from the prior session.
- The existing script (`scripts/run_parallelism_sweep.sh`) is re-invoked unchanged. Because `execution.resume: true` is set and the previous failures left no per-workload `result.json` files, every remaining candidate is re-attempted from scratch in the same run directory.
- The driver finishes by running `normalize`, `rank`, and `report`, producing a usable `report.html` covering 9 candidates × 6 workloads = 54 cells in roughly 1 hour.

Server-ready timeout, cool-down, the cleanup script, and the SGLang `attention_backend` choices remain untouched. The grill explicitly considered and rejected each of those changes because none was implicated in the actual failure.

## User Stories

1. As a benchmark operator, I want the parallelism sweep to run end-to-end without a server-startup cascade, so that I get a usable `report.html` from a single invocation.
2. As a benchmark operator, I want infeasible candidates (model larger than per-GPU memory under the chosen layout) excluded from the config, so that I do not burn ~40 minutes of timeout per impossible candidate.
3. As a benchmark operator, I want vLLM's `gpu_memory_utilization` set low enough that the NCCL P2P allocation does not break the per-rank free-memory check on first init, so that the first candidate of a TP=4 sweep does not die at startup.
4. As a benchmark operator, I want SGLang's `mem_fraction_static` lowered symmetrically with vLLM's GMU, so that the two backends' memory headroom is comparable and the SGLang phase has the same resilience to startup residue.
5. As a benchmark operator, I want the existing run directory `results/qwen3-a3b-4gpu-parallelism/` reused via `execution.resume: true`, so that I do not have to wipe state or remember a new `run_id`, and so the previous failed-candidate dirs are naturally overwritten when each server is retried.
6. As a benchmark operator, I want any root-owned GPU orphans from the prior session removed before the script runs, so that the script's cleanup step does not need to escalate to `sudo -n` mid-run and there is no residual GPU memory holding back the first candidate.
7. As a benchmark operator, I want the stray `vllm serve Qwen/Qwen3-1.7B` server on port 8000 stopped, so that it does not pin GPU memory or compete with the sweep, even though the script's `pkill -f 'vllm serve'` would also catch it.
8. As a benchmark operator, I want the YAML header comment updated to reflect 9 candidates × 6 workloads = 54 cells (down from 11 × 6 = 66), so that the config self-documents the post-fix shape and so anyone reading it does not look for the missing TP=1 DP=4 candidates.
9. As a sweep designer, I want the SGLang candidates to keep their per-layout attention backend choices (default / flashinfer for EP / triton for DP-attention), so that each layout is benchmarked in the configuration it would actually be deployed under, with the confound documented rather than removed.
10. As a sweep designer, I want `server_ready_timeout_sec` left at 1200, so that we do not cargo-cult a higher timeout that had no relevance to the actual failure (all crashes were within ~60 seconds of launch).
11. As a sweep designer, I want the existing cleanup script left as-is, so that we do not add complexity (post-cleanup idle gates, sudo retries) that the verified-working `sudo -n` fallback and the GMU fix already obviate.
12. As a future reader of the run, I want the report to clearly omit the dropped TP=1 DP=4 candidates rather than show them as failures, so that the leaderboard reflects only candidates that were ever feasible on this hardware.
13. As a future reader of `context.md`, I want the misdiagnosis (orphan-caused first failure) corrected to the actual cause (NCCL P2P + GMU=0.90), so that the next person hitting a similar symptom looks at the right knob first.
14. As a benchmark operator, I want the post-run `normalize`, `rank`, and `report` steps to run automatically as part of the driver script, so that I do not need to remember to invoke them manually after the GPU work completes.
15. As a benchmark operator, I want the final HTML report to include all 9 surviving candidates with valid per-workload measurements, so that I can use the constraint-slider leaderboard and Pareto plots to pick a deployment configuration.

## Implementation Decisions

- **Scope is a config patch, not a code change.** No Python files, no shell scripts, no new modules. The fix lives entirely in `configs/moe_parallelism_4gpu.yaml` plus one manual `sudo` kill.
- **Drop the two pure-DP vLLM candidates** (`vllm_tp1dp4`, `vllm_tp1dp4_ep`) from `serve_configs`. The header comment's "vLLM layouts" block is updated to list only the 4 surviving (TP, DP) ∈ {(4,1), (2,2)} × EP {off, on} candidates.
- **Lower `gpu_memory_utilization` from 0.90 to 0.85** in the `&vllm_common` anchor, so all surviving vLLM candidates inherit the change.
- **Lower `mem_fraction_static` from 0.90 to 0.85** in the `&sglang_common` anchor, symmetric with the vLLM change.
- **Keep `execution.server_ready_timeout_sec: 1200`**. The grill rejected a bump to 1800 because no candidate ever ran long enough for the existing timeout to be the limiting factor.
- **Keep `execution.cool_down_sec: 10`**. The runner's `terminate_tree` already waits up to 20s grace before SIGKILL, giving 30s total between candidates — sufficient for NCCL handle release on graceful shutdown.
- **Keep `execution.resume: true` and the existing `run_id: qwen3-a3b-4gpu-parallelism`**. Verified via code inspection (`runner.py` resume check at the workload level requires both `status.json` and `result.json`) that failed candidates from the prior run will be cleanly re-attempted. Stale `status.json` files at the candidate level are overwritten on retry; orphan subdirs for the removed candidates are silently ignored by `normalize` (which globs for `*/*/result.json`).
- **Update the header comment** to state 9 candidates × 6 workloads = 54 cells, target ~1h, and to drop the TP=1 DP=4 lines from the vLLM-layouts list.
- **Pre-flight kill** of orphan PID 2755203 (`sudo -n kill -9 2755203`) is a manual one-shot, not codified in the script. Passwordless sudo is verified working on this host.
- **The unrelated `vllm serve Qwen/Qwen3-1.7B` on port 8000 (PID 2753718) is killed.** The driver script's existing `pkill -TERM -f 'vllm serve'` matches it; no extra step is required, but operators may also kill it manually for cleanliness.
- **Leave `scripts/run_parallelism_sweep.sh` unchanged.** Its current cleanup function (pkill → nvidia-smi-driven kill → sudo -n kill -9 fallback → log) is sufficient given the verified-working `sudo -n` and the fact that the real failure was a config issue, not a script issue. No post-cleanup idle gate is added.
- **No new objective, no new ranking rule, no new report section.** Out of scope; covered by separate items in `context.md`.

## Testing Decisions

This change is too small and too operational for a unit-test layer to be meaningful. The only thing that matters is whether the patched config actually produces a complete, valid `report.html`. Verification is the run itself, with these acceptance signals:

- **All 9 candidates reach `status: ready`** in their candidate-level `status.json`. Concretely: none of `vllm_tp4dp1`, `vllm_tp4dp1_ep`, `vllm_tp2dp2`, `vllm_tp2dp2_ep`, `sglang_tp4`, `sglang_tp4_ep`, `sglang_tp4_dpattn`, `sglang_tp4_dpattn_ep`, `sglang_tp4_dpattn2_ep` ends with `server_ready_timeout`.
- **All 54 (candidate × workload) cells produce a `result.json`** under `results/qwen3-a3b-4gpu-parallelism/raw/<backend>/<serve>/<workload>/`. The `failures.csv` is empty (or, at worst, contains a small number of `bench_failed` / `oom` cells that are individually investigable rather than a uniform cascade).
- **The first vLLM candidate's `server.log`** does NOT contain the "Free memory on device cuda:X … less than desired GPU memory utilization" line. This is the smoking gun the GMU=0.85 change is designed to eliminate.
- **`report.html` opens in a browser** and shows the global-winner, constraint-slider leaderboard, Pareto, and per-workload drilldown sections populated with all 9 candidates.

There is no test suite in this repo (see `CLAUDE.md`); the precedent for validating changes is "dry-run plus a small real config end-to-end and inspect `report.md`." This change follows that precedent: the validation is the real sweep run.

If the sweep fails again, the diagnostic procedure is fixed: read the *first* failed candidate's `server.log` (`grep -E "Free memory|OutOfMemory|Killed"`) and decide whether to (a) lower GMU further, (b) drop more candidates, or (c) revisit a non-config issue.

## Out of Scope

- **`scripts/run_parallelism_sweep.sh` hardening.** Adding a post-cleanup "GPUs must be idle" gate was considered and rejected: the cleanup function already works (sudo -n verified), and the original cascade was a config issue, not a cleanup issue.
- **`server_ready_timeout_sec` bump.** Considered and rejected — no candidate was time-limited.
- **Wiping `results/qwen3-a3b-4gpu-parallelism/` before re-running.** Not needed; resume semantics handle it cleanly.
- **Substituting a 2-GPU TP=1 DP=2 candidate for the dropped TP=1 DP=4 ones.** Would not use all 4 GPUs and would not be directly comparable to the 4-GPU candidates.
- **Normalizing SGLang `attention_backend` across candidates.** The current choice (default / flashinfer for EP / triton for DP-attention) reflects realistic per-layout best practice; documenting the confound is preferred over removing it.
- **The bigger pending design items in `context.md`** (configurable `report:` block, new report sections like `failure_summary` / `workload_group_winners` / `slo_compliance`, `top_k_candidate_cards` cap, robustness zero-floor fix). Each is its own PRD.
- **Correcting the misdiagnosis in `context.md`.** Worth doing as a follow-up edit so the next operator looks at GMU, not orphans, but not part of this PRD.
- **Capturing the NCCL P2P startup overhead as a sweep-design lesson** (e.g. a `gpu_memory_utilization` upper-bound recommendation in the docs). Useful but separate.

## Further Notes

- The 0.85 value is empirically motivated: with `Free memory = 34.35 GiB` observed on cuda:3 after NCCL P2P init on a clean GPU, `0.85 × 39.49 = 33.57 GiB` passes the check with ~0.78 GiB headroom. A more conservative value (0.80) would work too but reduces KV-cache budget unnecessarily; `max_num_batched_tokens=8192` already caps the active KV window, so 0.85 leaves more than enough.
- The per-GPU model memory at the heaviest layout (TP=2 DP=2) is ~28.5 GiB. At 0.85 × 40 = 34 GiB cap, this leaves ~5.5 GiB for KV cache plus activations; comfortable for the 8192-token batched window.
- If the first vLLM candidate still fails at 0.85, the next-step lever is 0.80, then dropping `max_num_seqs` to 64. Both are quick edits if needed.
- The grill explicitly verified `sudo -n` works on this host (exit 0), so the script's escalation path is functional even if a future root-owned orphan appears.
- The pre-flight kill of PID 2755203 is best-effort hygiene; if that PID no longer exists by the time the operator runs the command, the run still proceeds (the script's cleanup function will handle whatever it finds).
- The `context.md` write-up attributed the first failure to "GPU 3 had ~5 GB residual on GPU 3 from a prior orphan `VLLM::EngineCore` left over after the quick sweep cancellation." The actual log (`results/qwen3-a3b-4gpu-parallelism/raw/vllm/vllm_tp4dp1/server.log`) shows the cleanup script reported all GPUs at 0 MiB immediately before launch, and vLLM's own multi-rank init explains the missing ~5 GiB. The 23 GiB orphan observed at session end was the *aftermath* of the cascade, not its trigger. Worth correcting in `context.md` so the next operator does not chase the wrong fix.
