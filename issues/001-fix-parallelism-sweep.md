## Parent PRD

`issues/prd.md`

## What to build

Apply the targeted fix-up that makes the Qwen3-30B-A3B 4-GPU parallelism sweep run end-to-end and produce a usable `report.html`. The work is a single end-to-end change with three parts that must all land together to be demoable:

1. **Pre-flight cleanup** — `sudo -n kill -9 2755203` to remove the root-owned `VLLM::EngineCore` orphan left on GPU 3 from the prior session. Hygiene only; the actual fix is the config change below.
2. **Config patch** to `configs/moe_parallelism_4gpu.yaml` — drop the two infeasible pure-DP vLLM candidates (`vllm_tp1dp4`, `vllm_tp1dp4_ep`), lower `gpu_memory_utilization` in `&vllm_common` from `0.90` → `0.85` so vLLM's per-rank free-memory check passes after NCCL P2P init, lower `mem_fraction_static` in `&sglang_common` symmetrically, and update the header comment to read 9 candidates × 6 workloads = 54 cells.
3. **Run and validate** — invoke the existing `scripts/run_parallelism_sweep.sh` driver. With `execution.resume: true` and the same `run_id`, the prior failed candidates will be naturally re-attempted in `results/qwen3-a3b-4gpu-parallelism/`. Driver finishes by running `normalize`, `rank`, `report`.

See the parent PRD for the full root-cause analysis (NCCL P2P + GMU=0.90, not the orphan) and the rationale for each rejected alternative (timeout bump, script hardening, dir wipe).

## Acceptance criteria

- [ ] `configs/moe_parallelism_4gpu.yaml` no longer contains `vllm_tp1dp4` or `vllm_tp1dp4_ep` entries.
- [ ] `&vllm_common` has `gpu_memory_utilization: 0.85`; `&sglang_common` has `mem_fraction_static: 0.85`.
- [ ] Header comment in the YAML reflects 9 candidates × 6 workloads = 54 cells, and the TP=1 DP=4 lines are removed from the layout list.
- [ ] After the run, every candidate-level `status.json` under `results/qwen3-a3b-4gpu-parallelism/raw/{vllm,sglang}/*/status.json` reads `"status": "ready"` (not `server_ready_timeout`).
- [ ] `results/qwen3-a3b-4gpu-parallelism/raw/vllm/vllm_tp4dp1/server.log` does NOT contain the string `Free memory on device cuda`. This is the smoking-gun line the GMU change is designed to eliminate.
- [ ] All 54 (candidate × workload) cells produce a per-workload `result.json`. `failures.csv` is empty, or contains only individually-investigable cells (no uniform cascade).
- [ ] `results/qwen3-a3b-4gpu-parallelism/report.html` opens in a browser and shows the global-winner, constraint-slider leaderboard, Pareto, and per-workload drilldown sections populated with all 9 candidates.
- [ ] Wall-clock from script start to `report.html` written is on the order of 1 hour (acceptance is "completes", not a hard budget).

## Blocked by

None — can start immediately.

## User stories addressed

Reference by number from the parent PRD:

- User story 1
- User story 2
- User story 3
- User story 4
- User story 5
- User story 6
- User story 7
- User story 8
- User story 9
- User story 10
- User story 11
- User story 12
- User story 14
- User story 15

(User story 13 — correcting the misdiagnosis in `context.md` — is listed in the PRD as Out of Scope for this issue; it is a follow-up edit.)

## Progress note

**2026-05-24 — config-patch phase complete; awaiting operator sweep.**

Landed in `configs/moe_parallelism_4gpu.yaml`:

- Dropped `vllm_tp1dp4` and `vllm_tp1dp4_ep` candidate blocks entirely.
- `&vllm_common.gpu_memory_utilization: 0.90 → 0.85`.
- `&sglang_common.mem_fraction_static: 0.90 → 0.85`.
- Header comment updated: "vLLM layouts (4 candidates)", TP=1 DP=4 line replaced with an infeasibility note, "Total cells: 9 candidates x 6 workloads = 54". Added a paragraph explaining the NCCL P2P / GMU=0.90 root cause so the rationale lives next to the value.
- `execution.server_ready_timeout_sec` and `execution.cool_down_sec` left at 1200 / 10 as decided in the PRD.

Verification done in this pass:

- `yaml.safe_load` of the file succeeds; candidate count = 9; GMU = 0.85 in both anchors; ready_timeout still 1200.
- `moe-bench run … --dry-run` reports "Serve candidates: 9 / Workloads: 6 / Total cells: 54" with the expected per-candidate line-up.

**What still needs to happen (operator):**

1. `sudo -n kill -9 2755203` to clear the root-owned `VLLM::EngineCore` orphan on GPU 3 (hygiene; the real fix is the GMU change).
2. Optionally `kill 2753718` to stop the unrelated `vllm serve Qwen/Qwen3-1.7B` on port 8000 (the script's `pkill -f 'vllm serve'` will catch it anyway).
3. Run `scripts/run_parallelism_sweep.sh`. With `execution.resume: true` and the same `run_id`, the previously failed candidates will be naturally re-attempted in `results/qwen3-a3b-4gpu-parallelism/`.
4. After completion, confirm acceptance: every candidate-level `status.json` reads `"ready"`; `results/qwen3-a3b-4gpu-parallelism/raw/vllm/vllm_tp4dp1/server.log` no longer contains "Free memory on device cuda"; 54 per-workload `result.json` files exist; `report.html` opens with all 9 candidates populated.

This issue should be moved to `issues/done/` only after step 4 passes. If the first vLLM candidate still fails with the same free-memory error, the next lever is GMU 0.80 (see PRD Further Notes).

Not committed: repo is not a git repository at `/mnt/projects/AI/josh/moe-bench`, so the skill's commit step is N/A.
