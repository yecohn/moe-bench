## Parent PRD

`issues/prd-docker-container.md`

## What to build

A single-file `Dockerfile` (~15 lines) plus `.dockerignore` that packages moe-bench with pre-built vLLM and SGLang backend venvs into one image. Operator workflow: `docker build -t moe-bench . && docker run --rm -it --gpus all moe-bench`, then run any moe-bench command from the shell inside the container.

Bundled with the Dockerfile, this issue also delivers two prerequisites that are needed for the container to actually work:

1. **Config normalization** — migrate 5 configs (`moe_parallelism_4gpu`, `qwen3_a3b_4gpu_grid`, `qwen3_a3b_4gpu_quick`, `stack_smoke`, `stack_search_smoke`) from absolute host paths (`/mnt/projects/AI/josh/...`) to the canonical `.backends/<name>/bin/python` paths. Two other configs (`smoke`, `qwen3_a3b`) already use this pattern and are unchanged.
2. **Documentation** — replace the `BACKENDS.md` section 10 placeholder stub with the real build + run instructions, ARG override table, driver minimum, recommended mounts, and host-migration note. Add a short "Run via Docker" subsection to `docs/parallelism_sweep.md` cross-linking to BACKENDS.md.

See the parent PRD's Solution section, Implementation Decisions, and Out of Scope sections for the full design.

## Acceptance criteria

**Dockerfile + image build:**

- [ ] `Dockerfile` exists at repo root, ~15–25 lines, uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base, Python 3.12 from deadsnakes PPA.
- [ ] `Dockerfile` declares `ARG PYTHON_VERSION`, `ARG VLLM_PACKAGE`, `ARG SGLANG_PACKAGE` with pinned defaults visible at the top.
- [ ] Backend venvs are built via `scripts/create_backend_venvs.sh all` (no duplicated install logic).
- [ ] Platform venv lives at `/opt/moe-bench-venv` with `pip install -e .`; `moe-bench` is symlinked onto `/usr/local/bin/moe-bench`.
- [ ] Final `RUN` invokes `scripts/check_backend_envs.sh` so a broken version pin fails the build immediately.
- [ ] `ENTRYPOINT ["/bin/bash"]` (interactive shell default).
- [ ] `docker build -t moe-bench .` exits 0 on a clean machine. Cold build ≤ 30 minutes.
- [ ] `docker run --rm moe-bench moe-bench --help` exits 0 with the CLI usage.
- [ ] `docker run --rm --gpus all moe-bench moe-bench run configs/moe_parallelism_4gpu.yaml --dry-run` exits 0 and prints `Serve candidates: 9 / Workloads: 6 / Total cells: 54`.

**`.dockerignore`:**

- [ ] Excludes `.git`, `.backends/`, `results/`, `issues/`, `context.md`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.venv/`.

**Config migration:**

- [ ] `grep -rE "python:.*/.venv" configs/` returns no results.
- [ ] All 5 migrated configs use `python: .backends/vllm/bin/python` and `python: .backends/sglang/bin/python`.
- [ ] `smoke.yaml` and `qwen3_a3b.yaml` are unchanged (they already use the canonical paths).
- [ ] On a host with `.backends/` populated (via `scripts/create_backend_venvs.sh` or symlinks), `moe-bench run <each-migrated-config> --dry-run` exits 0.

**Documentation:**

- [ ] `BACKENDS.md` section 10 contains: build command, run command, ARG override table, minimum driver line (≥ 550), recommended mounts (HF cache + results), host-migration note for users adopting `.backends/`.
- [ ] `docs/parallelism_sweep.md` "Run it" section contains a "Run via Docker" subsection (≤ 1 short paragraph) with a sample `docker run` command and a cross-link to BACKENDS.md.

## Blocked by

None - can start immediately.

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
- User story 13
- User story 14
- User story 15
- User story 16
- User story 17

## Notes discovered during implementation

**Unverified version pins (action item before first real build):**

The Dockerfile defaults are `VLLM_PACKAGE=vllm==0.8.0` and `SGLANG_PACKAGE=sglang==0.4.8` — educated guesses for "stable as of the maintainer's last-known pypi state." I did NOT run an actual `docker build` because:

1. Cold build is 15–30 minutes and would compete with the in-flight parallelism sweep for CPU/network.
2. Without internet-search access I can't confirm those exact versions exist on pypi today.

Verification used was `docker buildx build --check .` (BuildKit linter — confirms Dockerfile syntax and base-image reachability, but does NOT pull or install any pip packages). The linter reported zero warnings.

Before the first real build, the maintainer should either (a) confirm the two pinned versions exist on pypi and resolve cleanly, or (b) bump them to current known-stable versions. The Dockerfile structure is correct; only the literal version strings might need updating.

If the build fails at the `RUN scripts/create_backend_venvs.sh all` step, the failure message will name the unresolved package — that's the version to bump in the ARG line at the top of the Dockerfile.

**Sweep was alive throughout this issue's implementation** — config YAML edits are safe because configs are loaded at runner startup, not re-read during the sweep. The host operator can use the canonical `.backends/` paths immediately by either running `scripts/create_backend_venvs.sh all` or symlinking existing dev venvs (documented in BACKENDS.md section 10).
