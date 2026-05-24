## Problem Statement

A new operator who wants to run `moe-bench` on their own 4×A100 (or comparable) machine currently has to: install Python, create a platform venv, install moe-bench in it, install Python and a CUDA toolkit chain to build vLLM and SGLang venvs, populate `.backends/`, point YAML configs at the right Python interpreters, and only then run the sweep. The setup is documented in `BACKENDS.md` and works, but it's manual, friction-heavy, and the "happy path" depends on the operator's host already having a compatible NVIDIA driver, the right CUDA system libraries, and matching Python versions.

Concrete pain points:

1. **Backend env setup is the longest pole.** Even with `--only-binary=:all:` and prebuilt wheels, the operator has to know about Python version compatibility, the `.backends/` convention, and the per-backend wheel installs. First-time setup commonly fails on Python-version mismatches or missing CUDA system libs.
2. **YAML configs hardcode absolute host paths.** 5 of 7 checked-in configs (`moe_parallelism_4gpu`, `qwen3_a3b_4gpu_grid`, `qwen3_a3b_4gpu_quick`, `stack_smoke`, `stack_search_smoke`) point `backends.<name>.python` at `/mnt/projects/AI/josh/vllm/.venv/bin/python` and the SGLang equivalent. These paths only exist on one developer's workstation. Anyone else who clones the repo gets a config that's dead on arrival until they hand-edit the python paths.
3. **The `BACKENDS.md` section 10 placeholder** already acknowledges this: "A prebuilt Docker image can avoid Python dependency installation entirely… For v1, use prebuilt wheels in `.backends/`. Container-based backends can be added later." That "later" is now.
4. **No reproducibility story for the version stack.** vLLM and SGLang both move fast; what works for one operator on a given week may not for another a week later. The host workflow today uses dev builds from local git checkouts that aren't reproducible without coordinating commit SHAs.

The job is to ship a Dockerfile that an operator with a Linux box + 4 GPUs + Docker + nvidia-container-toolkit can clone the repo and run `docker build && docker run` to get a working sweep environment, without ever touching pip, venv, or absolute paths.

## Solution

A single-file `Dockerfile` (~15 lines) plus a `.dockerignore`, packaging the moe-bench platform and pre-built vLLM + SGLang backend venvs into one image. Reuses the existing `scripts/create_backend_venvs.sh` so the build steps inside the container are the same code that already runs on hosts; this keeps the two paths in sync and avoids duplicating install logic.

From the operator's perspective:

```bash
docker build -t moe-bench .
docker run --rm -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/moe-bench/results \
  moe-bench
# inside the container:
moe-bench run configs/moe_parallelism_4gpu.yaml --report
```

The image:

- Ships **pinned-but-overridable** vLLM and SGLang package versions via build args (`ARG VLLM_PACKAGE=vllm==X`, `ARG SGLANG_PACKAGE=sglang==Y`). Operators can `docker build --build-arg VLLM_PACKAGE=vllm==Z ...` to bump.
- Uses **CUDA 12.4 runtime** as base (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`). Documented minimum NVIDIA driver: ≥ 550.
- Builds **two backend venvs at `.backends/{vllm,sglang}/bin/python`**, the canonical location per `BACKENDS.md`.
- Builds a **platform venv at `/opt/moe-bench-venv`** with `pip install -e .` and symlinks `moe-bench` onto `/usr/local/bin/moe-bench` so it's on PATH.
- Defaults to an **interactive shell** (`ENTRYPOINT ["/bin/bash"]`). The operator chooses what to run; we don't bake in a specific sweep.
- Ends the build with `scripts/check_backend_envs.sh` so a broken version pin fails the build immediately rather than at first `docker run`.

Alongside the Dockerfile, **5 model-specific configs are migrated** from absolute host paths to `.backends/<name>/bin/python` so they work both in the container and on hosts where the operator has populated `.backends/` (the BACKENDS.md-documented convention). Two already-migrated configs (`smoke.yaml`, `qwen3_a3b.yaml`) are unchanged. This is a small backwards-incompatible change to the host workflow for anyone who previously relied on the absolute paths; they get a one-time migration: either run `scripts/create_backend_venvs.sh` or symlink their existing dev venv into `.backends/`.

Documentation lives in `BACKENDS.md` section 10 (replacing the existing placeholder stub) and a short subsection in `docs/parallelism_sweep.md` cross-linking to it. No new top-level docs file.

Mounts, HuggingFace authentication, and model weight caching are intentionally not baked into the image. The image is mount-agnostic; the docs *suggest* bind-mounting `~/.cache/huggingface` and `results/` but don't require it. `HF_TOKEN` and any other env-based auth are the operator's responsibility, surfaced as `-e HF_TOKEN=...` in the example commands.

## User Stories

1. As a newcomer with 4 A100s and Docker installed, I want to run two shell commands (`docker build`, `docker run`) and have a working `moe-bench` environment, so I don't have to learn Python venvs, CUDA wheel indices, or absolute paths before I can produce my first report.
2. As an operator running on a clean host, I want the image to ship pre-built vLLM and SGLang venvs, so the first `docker run` is ready to launch a sweep without further install steps.
3. As an operator with a specific vLLM or SGLang version requirement, I want to override the pinned versions at build time via `--build-arg`, so I can target a specific release without forking the Dockerfile.
4. As a maintainer of moe-bench, I want the Docker build steps to reuse `scripts/create_backend_venvs.sh`, so install logic for the two backends lives in exactly one place and the host + container paths stay in sync as the script evolves.
5. As a reviewer of a Docker build that picked up a broken vLLM version pin, I want the build to fail with a clear backend-import error at build time (via `scripts/check_backend_envs.sh`), so I never ship an image where `import vllm` fails at first `docker run`.
6. As an operator who already has a populated `~/.cache/huggingface` on the host, I want documentation that recommends bind-mounting it into the container, so a 57 GiB model download doesn't happen on every fresh `docker run`.
7. As an operator with results from a previous container run, I want documentation that recommends bind-mounting `$(pwd)/results` into the container, so my `report.html` survives container exit and I can `cd results/<run_id>/` on the host to inspect.
8. As an operator on a host with an older NVIDIA driver, I want the docs to clearly state the minimum driver version (≥ 550 for CUDA 12.4 wheels), so I don't waste time debugging at `docker run` when the cause is a driver mismatch.
9. As an operator running interactively, I want `docker run -it moe-bench` to drop me into a bash shell at the moe-bench source tree, so I can experiment with multiple sweep configs from one container session.
10. As an operator running the parallelism sweep, I want `moe-bench run configs/moe_parallelism_4gpu.yaml --report` to work out of the box inside the container, so the canonical sweep is the canonical happy-path demo.
11. As a host-workflow user (no Docker), I want the 5 currently-broken configs to work after running `scripts/create_backend_venvs.sh` (or after symlinking my existing venvs into `.backends/`), so the same configs serve both paths without duplication.
12. As a host-workflow user who previously relied on the absolute paths in `moe_parallelism_4gpu.yaml`, I want the migration documented as a one-time setup step, so I know exactly what changed and how to adapt.
13. As a CI reviewer, I want `.dockerignore` to exclude `results/`, `.git/`, `.backends/`, `issues/`, `context.md`, and `__pycache__/`, so the build context is small and the image doesn't accidentally bake in someone's stale run artifacts.
14. As an operator who just ran a Docker build that took 20 minutes, I want subsequent builds to be fast (seconds) when no version pin or source file changed, so iterating on documentation or moe-bench platform code isn't a 20-minute round-trip.
15. As an operator running on a fresh, internet-less host, I want a clear error message when `docker build` can't reach pypi for vLLM/SGLang wheels, so I know to set up a pip index mirror or pre-fetch the wheels rather than guess at network issues.
16. As an operator reading `BACKENDS.md`, I want section 10 to contain real, copy-pasteable build/run commands instead of the "v1 placeholder" stub, so the documentation describes the actual shipping state of the project.
17. As an operator reading `docs/parallelism_sweep.md`, I want a one-paragraph "Run via Docker" note in the "Run it" section linking to BACKENDS.md, so the per-sweep docs surface the Docker option without duplicating the build instructions.

## Implementation Decisions

### Image structure

- Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. CUDA runtime image (not devel) because all backend installs are prebuilt wheels; no compiler needed.
- Python 3.12 via the deadsnakes PPA. Matches the BACKENDS.md "Python 3.10-3.12 is usually safest" guidance and the host workflow's `python3.12` default.
- Single-stage Dockerfile (not multi-stage). The earlier multi-stage proposal was scope creep; a linear ~15-line Dockerfile that reuses `scripts/create_backend_venvs.sh` is honest about the work and easier to read.
- WORKDIR `/app/moe-bench`. Repo source copied here.
- Backend venvs at `/app/moe-bench/.backends/{vllm,sglang}/bin/python` (the canonical BACKENDS.md location, identical inside and outside the container).
- Platform venv at `/opt/moe-bench-venv` with `pip install -e .` against the source copied to `/app/moe-bench/`. Symlink `/opt/moe-bench-venv/bin/moe-bench` to `/usr/local/bin/moe-bench` so it's on PATH.

### Build-time configuration

Three ARGs, each with pinned defaults visible at the top of the Dockerfile:

- `ARG PYTHON_VERSION=3.12`
- `ARG VLLM_PACKAGE=vllm==<pinned-version>` — passed to `create_backend_venvs.sh` via env var.
- `ARG SGLANG_PACKAGE=sglang==<pinned-version>` — same.

Operators override via `docker build --build-arg VLLM_PACKAGE=vllm==0.8.0 -t moe-bench .`. The default values are checked into the Dockerfile for reproducibility; bumping is a one-line edit.

The Dockerfile does NOT add a `CUDA_BASE` ARG; the base image FROM is hardcoded. Bumping CUDA is a deliberate edit, not a build-time override, because the wheel/CUDA compatibility is non-trivial.

### Backend install

`scripts/create_backend_venvs.sh all` is invoked once during build. It:

- Creates the two venvs at the canonical `.backends/{vllm,sglang}/` location.
- Installs `--only-binary=:all:` against pypi (no source builds, no surprise multi-hour compiles).
- Honors the `VLLM_PACKAGE` / `SGLANG_PACKAGE` env vars set from the Dockerfile ARGs.

After the install, `scripts/check_backend_envs.sh` runs as the final `RUN` step. It imports `vllm`, `vllm.entrypoints.cli.main`, `sglang`, `sglang.launch_server`, `sglang.bench_serving` and prints versions. No GPU is needed (pure Python imports). Build fails if either backend doesn't import.

### Config normalization

5 configs migrated from absolute host paths to `.backends/<name>/bin/python`:

- `configs/moe_parallelism_4gpu.yaml`
- `configs/qwen3_a3b_4gpu_grid.yaml`
- `configs/qwen3_a3b_4gpu_quick.yaml`
- `configs/stack_smoke.yaml`
- `configs/stack_search_smoke.yaml`

`configs/smoke.yaml` and `configs/qwen3_a3b.yaml` already use the canonical paths and are unchanged. The migration is a mechanical sed of two strings; no semantic changes elsewhere in the YAMLs.

### Mount-agnostic image

The Dockerfile does not declare `VOLUME` directives. The operator chooses what to mount; the docs recommend two bind-mounts (HF cache, results) but neither is required. `HF_TOKEN` is operator-supplied via `-e HF_TOKEN=...`; the image does not configure HF auth in any way.

### Documentation

Two doc updates:

1. **`BACKENDS.md` section 10** — replace the placeholder stub with the actual build/run instructions, ARG override table, minimum driver note, recommended mounts, and a one-paragraph "host migration" note for users adopting the new `.backends/` convention.
2. **`docs/parallelism_sweep.md` "Run it" section** — add a "Run via Docker" subsection (≤ 1 paragraph) with a sample `docker run` command and a cross-link to BACKENDS.md.

No new top-level docs file. No README.md changes in this PRD (the README already points at BACKENDS.md for backend setup).

### `.dockerignore`

Excludes from build context: `.git`, `.backends/`, `results/`, `issues/`, `context.md`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.venv/`. Keeps the build context small (no committed venvs or runtime artifacts) and avoids accidentally baking in someone's stale run data.

### Explicitly NOT in this PRD

- Multi-stage Dockerfile, layer-caching optimizations, multi-arch builds, registry publishing, Compose files. Re-add later if anyone needs them.
- Image variants (vllm-only, sglang-only). Both backends ship together.
- Pre-downloading model weights into the image. Operators bind-mount their HF cache.
- HF auth handling. Operator supplies via env var; docs mention it as a recommendation, image does nothing.
- `tini` as PID 1 init. Listed during grilling, deferred.
- Backend-as-container architecture (where the moe-bench runner shells out to `docker run vllm-server` instead of a local Python). Out of scope; the runner still uses the canonical local-Python interface.

## Testing Decisions

This is a packaging change; the existing pytest suite is unaffected and continues to be the test layer for code changes. Verification of the Docker work is via build + smoke run on a real GPU host.

### What makes a good test here

- **End-to-end behavior, not internal mechanics.** Tests assert "build succeeds and produces an image that can dry-run the parallelism sweep", not "the Dockerfile contains a `pip install` step on line N."
- **Cheap iteration.** Verification steps should run in seconds-to-minutes, not require a 1-hour sweep. A real sweep is the ultimate validation; the test plan covers the Docker-specific surface short of that.

### Verification steps (in acceptance criteria order)

For the **Dockerfile + image build**:

- `docker build -t moe-bench .` exits 0 on a clean machine. Cold build ≤ 30 min.
- The final `RUN scripts/check_backend_envs.sh` line prints expected backend versions and "import ok" for both. A broken pin makes the build fail here.
- `docker run --rm moe-bench moe-bench --help` exits 0 with the CLI usage. Confirms platform venv + PATH symlink.
- `docker run --rm --gpus all moe-bench moe-bench run configs/moe_parallelism_4gpu.yaml --dry-run` exits 0 and prints "Serve candidates: 9 / Workloads: 6 / Total cells: 54". Confirms backend venvs are reachable from inside the container at their canonical paths.

For the **config migration**:

- `grep -rE "python:.*\.venv" configs/` returns no results (all absolute paths gone).
- `moe-bench run configs/<each-migrated-config>.yaml --dry-run` on a host with `.backends/` populated exits 0. Same on a host with symlinks.
- The host's currently-running parallelism sweep (started before this PRD) continues to function — the config is loaded at startup, so an edit after launch doesn't affect it.

For the **doc updates**:

- `BACKENDS.md` section 10 contains the canonical `docker build` and `docker run` commands and a "minimum driver ≥ 550" line.
- `docs/parallelism_sweep.md` "Run it" section contains a Docker subsection that cross-links to BACKENDS.md.

### Tests deliberately NOT written

- No automated CI for the Docker build. The repo has no CI today and this PRD doesn't introduce it.
- No unit test that "asserts the Dockerfile contains `nvidia/cuda:12.4`". Would couple the test to implementation; the acceptance criteria above are higher-leverage.
- No host-OS portability matrix. Targeting Linux + Docker + nvidia-container-toolkit. Mac/Windows are out of scope (no GPU).
- No multi-arch build. x86_64 only. A100s are x86_64.

## Out of Scope

- **Backend-as-container architecture.** Where the moe-bench runner spins up per-candidate Docker containers for vLLM/SGLang servers. BACKENDS.md section 10 hints at this as a future enhancement. Not part of this PRD; the runner continues to shell out to a local Python interpreter (which lives inside the same container as the platform).
- **Compose or Kubernetes manifests.** A single `docker run` is the supported workflow.
- **Pushing the image to a public registry.** Local build only. Operators rebuild on their hosts.
- **Pre-baking model weights into the image.** Operators bind-mount their HF cache.
- **Image variants** (vllm-only, sglang-only, platform-only). One image, both backends.
- **Pin updates as a recurring task.** The ARG defaults are pinned once; bumping is a manual edit by the maintainer, triggered by the operator's `--build-arg` if they need a newer version.
- **A `BENCH_REQUIRE_REAL_REPORT_REGEN` end-to-end test** (the upstream `HTML_REPORT_SKIPPED.txt` silent-failure mode flagged in issue 006's notes). Worth its own follow-up issue; not part of this Docker PRD.
- **The K8s pod that competes with the sweep on the host** (the `qwen3-1.7b-deployment-vllm` issue surfaced during the live sweep run). Operational issue with shared infrastructure, not a code change in moe-bench. Out of scope.

## Further Notes

- **Defaults for `VLLM_PACKAGE` and `SGLANG_PACKAGE`** are educated guesses for "stable as of the maintainer's last-known pypi state." The maintainer should bump them once before the first real build to match what's actually current on pypi. The Dockerfile structure (ARGs at top) makes this a one-line change.
- **The `--only-binary=:all:` guard** from `create_backend_venvs.sh` is preserved automatically since the Dockerfile calls the script unchanged. A pinned version without a wheel for the target CUDA fails fast; this is intentional.
- **Iterating on platform code** (the moe-bench Python package): fast. The backend-install layers are cached across rebuilds as long as ARGs don't change. Only the COPY layer and the platform-venv `pip install -e .` layer rebuild.
- **Iterating on pinned versions**: slow. Bumping `VLLM_PACKAGE` invalidates the backend-install layer; cold rebuild required.
- **Operators on hosts without `--only-binary=:all:` cooperation** (e.g., a custom pypi mirror that serves source distributions): set `PIP_ONLY_BINARY=""` via `--build-arg` and the script honors it. Documented in BACKENDS.md.
- **The host's currently-running sweep** (PID 326075, started 15:30 UTC) is unaffected by this PRD: configs are loaded at startup, and editing the YAML on disk doesn't change the in-flight run. After the sweep completes, the host operator should run `scripts/create_backend_venvs.sh` (or symlink) so subsequent runs work against the migrated configs.
