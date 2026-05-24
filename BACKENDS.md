# Building backend environments

`moe-bench` is intentionally separate from the inference engines. The platform can run from one Python environment, while each backend runs from its own environment.

Recommended layout:

```text
/mnt/projects/AI/josh/
  moe-bench/
  vllm/
  sglang/
```

Backend venvs are created under:

```text
moe-bench/.backends/vllm/
moe-bench/.backends/sglang/
```

The default configs already point to:

```yaml
backends:
  vllm:
    python: .backends/vllm/bin/python
  sglang:
    python: .backends/sglang/bin/python
```

Run `moe-bench` commands from the `moe-bench` directory when using these relative paths.

## 1. Platform environment

Use your existing platform env, or create one:

```bash
source /mnt/projects/AI/josh/aiq/.venv/bin/activate
cd /mnt/projects/AI/josh/moe-bench
pip install -e .
```

This env only needs the benchmark platform dependencies. It does **not** need vLLM/SGLang installed.

## 2. Create backend envs with prebuilt wheels

Default mode installs prebuilt Python packages/wheels and avoids local compilation:

```bash
./scripts/create_backend_venvs.sh all
```

This creates:

```text
.backends/vllm/bin/python
.backends/sglang/bin/python
```

and runs roughly:

```bash
.backends/vllm/bin/python -m pip install --only-binary=:all: vllm
.backends/sglang/bin/python -m pip install --only-binary=:all: sglang
```

`--only-binary=:all:` is intentional: if pip would need to compile from source, installation fails instead of silently spending hours building CUDA/Rust/C++ extensions.

You can build only one backend:

```bash
./scripts/create_backend_venvs.sh vllm
./scripts/create_backend_venvs.sh sglang
```

## 3. Verify backend envs

```bash
./scripts/check_backend_envs.sh
```

Expected output includes Torch and backend versions plus successful imports of:

- `vllm.entrypoints.cli.main`
- `sglang.launch_server`
- `sglang.bench_serving`

## 4. Editable source repo paths

Repo paths are only used with `INSTALL_MODE=editable`. If the repos are elsewhere:

```bash
INSTALL_MODE=editable \
VLLM_REPO=/path/to/vllm \
SGLANG_REPO=/path/to/sglang/python \
./scripts/create_backend_venvs.sh all
```

## 5. Custom Python version

Use a Python version compatible with backend dependencies. Python 3.10-3.12 is usually safest for CUDA/Torch stacks.

```bash
PYTHON_BIN=python3.12 ./scripts/create_backend_venvs.sh all
```

## 6. Pin package versions

For reproducibility, pin backend package versions once you choose them:

```bash
VLLM_PACKAGE='vllm==0.x.y' \
SGLANG_PACKAGE='sglang==0.x.y' \
./scripts/create_backend_venvs.sh all
```

If the desired package version is hosted on a custom CUDA/nightly index, pass extra pip flags:

```bash
VLLM_PIP_INSTALL_ARGS='--extra-index-url https://download.pytorch.org/whl/cu128' \
SGLANG_PIP_INSTALL_ARGS='--extra-index-url https://download.pytorch.org/whl/cu128' \
./scripts/create_backend_venvs.sh all
```

## 7. Disable the binary-only guard

Not recommended, but useful if a pure-Python dependency has no wheel:

```bash
PIP_ONLY_BINARY= ./scripts/create_backend_venvs.sh all
```

## 8. Editable installs from local repos

Use this only when you need local source changes. It may compile backend extensions:

```bash
INSTALL_MODE=editable ./scripts/create_backend_venvs.sh all
```

This installs:

- vLLM editable from `../vllm`
- SGLang editable from `../sglang/python`

## 9. Skip installation and only create venvs

Useful if you want to manually install packages yourself:

```bash
INSTALL_MODE=skip ./scripts/create_backend_venvs.sh all
```

Then manually install prebuilt packages, for example:

```bash
.backends/vllm/bin/python -m pip install --only-binary=:all: vllm
.backends/sglang/bin/python -m pip install --only-binary=:all: sglang
```

## 10. Docker image

A `Dockerfile` at the repo root packages moe-bench with pre-built vLLM and SGLang backend venvs into a single image. Use it when you'd rather not deal with host Python / CUDA setup.

### Prerequisites

- Linux host with NVIDIA driver **≥ 550** (CUDA 12.4 wheels) and at least one supported GPU.
- Docker with the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so `--gpus all` works.

### Build

```bash
docker build -t moe-bench .
```

Cold build is 15–30 minutes (mostly pip-downloading the backend wheels). Subsequent builds are seconds when source changes but version pins don't.

### Run

```bash
docker run --rm -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/results:/app/moe-bench/results \
  moe-bench
```

This drops you into a bash shell at `/app/moe-bench`. From there:

```bash
moe-bench run configs/moe_parallelism_4gpu.yaml --report
# or, with the GPU-cleanup-between-backends wrapper:
scripts/run_parallelism_sweep.sh
```

Output lands in `results/<run_id>/report.html`. If you bind-mounted `results/` (as in the example above) the report is visible from the host at `$(pwd)/results/<run_id>/report.html`.

### Overriding versions at build time

| Build arg | Default | What it pins |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | Python in both venvs |
| `VLLM_PACKAGE` | `vllm==0.8.0` | Passed to `scripts/create_backend_venvs.sh` |
| `SGLANG_PACKAGE` | `sglang==0.4.8` | Passed to `scripts/create_backend_venvs.sh` |

Example: bump vLLM and rebuild:

```bash
docker build --build-arg VLLM_PACKAGE=vllm==0.9.0 -t moe-bench .
```

The defaults are checked into the Dockerfile for reproducibility; if a default doesn't resolve against pypi the build fails at the backend-install step.

### Recommended mounts

The image itself doesn't declare volumes; the operator chooses. Two bind-mounts are useful for almost every run:

- `~/.cache/huggingface` → `/root/.cache/huggingface`: reuses any model weights you already downloaded on the host. For Qwen3-30B-A3B that's a 57 GiB savings on every fresh `docker run`.
- `$(pwd)/results` → `/app/moe-bench/results`: persists sweep output across container exit; lets you `cd results/<run_id>/` on the host to inspect.

### Hugging Face authentication

Not handled by the image. Pass `-e HF_TOKEN=$HF_TOKEN` at `docker run` if your model needs it; both vLLM and SGLang honor `HF_TOKEN` automatically.

### Host-workflow note

The checked-in configs (`moe_parallelism_4gpu.yaml`, the grid/quick variants, the stack smoke configs, plus `smoke.yaml` and `qwen3_a3b.yaml`) point at `.backends/<name>/bin/python` — the canonical location built by `scripts/create_backend_venvs.sh`. The container builds them at exactly that path. If you previously ran the configs on the host with absolute paths to dev venvs, you now have two options:

```bash
# Option A: build .backends/ on the host
scripts/create_backend_venvs.sh all

# Option B: symlink existing dev venvs into .backends/
mkdir -p .backends
ln -s /path/to/your/vllm/.venv .backends/vllm
ln -s /path/to/your/sglang/python/.venv .backends/sglang
```

Then verify with `scripts/check_backend_envs.sh`.

### What's NOT in the image

- Model weights — operator mounts the HF cache.
- HF auth — operator supplies via env var.
- Multi-stage build optimizations, registry publishing, Compose, K8s manifests. One image, local build, single `docker run`.
- Per-backend container architecture (runner shelling out to `docker run vllm-server ...` per candidate). Possible future enhancement; not how this image works today.

## 11. Using existing backend envs

If the source repos already have working envs, you can skip `.backends` and edit the config:

```yaml
backends:
  vllm:
    python: /mnt/projects/AI/josh/vllm/.venv/bin/python
  sglang:
    python: /mnt/projects/AI/josh/sglang/python/.venv/bin/python
```

Then verify manually:

```bash
/mnt/projects/AI/josh/vllm/.venv/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'
/mnt/projects/AI/josh/sglang/python/.venv/bin/python -c 'import sglang, torch; print(getattr(sglang, "__version__", "unknown"), torch.__version__)'
```

## 12. Smoke test after backend setup

```bash
source /mnt/projects/AI/josh/aiq/.venv/bin/activate
cd /mnt/projects/AI/josh/moe-bench
moe-bench run configs/smoke.yaml --dry-run
```

A real smoke run requires GPUs and model access:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 moe-bench run configs/smoke.yaml --report
```
