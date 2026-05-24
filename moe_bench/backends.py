from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import read_json


def _kebab(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_kv_args(cmd: list[str], params: dict[str, Any]) -> list[str]:
    for key, value in params.items():
        if key in {"extra_args", "args", "raw_args", "server_args"} or key.startswith("_"):
            continue
        flag = _kebab(key)
        if isinstance(value, bool):
            cmd.append(flag if value else "--no-" + key.replace("_", "-"))
        elif value is None:
            continue
        else:
            cmd += [flag, str(value)]
    return cmd


def _pop_extra_args(params: dict[str, Any]) -> list[str]:
    """Return raw CLI args while keeping `extra_args` as the preferred name.

    `args`, `raw_args`, and `server_args` remain accepted for backward
    compatibility with early configs and imported examples.
    """
    for key in ("extra_args", "raw_args", "args", "server_args"):
        if key in params:
            return list(params.pop(key) or [])
    return []


class Backend:
    name = "base"

    def __init__(self, cfg: dict[str, Any], backend_cfg: dict[str, Any]):
        self.cfg = cfg
        self.backend_cfg = backend_cfg
        exp = cfg.get("experiment", {})
        self.model = exp.get("model")
        self.host = exp.get("host", "127.0.0.1")
        self.dtype = exp.get("dtype", "bfloat16")
        self.python = backend_cfg.get("python", "python3")
        self.port = int(backend_cfg.get("port", 8000))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ready_url(self) -> str:
        endpoint = self.backend_cfg.get("ready_endpoint", "/v1/models")
        return self.base_url + endpoint

    def server_cmd(self, serve: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def bench_cmd(self, workload: dict[str, Any], run_dir: Path) -> tuple[list[str], Path]:
        raise NotImplementedError

    def parse_result_path(self, run_dir: Path) -> Path:
        return run_dir / "result.json"


class VllmBackend(Backend):
    name = "vllm"

    def server_cmd(self, serve: dict[str, Any]) -> list[str]:
        params = dict(serve.get("vllm") or {})
        served = self.backend_cfg.get("served_model_name", "moe-bench")
        dtype = params.pop("dtype", self.dtype)
        cmd = [
            str(self.python), "-m", "vllm.entrypoints.cli.main", "serve", str(self.model),
            "--served-model-name", str(served),
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", str(dtype),
            "--trust-remote-code",
        ]
        extra_args = _pop_extra_args(params)
        _append_kv_args(cmd, params)
        cmd += extra_args
        return cmd

    def bench_cmd(self, workload: dict[str, Any], run_dir: Path) -> tuple[list[str], Path]:
        result = run_dir / "result.json"
        served = self.backend_cfg.get("served_model_name", "moe-bench")
        cmd = [
            str(self.python), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm",
            "--endpoint", "/v1/completions",
            "--model", str(served),
            "--tokenizer", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "--dataset-name", "random",
            "--ignore-eos",
            "--temperature", "0",
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "50,90,95,99",
            "--random-input-len", str(workload["input_len"]),
            "--random-output-len", str(workload["output_len"]),
            "--random-range-ratio", str(workload.get("random_range_ratio", "0.0")),
            "--num-prompts", str(workload.get("num_prompts", 256)),
            "--num-warmups", str(workload.get("num_warmups", 0)),
            "--request-rate", str(workload.get("request_rate", "inf")),
            "--save-result",
            "--result-dir", str(run_dir),
            "--result-filename", result.name,
        ]
        if workload.get("max_concurrency") is not None:
            cmd += ["--max-concurrency", str(workload["max_concurrency"])]
        if workload.get("seed") is not None:
            cmd += ["--seed", str(workload["seed"])]
        return cmd, result


class SglangBackend(Backend):
    name = "sglang"

    def __init__(self, cfg: dict[str, Any], backend_cfg: dict[str, Any]):
        super().__init__(cfg, backend_cfg)
        if "port" not in backend_cfg:
            self.port = 30000

    def server_cmd(self, serve: dict[str, Any]) -> list[str]:
        params = dict(serve.get("sglang") or {})
        dtype = params.pop("dtype", self.dtype)
        cmd = [
            str(self.python), "-m", "sglang.launch_server",
            "--model-path", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", str(dtype),
            "--trust-remote-code",
        ]
        extra_args = _pop_extra_args(params)
        _append_kv_args(cmd, params)
        cmd += extra_args
        return cmd

    def bench_cmd(self, workload: dict[str, Any], run_dir: Path) -> tuple[list[str], Path]:
        jsonl = run_dir / "bench.jsonl"
        result = run_dir / "result.json"
        cmd = [
            str(self.python), "-m", "sglang.bench_serving",
            "--host", self.host,
            "--port", str(self.port),
            "--model", str(self.model),
            "--tokenizer", str(self.model),
            "--output-file", str(jsonl),
            "--backend", "sglang-oai",
            "--dataset-name", "random",
            "--num-prompts", str(workload.get("num_prompts", 256)),
            "--random-input-len", str(workload["input_len"]),
            "--random-output-len", str(workload["output_len"]),
            "--random-range-ratio", str(workload.get("random_range_ratio", "0.0")),
            "--request-rate", str(workload.get("request_rate", "inf")),
            "--warmup-requests", str(workload.get("num_warmups", 0)),
            "--ready-check-timeout-sec", "0",
            "--disable-tqdm",
            "--output-details",
        ]
        if workload.get("max_concurrency") is not None:
            cmd += ["--max-concurrency", str(workload["max_concurrency"])]
        if workload.get("seed") is not None:
            cmd += ["--seed", str(workload["seed"])]
        return cmd, result

    def parse_result_path(self, run_dir: Path) -> Path:
        result = run_dir / "result.json"
        jsonl = run_dir / "bench.jsonl"
        if result.exists() or not jsonl.exists():
            return result
        lines = [ln for ln in jsonl.read_text(errors="replace").splitlines() if ln.strip()]
        if lines:
            data = json.loads(lines[-1])
            with result.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.write("\n")
        return result


def make_backend(name: str, cfg: dict[str, Any]) -> Backend:
    bcfg = cfg.get("backends", {}).get(name, {}) or {}
    if name == "vllm":
        return VllmBackend(cfg, bcfg)
    if name == "sglang":
        return SglangBackend(cfg, bcfg)
    raise ValueError(f"Unsupported backend: {name}")
