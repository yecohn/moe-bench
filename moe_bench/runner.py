from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import __version__
from .backends import make_backend
from .config import apply_cli_overrides, candidate_id, config_hash, enabled_backends, expand_serve_configs, expand_workloads, load_config
from .utils import detect_oom, run_capture, safe_name, tail_text, terminate_tree, utc_now, wait_ready, write_json


def make_run_id(cfg: dict[str, Any]) -> str:
    exp = cfg.get("experiment", {})
    if exp.get("run_id"):
        return str(exp["run_id"])
    stamp = utc_now().replace(":", "").replace("+0000", "Z")
    return safe_name(f"{exp.get('name', 'run')}-{stamp}")


def create_manifest(cfg: dict[str, Any], cfg_path: Path, run_id: str) -> dict[str, Any]:
    exp = cfg.get("experiment", {})
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "model": exp.get("model"),
        "gpus": exp.get("gpus"),
        "dtype": exp.get("dtype"),
        "started_at": utc_now(),
        "platform_version": __version__,
        "config_hash": config_hash(cfg_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "backends": {},
    }
    for name, bcfg in (cfg.get("backends") or {}).items():
        py = str((bcfg or {}).get("python", "python3"))
        rc, version = run_capture([py, "-c", f"import {('vllm' if name == 'vllm' else 'sglang')}; print({('vllm' if name == 'vllm' else 'sglang')}.__version__)"], timeout=30)
        manifest["backends"][name] = {
            "enabled": (bcfg or {}).get("enabled", True),
            "python": py,
            "version": version if rc == 0 else None,
            "version_error": None if rc == 0 else version,
        }
    rc, nvsmi = run_capture(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], timeout=30)
    manifest["nvidia_smi"] = nvsmi if rc == 0 else None
    return manifest


def dry_run(cfg: dict[str, Any], only_backends: str | None = None) -> None:
    exp = cfg.get("experiment", {}) or {}
    print("Model:", exp.get("model"))
    print("Run ID:", exp.get("run_id") or exp.get("name"))
    print("Dtype:", exp.get("dtype"))
    backends = enabled_backends(cfg, only_backends)
    workloads = expand_workloads(cfg)
    print("Backends:", ", ".join(backends))
    serve_configs = expand_serve_configs(cfg)
    print("Serve candidates:", len(serve_configs))
    print("Workloads:", len(workloads))
    print("Total cells:", sum(1 for b in backends for s in serve_configs if b in s) * len(workloads))
    for b in backends:
        for s in serve_configs:
            if b in s:
                print(f"  {b}/{s['name']}: {len(workloads)} workloads")


def run_sweep(
    config_path: str | Path,
    out_root: str | Path = "results",
    only_backends: str | None = None,
    dry: bool = False,
    *,
    model: str | None = None,
    run_id: str | None = None,
    served_model_name: str | None = None,
    dtype: str | None = None,
) -> Path:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    cfg = apply_cli_overrides(
        cfg,
        model=model,
        run_id=run_id,
        served_model_name=served_model_name,
        dtype=dtype,
    )
    if dry:
        dry_run(cfg, only_backends)
        return Path()

    run_id = make_run_id(cfg)
    result_dir = Path(out_root) / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, result_dir / "config.yaml")
    manifest = create_manifest(cfg, config_path, run_id)
    write_json(result_dir / "manifest.json", manifest)

    execution = cfg.get("execution", {})
    resume = bool(execution.get("resume", True))
    fail_fast = bool(execution.get("fail_fast", False))
    ready_timeout = int(execution.get("server_ready_timeout_sec", 900))
    bench_timeout = int(execution.get("bench_timeout_sec", 1800))
    cool_down = int(execution.get("cool_down_sec", 10))
    env = os.environ.copy()

    workloads = expand_workloads(cfg)
    serve_configs = expand_serve_configs(cfg)
    for backend_name in enabled_backends(cfg, only_backends):
        backend = make_backend(backend_name, cfg)
        for serve in serve_configs:
            if backend_name not in serve:
                continue
            serve_name = safe_name(serve["name"])
            backend_params = serve.get(backend_name) if isinstance(serve.get(backend_name), dict) else {}
            cid = candidate_id(backend_name, backend_params)
            serve_dir = result_dir / "raw" / backend_name / serve_name
            serve_dir.mkdir(parents=True, exist_ok=True)
            server_log = serve_dir / "server.log"
            server_cmd = backend.server_cmd(serve)
            write_json(serve_dir / "server_command.json", server_cmd)
            print(f"[moe-bench] start {backend_name}/{serve_name}")
            with server_log.open("w", encoding="utf-8") as lf:
                proc = subprocess.Popen(server_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            server_ready = False
            try:
                server_ready = wait_ready(backend.ready_url, ready_timeout)
                if not server_ready:
                    status = {
                        "backend": backend_name,
                        "serve_config": serve_name,
                        "candidate_id": cid,
                        "status": "server_ready_timeout",
                        "valid": False,
                        "failure_reason": "server_ready_timeout",
                        "server_log_tail": tail_text(server_log),
                        "started_at": utc_now(),
                    }
                    write_json(serve_dir / "status.json", status)
                    if fail_fast:
                        raise RuntimeError(f"server not ready: {backend_name}/{serve_name}")
                    continue
                write_json(serve_dir / "status.json", {"backend": backend_name, "serve_config": serve_name, "candidate_id": cid, "server_params": backend_params, "status": "ready", "valid": True})

                for workload in workloads:
                    run_dir = serve_dir / safe_name(workload["name"])
                    result_path = run_dir / "result.json"
                    status_path = run_dir / "status.json"
                    if resume and status_path.exists() and result_path.exists():
                        print(f"[moe-bench] skip existing {backend_name}/{serve_name}/{workload['name']}")
                        continue
                    run_dir.mkdir(parents=True, exist_ok=True)
                    bench_cmd, expected_result = backend.bench_cmd(workload, run_dir)
                    write_json(run_dir / "command.json", {"server_cmd": server_cmd, "bench_cmd": bench_cmd})
                    write_json(run_dir / "metadata.json", {"backend": backend_name, "serve_config": serve_name, "candidate_id": cid, "server_params": backend_params, "serve": serve, "workload": workload})
                    bench_log = run_dir / "bench.log"
                    started = time.time()
                    print(f"[moe-bench] bench {backend_name}/{serve_name}/{workload['name']}")
                    timeout_hit = False
                    with bench_log.open("w", encoding="utf-8") as bf:
                        try:
                            completed = subprocess.run(bench_cmd, stdout=bf, stderr=subprocess.STDOUT, env=env, timeout=bench_timeout)
                            rc = completed.returncode
                        except subprocess.TimeoutExpired:
                            timeout_hit = True
                            rc = 124
                    # SGLang writes JSONL; materialize result.json if needed.
                    parsed_result = backend.parse_result_path(run_dir)
                    log_tail = tail_text(bench_log)
                    failure = None
                    if timeout_hit:
                        failure = "bench_timeout"
                    elif rc != 0:
                        failure = "oom" if detect_oom(log_tail) else "bench_failed"
                    elif not parsed_result.exists():
                        failure = "missing_result"
                    status = {
                        "backend": backend_name,
                        "serve_config": serve_name,
                        "candidate_id": cid,
                        "server_params": backend_params,
                        "workload": workload,
                        "returncode": rc,
                        "valid": failure is None,
                        "failure_reason": failure,
                        "elapsed_sec": round(time.time() - started, 3),
                        "result_path": str(parsed_result),
                        "log_tail": log_tail if failure else None,
                    }
                    write_json(status_path, status)
                    if fail_fast and failure:
                        raise RuntimeError(f"benchmark failed: {backend_name}/{serve_name}/{workload['name']}: {failure}")
            finally:
                terminate_tree(proc)
                time.sleep(cool_down)

    manifest["finished_at"] = utc_now()
    write_json(result_dir / "manifest.json", manifest)
    print(result_dir)
    return result_dir
