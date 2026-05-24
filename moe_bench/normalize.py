from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .config import candidate_id
from .utils import read_json, write_json

CANONICAL_FIELDS = [
    "backend", "run_id", "candidate_id", "server_state_sha", "serve_config", "serve_params_json", "server_cmd", "workload", "input_len", "output_len", "max_concurrency", "request_rate", "seed", "repeat",
    "valid", "failure_reason", "completed", "failed", "duration_sec", "request_throughput", "output_throughput", "total_throughput",
    "output_tok_s_per_gpu", "total_tok_s_per_gpu", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "mean_e2e_ms", "p95_e2e_ms", "p99_e2e_ms", "source_path",
]


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    if s.lower() in {"inf", "infinity"}:
        return math.inf
    try:
        return float(s)
    except ValueError:
        return None


def to_int(v: Any) -> int | None:
    f = to_float(v)
    if f is None or math.isinf(f):
        return None
    return int(f)


def nested(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def workload_from_name(name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"prompt(\d+)", name or "")
    if m: out["input_len"] = int(m.group(1))
    m = re.search(r"(?:out|x)(\d+)", name or "")
    if m: out["output_len"] = int(m.group(1))
    m = re.search(r"(?:conc|batch)(\d+)", name or "")
    if m: out["max_concurrency"] = int(m.group(1))
    m = re.search(r"seed(\d+)", name or "")
    if m: out["seed"] = int(m.group(1))
    m = re.search(r"rep(\d+)", name or "")
    if m: out["repeat"] = int(m.group(1))
    return out


def normalize_backend_name(raw: Any, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    s = str(raw or "").lower()
    if "sglang" in s:
        return "sglang"
    if "vllm" in s:
        return "vllm"
    return s or "unknown"


def server_state_sha(backend: str, params: dict[str, Any], effective: dict[str, Any] | None = None) -> str:
    payload = {
        "backend": backend,
        "explicit_params": params or {},
        "effective_server_info": effective or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def derive_valid(row: dict[str, Any], status: dict[str, Any] | None = None) -> tuple[bool, str]:
    if status and status.get("failure_reason"):
        return False, str(status["failure_reason"])
    if status and status.get("valid") is False:
        return False, str(status.get("status") or "invalid")
    if row.get("failed") not in {None, ""} and (to_float(row.get("failed")) or 0) > 0:
        return False, "failed_requests"
    if row.get("completed") is None:
        return False, "missing_completed"
    if (to_float(row.get("completed")) or 0) <= 0:
        return False, "no_completed_requests"
    if row.get("output_throughput") is None:
        return False, "missing_output_throughput"
    if row.get("p99_ttft_ms") is None and row.get("mean_ttft_ms") is None:
        return False, "missing_ttft"
    if row.get("p99_tpot_ms") is None and row.get("mean_tpot_ms") is None:
        return False, "missing_tpot"
    return True, ""


def row_from_result(path: Path, data: dict[str, Any], gpus: int, backend: str | None = None, status: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None, run_id: str = "") -> dict[str, Any]:
    metadata = metadata or {}
    workload_meta = metadata.get("workload") if isinstance(metadata.get("workload"), dict) else {}
    sweep = data.get("_sweep") if isinstance(data.get("_sweep"), dict) else {}
    bench_params = sweep.get("bench_params") or data.get("bench_params") or {}
    serve_params = sweep.get("serve_params") or {}

    workload = str(workload_meta.get("name") or data.get("benchmark_name") or bench_params.get("_benchmark_name") or data.get("_benchmark_name") or path.parent.name)
    parsed = workload_from_name(workload)
    backend_name = normalize_backend_name(data.get("backend") or data.get("endpoint_type"), backend)
    serve_config = str(metadata.get("serve_config") or data.get("serve_config") or serve_params.get("_benchmark_name") or path.parent.parent.name)
    serve_meta = metadata.get("serve") if isinstance(metadata.get("serve"), dict) else {}
    backend_serve_params = serve_meta.get(backend_name) if isinstance(serve_meta.get(backend_name), dict) else serve_params
    if not backend_serve_params and data.get("server_args") is not None:
        backend_serve_params = {"server_args": data.get("server_args")}
    effective_server_info = data.get("server_info") if isinstance(data.get("server_info"), dict) else {}
    command_path = path.with_name("command.json")
    server_cmd = ""
    if command_path.exists():
        try:
            command_data = read_json(command_path)
            cmd_value = command_data.get("server_cmd") if isinstance(command_data, dict) else None
            if isinstance(cmd_value, list):
                server_cmd = " ".join(str(x) for x in cmd_value)
            elif cmd_value:
                server_cmd = str(cmd_value)
        except Exception:
            server_cmd = ""
    elif nested(data, "_sweep", "serve_cmd"):
        server_cmd = str(nested(data, "_sweep", "serve_cmd"))
    failed = data.get("failed_requests", data.get("failed"))
    if failed is None and data.get("errors") is not None and isinstance(data.get("errors"), list):
        failed = sum(1 for e in data["errors"] if e)

    row: dict[str, Any] = {
        "backend": backend_name,
        "run_id": run_id,
        "candidate_id": metadata.get("candidate_id") or (status or {}).get("candidate_id") or candidate_id(backend_name, backend_serve_params or {}),
        "server_state_sha": server_state_sha(backend_name, backend_serve_params or {}, effective_server_info),
        "serve_config": serve_config,
        "serve_params_json": json.dumps(backend_serve_params or {}, sort_keys=True, separators=(",", ":"), default=str),
        "server_cmd": server_cmd,
        "workload": workload,
        "input_len": workload_meta.get("input_len") or bench_params.get("random_input_len") or data.get("random_input_len") or parsed.get("input_len"),
        "output_len": workload_meta.get("output_len") or bench_params.get("random_output_len") or data.get("random_output_len") or parsed.get("output_len"),
        "max_concurrency": workload_meta.get("max_concurrency") or bench_params.get("max_concurrency") or data.get("max_concurrency") or parsed.get("max_concurrency"),
        "request_rate": workload_meta.get("request_rate") or bench_params.get("request_rate") or data.get("request_rate"),
        "seed": workload_meta.get("seed") if workload_meta.get("seed") is not None else bench_params.get("seed", parsed.get("seed")),
        "repeat": workload_meta.get("repeat", parsed.get("repeat", data.get("run_number", 0))),
        "completed": data.get("completed"),
        "failed": failed,
        "duration_sec": data.get("duration", data.get("elapsed_sec")),
        "request_throughput": data.get("request_throughput"),
        "output_throughput": data.get("output_throughput"),
        "total_throughput": data.get("total_throughput", data.get("total_token_throughput")),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "p50_ttft_ms": data.get("p50_ttft_ms", data.get("median_ttft_ms")),
        "p95_ttft_ms": data.get("p95_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": data.get("mean_tpot_ms"),
        "p50_tpot_ms": data.get("p50_tpot_ms", data.get("median_tpot_ms")),
        "p95_tpot_ms": data.get("p95_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "mean_e2e_ms": data.get("mean_e2el_ms", data.get("mean_e2e_latency_ms")),
        "p95_e2e_ms": data.get("p95_e2el_ms", data.get("p95_e2e_latency_ms")),
        "p99_e2e_ms": data.get("p99_e2el_ms", data.get("p99_e2e_latency_ms")),
        "source_path": str(path),
    }
    for k in ["input_len", "output_len", "max_concurrency", "seed", "repeat", "completed", "failed"]:
        row[k] = to_int(row.get(k))
    for k in ["duration_sec", "request_throughput", "output_throughput", "total_throughput", "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms", "mean_e2e_ms", "p95_e2e_ms", "p99_e2e_ms"]:
        row[k] = to_float(row.get(k))
    row["output_tok_s_per_gpu"] = (row["output_throughput"] / gpus) if row.get("output_throughput") is not None and gpus else None
    row["total_tok_s_per_gpu"] = (row["total_throughput"] / gpus) if row.get("total_throughput") is not None and gpus else None
    valid, reason = derive_valid(row, status)
    row["valid"] = valid
    row["failure_reason"] = reason
    return {k: row.get(k) for k in CANONICAL_FIELDS}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or CANONICAL_FIELDS
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def _load_params(params_json: str) -> dict[str, Any]:
    try:
        params = json.loads(params_json or "{}")
    except Exception:
        return {}
    return params if isinstance(params, dict) else {}


def _cmd_flag_value(cmd: str, flag: str) -> str | None:
    parts = str(cmd or "").split()
    if flag not in parts:
        return None
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


VLLM_CANDIDATE_PARAMS = [
    "vllm_tensor_parallel_size",
    "vllm_data_parallel_size",
    "vllm_enable_expert_parallel",
    "vllm_dtype",
    "vllm_moe_backend",
    "vllm_all2all_backend",
    "vllm_expert_placement_strategy",
    "vllm_max_num_seqs",
    "vllm_max_num_batched_tokens",
    "vllm_gpu_memory_utilization",
]

SGLANG_CANDIDATE_PARAMS = [
    "sglang_tp",
    "sglang_dp",
    "sglang_ep",
    "sglang_dtype",
    "sglang_attention_backend",
    "sglang_moe_runner_backend",
    "sglang_moe_a2a_backend",
    "sglang_max_running_requests",
    "sglang_chunked_prefill_size",
    "sglang_mem_fraction_static",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "server_state_sha",
    "backend",
    "serve_config",
    *VLLM_CANDIDATE_PARAMS,
    *SGLANG_CANDIDATE_PARAMS,
    "server_cmd",
]


def curated_candidate_params(backend: str, params: dict[str, Any], server_cmd: str) -> dict[str, Any]:
    """Return the small audited parameter set used to define a candidate.

    The full backend default surface is huge and unreadable. These are the
    parameters we expect researchers to tune first for MoE serving throughput /
    latency: parallelism, expert/MoE kernel/communication choices, dtype,
    memory budget, and scheduling/batching capacity.
    """
    if backend == "vllm":
        return {
            "vllm_tensor_parallel_size": params.get("tensor_parallel_size", 1),
            "vllm_data_parallel_size": params.get("data_parallel_size", 1),
            "vllm_enable_expert_parallel": params.get("enable_expert_parallel", False),
            "vllm_dtype": params.get("dtype") or _cmd_flag_value(server_cmd, "--dtype") or "auto",
            "vllm_moe_backend": params.get("moe_backend", "auto"),
            "vllm_all2all_backend": params.get("all2all_backend", "auto"),
            "vllm_expert_placement_strategy": params.get("expert_placement_strategy", "linear"),
            "vllm_max_num_seqs": params.get("max_num_seqs", "default"),
            "vllm_max_num_batched_tokens": params.get("max_num_batched_tokens", "auto"),
            "vllm_gpu_memory_utilization": params.get("gpu_memory_utilization", 0.9),
        }
    if backend == "sglang":
        return {
            "sglang_tp": params.get("tp", 1),
            "sglang_dp": params.get("dp", 1),
            "sglang_ep": params.get("ep", "none"),
            "sglang_dtype": params.get("dtype") or _cmd_flag_value(server_cmd, "--dtype") or "bfloat16",
            "sglang_attention_backend": params.get("attention_backend", "auto"),
            "sglang_moe_runner_backend": params.get("moe_runner_backend", "auto"),
            "sglang_moe_a2a_backend": params.get("moe_a2a_backend", "none"),
            "sglang_max_running_requests": params.get("max_running_requests", "default"),
            "sglang_chunked_prefill_size": params.get("chunked_prefill_size", "default"),
            "sglang_mem_fraction_static": params.get("mem_fraction_static", "default"),
        }
    return {}


def candidate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("candidate_id") or "")
        if not cid or cid in seen:
            continue
        backend = str(row.get("backend") or "")
        server_cmd = str(row.get("server_cmd") or "")
        params = _load_params(str(row.get("serve_params_json") or "{}"))
        seen[cid] = {
            "candidate_id": cid,
            "server_state_sha": row.get("server_state_sha") or cid,
            "backend": backend,
            "serve_config": row.get("serve_config"),
            **curated_candidate_params(backend, params, server_cmd),
            "server_cmd": server_cmd,
        }
    return list(seen.values()), CANDIDATE_FIELDS


def write_outputs(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(result_dir / "normalized.csv", rows)
    write_csv(result_dir / "measurements.csv", rows)
    failures = [r for r in rows if str(r.get("valid")).lower() != "true"]
    write_csv(result_dir / "failures.csv", failures)
    candidates, fields = candidate_rows(rows)
    write_csv(result_dir / "candidates.csv", candidates, fields)


def normalize_run(result_dir: str | Path, gpus: int | None = None) -> Path:
    result_dir = Path(result_dir)
    manifest = read_json(result_dir / "manifest.json") if (result_dir / "manifest.json").exists() else {}
    gpus = int(gpus or manifest.get("gpus") or 1)
    run_id = manifest.get("run_id", result_dir.name)
    rows: list[dict[str, Any]] = []
    for backend_dir in sorted((result_dir / "raw").glob("*")) if (result_dir / "raw").exists() else []:
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name
        for result_path in sorted(backend_dir.glob("*/*/result.json")):
            try:
                data = read_json(result_path)
            except Exception:
                data = {}
            status_path = result_path.with_name("status.json")
            metadata_path = result_path.with_name("metadata.json")
            status = read_json(status_path) if status_path.exists() else None
            metadata = read_json(metadata_path) if metadata_path.exists() else None
            rows.append(row_from_result(result_path, data, gpus, backend=backend, status=status, metadata=metadata, run_id=run_id))
    write_outputs(result_dir, rows)
    return result_dir / "normalized.csv"


def import_legacy(vllm_path: str | None, sglang_path: str | None, out: str | Path, gpus: int = 4) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": out.name, "gpus": gpus, "imported": True, "sources": {"vllm": vllm_path, "sglang": sglang_path}}
    write_json(out / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    if vllm_path:
        for p in sorted(Path(vllm_path).rglob("run=*.json")):
            try:
                rows.append(row_from_result(p, read_json(p), gpus, backend="vllm", run_id=out.name))
            except Exception:
                pass
    if sglang_path:
        for p in sorted(Path(sglang_path).rglob("run.json")):
            try:
                rows.append(row_from_result(p, read_json(p), gpus, backend="sglang", run_id=out.name))
            except Exception:
                pass
    write_outputs(out, rows)
    return out
