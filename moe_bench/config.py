from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml


def apply_cli_overrides(
    config: dict[str, Any],
    *,
    model: str | None = None,
    run_id: str | None = None,
    served_model_name: str | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(config)
    if model is not None:
        out.setdefault("experiment", {})["model"] = model
    if run_id is not None:
        exp = out.setdefault("experiment", {})
        exp["run_id"] = run_id
        exp["name"] = run_id
    if served_model_name is not None:
        vllm = out.get("backends", {}).get("vllm")
        if isinstance(vllm, dict):
            vllm["served_model_name"] = served_model_name
    if dtype is not None:
        out.setdefault("experiment", {})["dtype"] = dtype
        for serve in out.get("serve_configs", []) or []:
            for backend_name, backend_params in list(serve.items()):
                if backend_name == "name" or not isinstance(backend_params, dict):
                    continue
                if "dtype" in backend_params:
                    backend_params["dtype"] = dtype
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg, path)
    return cfg


def validate_config(cfg: dict[str, Any], path: Path | None = None) -> None:
    where = f" in {path}" if path else ""
    for key in ["experiment", "backends", "workloads"]:
        if key not in cfg:
            raise ValueError(f"Missing required key `{key}`{where}")
    if "serve_configs" not in cfg and "search_space" not in cfg:
        raise ValueError(f"Missing required key `serve_configs` or `search_space`{where}")
    if "serve_configs" in cfg:
        if not isinstance(cfg["serve_configs"], list) or not cfg["serve_configs"]:
            raise ValueError("serve_configs must be a non-empty list")
        for sc in cfg["serve_configs"]:
            if "name" not in sc:
                raise ValueError("every serve config needs a name")
    wl = cfg["workloads"]
    for key in ["input_lens", "output_lens", "concurrencies"]:
        if key not in wl:
            raise ValueError(f"workloads.{key} is required")


def config_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def enabled_backends(cfg: dict[str, Any], only: str | None = None) -> list[str]:
    requested = {b.strip() for b in only.split(",")} if only else None
    out: list[str] = []
    for name, info in cfg.get("backends", {}).items():
        if requested is not None and name not in requested:
            continue
        if info is None or info.get("enabled", True):
            out.append(name)
    return out


def candidate_id(backend: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"backend": backend, "params": params or {}}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _expand_backend_search_space(backend: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(spec.get("defaults", {})) if isinstance(spec.get("defaults"), dict) else {}
    grid = dict(spec.get("grid", spec))
    grid.pop("defaults", None)
    grid.pop("grid", None)
    names = list(grid.keys())
    values = [_as_list(grid[k]) for k in names]
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        params = dict(defaults)
        params.update({k: v for k, v in zip(names, combo) if v is not None})
        cid = candidate_id(backend, params)
        rows.append({"name": f"{backend}_{cid}", "_generated_from_search_space": True, backend: params})
    return rows


def expand_serve_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return explicit serve configs plus generated search-space candidates.

    `serve_configs` are optimization candidates. `search_space` is optional
    shorthand to generate many candidates from backend-specific parameter grids.
    """
    rows = list(cfg.get("serve_configs", []) or [])
    for backend, spec in (cfg.get("search_space") or {}).items():
        if not isinstance(spec, dict):
            continue
        rows.extend(_expand_backend_search_space(backend, spec))
    return rows


def expand_workloads(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    wl = cfg["workloads"]
    repeats = int(cfg.get("execution", {}).get("repeats", wl.get("repeats", 1)))
    seeds = wl.get("seeds", [0])
    rows: list[dict[str, Any]] = []
    for input_len, output_len, conc, seed, repeat in itertools.product(
        wl.get("input_lens", []),
        wl.get("output_lens", []),
        wl.get("concurrencies", []),
        seeds,
        range(repeats),
    ):
        name = f"prompt{input_len}_out{output_len}_conc{conc}_seed{seed}_rep{repeat}"
        rows.append({
            "name": name,
            "input_len": int(input_len),
            "output_len": int(output_len),
            "max_concurrency": int(conc) if conc is not None else None,
            "request_rate": wl.get("request_rate", "inf"),
            "num_prompts": int(wl.get("num_prompts", 256)),
            "num_warmups": int(wl.get("num_warmups", 0)),
            "seed": int(seed) if seed is not None else None,
            "repeat": int(repeat),
            "random_range_ratio": wl.get("random_range_ratio", "0.0"),
        })
    return rows
