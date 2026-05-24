from pathlib import Path

import yaml

from moe_bench.cli import main as cli_main
from moe_bench.config import apply_cli_overrides


def _base_config() -> dict:
    return {
        "experiment": {
            "name": "qwen3-a3b-4gpu-parallelism",
            "run_id": "qwen3-a3b-4gpu-parallelism",
            "model": "Qwen/Qwen3-30B-A3B",
            "dtype": "bfloat16",
            "gpus": 4,
        },
        "backends": {
            "vllm": {"served_model_name": "moe-bench-qwen3-a3b"},
            "sglang": {},
        },
        "serve_configs": [
            {"name": "vllm_tp4dp1", "vllm": {"dtype": "bfloat16", "tensor_parallel_size": 4}},
            {"name": "sglang_tp4", "sglang": {"dtype": "bfloat16", "tp": 4}},
        ],
        "workloads": {"input_lens": [256], "output_lens": [128], "concurrencies": [4]},
    }


def test_model_override_replaces_experiment_model():
    cfg = _base_config()
    out = apply_cli_overrides(cfg, model="meta-llama/Llama-3-8B")
    assert out["experiment"]["model"] == "meta-llama/Llama-3-8B"


def test_run_id_override_updates_both_run_id_and_name():
    cfg = _base_config()
    out = apply_cli_overrides(cfg, run_id="llama-3-8b-4gpu-parallelism")
    assert out["experiment"]["run_id"] == "llama-3-8b-4gpu-parallelism"
    assert out["experiment"]["name"] == "llama-3-8b-4gpu-parallelism"


def test_served_model_name_override_updates_vllm_backend():
    cfg = _base_config()
    out = apply_cli_overrides(cfg, served_model_name="moe-bench-llama-3-8b")
    assert out["backends"]["vllm"]["served_model_name"] == "moe-bench-llama-3-8b"


def test_served_model_name_override_is_noop_without_vllm_backend():
    cfg = _base_config()
    del cfg["backends"]["vllm"]
    out = apply_cli_overrides(cfg, served_model_name="moe-bench-llama-3-8b")
    assert "vllm" not in out["backends"]


def test_dtype_override_updates_experiment_dtype():
    cfg = _base_config()
    out = apply_cli_overrides(cfg, dtype="float16")
    assert out["experiment"]["dtype"] == "float16"


def test_dtype_override_cascades_into_every_candidate_backend_subdict():
    cfg = _base_config()
    out = apply_cli_overrides(cfg, dtype="float16")
    assert out["serve_configs"][0]["vllm"]["dtype"] == "float16"
    assert out["serve_configs"][1]["sglang"]["dtype"] == "float16"
    # Non-dtype keys in the same subdicts are untouched.
    assert out["serve_configs"][0]["vllm"]["tensor_parallel_size"] == 4
    assert out["serve_configs"][1]["sglang"]["tp"] == 4


def test_all_none_args_return_equal_but_independent_config():
    cfg = _base_config()
    out = apply_cli_overrides(cfg)
    assert out == cfg
    # Mutating the returned dict must not affect the input.
    out["experiment"]["model"] = "MUTATED"
    out["serve_configs"][0]["vllm"]["dtype"] = "MUTATED"
    assert cfg["experiment"]["model"] == "Qwen/Qwen3-30B-A3B"
    assert cfg["serve_configs"][0]["vllm"]["dtype"] == "bfloat16"


def test_cli_run_with_overrides_reaches_dry_run_output(tmp_path: Path, capsys):
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump(_base_config()))
    rc = cli_main([
        "run", str(cfg_path),
        "--model", "meta-llama/Llama-3-8B",
        "--run-id", "llama-3-8b-sweep",
        "--served-model-name", "moe-bench-llama-3-8b",
        "--dtype", "float16",
        "--dry-run",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "meta-llama/Llama-3-8B" in out
    assert "llama-3-8b-sweep" in out
