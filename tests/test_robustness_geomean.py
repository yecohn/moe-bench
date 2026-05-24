"""Tests for the zero-floored geomean in compute_robustness.

These tests pin the behavior that an SLO-violating candidate (with zero
relative_to_best scores on most workloads) must rank below a candidate
that passes SLO on all workloads with reasonable scores. They also pin
the no-op behavior when no zeros are present, and the configurability of
the zero_floor knob.

Following the project's existing test style (see test_config_overrides.py):
pytest, no fixtures beyond small local helpers, no conftest.
"""
from moe_bench.html_report import _geomean, compute_robustness


def _wkl(idx: int) -> dict:
    """Build a synthetic workload identity for a ranking row."""
    return {
        "workload": f"w{idx}",
        "input_len": 256,
        "output_len": 128,
        "max_concurrency": idx + 1,
    }


def _ranking_row(cid: str, idx: int, relative_to_best: float, rank: int) -> dict:
    row = _wkl(idx)
    row["candidate_id"] = cid
    row["relative_to_best"] = relative_to_best
    row["rank"] = rank
    return row


def _slo_violator_inputs() -> tuple[list[dict], list[dict]]:
    """Shared fixture: candidate X wins one workload but scores 0 on five;
    candidate Y scores 0.8 on every workload."""
    rankings: list[dict] = []
    # X: [1.0, 0, 0, 0, 0, 0]; ranks 1 on win, 5 on losses.
    rankings.append(_ranking_row("X", 0, 1.0, 1))
    for i in range(1, 6):
        rankings.append(_ranking_row("X", i, 0.0, 5))
    # Y: [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]; ranks 2 everywhere.
    for i in range(6):
        rankings.append(_ranking_row("Y", i, 0.8, 2))
    candidates = [
        {"candidate_id": "X", "backend": "vllm", "serve_config": "x_cfg"},
        {"candidate_id": "Y", "backend": "vllm", "serve_config": "y_cfg"},
    ]
    return rankings, candidates


def test_slo_violator_ranks_below_robust_candidate():
    """X wins one workload (1.0) but violates SLO on 5 others (0.0).
    Y is consistently 0.8 across all 6 workloads. Under the zero-floored
    geomean, Y must rank above X globally."""
    rankings, candidates = _slo_violator_inputs()
    out = compute_robustness(rankings, candidates)
    order = [r["candidate_id"] for r in out]
    assert order.index("Y") < order.index("X"), (
        f"Expected Y above X under zero-floored geomean; got order {order} "
        f"with geomeans {[(r['candidate_id'], r['geomean_relative_to_best']) for r in out]}"
    )


def test_geomean_no_zeros_unchanged_by_floor():
    """When all values are positive and above the floor, the floor is a
    no-op: the geomean must equal the un-floored geometric mean."""
    values = [0.5, 0.7, 0.9]
    # Default floor (0.05) — should be a no-op since min(values) > floor.
    result_default = _geomean(values)
    # Explicit higher floor that's still below the min should also be a no-op.
    result_floor = _geomean(values, zero_floor=0.1)
    # Expected geometric mean: (0.5 * 0.7 * 0.9) ** (1/3)
    import math
    expected = math.exp((math.log(0.5) + math.log(0.7) + math.log(0.9)) / 3)
    assert result_default is not None
    assert abs(result_default - expected) < 1e-12
    assert result_floor is not None
    assert abs(result_floor - expected) < 1e-12


def test_load_zero_floor_from_config(tmp_path):
    """``report.robustness.zero_floor`` in config.yaml is honored;
    missing / malformed values fall back to the 0.05 default."""
    import yaml
    from moe_bench.html_report import _load_zero_floor

    # Default when config.yaml is absent.
    assert _load_zero_floor(tmp_path) == 0.05

    # Default when config.yaml has no report block.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"experiment": {"name": "x"}}))
    assert _load_zero_floor(tmp_path) == 0.05

    # Honored when configured.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"robustness": {"zero_floor": 0.2}}})
    )
    assert _load_zero_floor(tmp_path) == 0.2

    # Malformed value (non-numeric) falls back to default.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"robustness": {"zero_floor": "not-a-number"}}})
    )
    assert _load_zero_floor(tmp_path) == 0.05


def test_zero_floor_is_configurable():
    """A larger zero_floor produces a less-severe penalty for SLO violations.
    With floor=0.5, X's geomean is closer to Y's than with floor=0.05."""
    rankings, candidates = _slo_violator_inputs()
    out_strict = compute_robustness(rankings, candidates, zero_floor=0.05)
    out_lenient = compute_robustness(rankings, candidates, zero_floor=0.5)
    x_strict = next(r for r in out_strict if r["candidate_id"] == "X")
    x_lenient = next(r for r in out_lenient if r["candidate_id"] == "X")
    # A higher floor cannot make the geomean worse — it only replaces zeros
    # with a larger value, so the geometric mean monotonically increases.
    assert x_lenient["geomean_relative_to_best"] > x_strict["geomean_relative_to_best"]
