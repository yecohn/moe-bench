"""Tests for compute_failure_summary in moe_bench.failure_summary.

A failure row is a measurements row where valid != "true". Each row carries
at minimum backend, candidate_id, serve_config, workload (the workload name),
and failure_reason. compute_failure_summary takes the failure rows plus the
total candidates and workloads in the sweep, and returns None for a clean
run or a FailureSummary dataclass otherwise.
"""

from moe_bench.failure_summary import FailureSummary, compute_failure_summary


def _failure_row(*, candidate_id: str, serve_config: str, workload: str, failure_reason: str = "server_ready_timeout", backend: str = "vllm") -> dict:
    return {
        "backend": backend,
        "candidate_id": candidate_id,
        "serve_config": serve_config,
        "workload": workload,
        "failure_reason": failure_reason,
        "valid": "False",
    }


def test_clean_run_returns_none():
    assert compute_failure_summary([], candidates=[], workloads=[]) is None


def test_one_candidate_all_workloads_failed():
    # Sweep has 2 candidates x 3 workloads = 6 cells. One candidate failed
    # on every workload. The other ran clean (no rows in failures list).
    workloads = ["prompt256_out128_conc1", "prompt256_out128_conc4", "prompt4096_out128_conc1"]
    candidates = ["vllm_tp4", "sglang_tp4"]
    failures = [
        _failure_row(candidate_id="vllm_tp4", serve_config="vllm_tp4", workload=w)
        for w in workloads
    ]
    summary = compute_failure_summary(failures, candidates=candidates, workloads=workloads)
    assert summary is not None
    assert isinstance(summary, FailureSummary)
    assert summary.total_cells == 6
    assert summary.failed_cells == 3
    assert summary.rate_pct == 50.0
    assert summary.by_candidate == ["vllm_tp4"]
    assert set(summary.by_workload) == set(workloads)
    assert len(summary.by_workload) == 3
    assert summary.statuses == {"server_ready_timeout": 3}


def test_one_workload_failed_for_every_candidate():
    # Converse shape: one workload (the long-prompt one) failed across all
    # three candidates; the other two workloads succeeded for everyone.
    workloads = ["prompt256_out128_conc1", "prompt4096_out128_conc16", "prompt256_out128_conc4"]
    candidates = ["vllm_tp4", "vllm_tp2dp2", "sglang_tp4"]
    failures = [
        _failure_row(candidate_id=c, serve_config=c, workload="prompt4096_out128_conc16", failure_reason="oom")
        for c in candidates
    ]
    summary = compute_failure_summary(failures, candidates=candidates, workloads=workloads)
    assert summary is not None
    assert summary.total_cells == 9
    assert summary.failed_cells == 3
    # 3/9 = 33.33...%
    assert abs(summary.rate_pct - (100.0 / 3.0)) < 1e-9
    assert set(summary.by_candidate) == set(candidates)
    assert len(summary.by_candidate) == 3
    assert summary.by_workload == ["prompt4096_out128_conc16"]
    assert summary.statuses == {"oom": 3}


def test_mixed_failure_statuses_are_counted_per_reason():
    workloads = ["w1", "w2", "w3", "w4"]
    candidates = ["c1", "c2", "c3"]
    failures = [
        _failure_row(candidate_id="c1", serve_config="c1", workload="w1", failure_reason="server_ready_timeout"),
        _failure_row(candidate_id="c1", serve_config="c1", workload="w2", failure_reason="server_ready_timeout"),
        _failure_row(candidate_id="c1", serve_config="c1", workload="w3", failure_reason="server_ready_timeout"),
        _failure_row(candidate_id="c1", serve_config="c1", workload="w4", failure_reason="server_ready_timeout"),
        _failure_row(candidate_id="c2", serve_config="c2", workload="w3", failure_reason="oom"),
        _failure_row(candidate_id="c2", serve_config="c2", workload="w4", failure_reason="oom"),
    ]
    summary = compute_failure_summary(failures, candidates=candidates, workloads=workloads)
    assert summary is not None
    assert summary.statuses == {"server_ready_timeout": 4, "oom": 2}
    assert summary.failed_cells == 6


def test_partial_failures_list_only_affected_candidates():
    # Three candidates, two workloads, six cells total. c1 failed both
    # workloads, c2 failed one, c3 was completely clean (no rows in
    # failures input). by_candidate must contain c1 and c2 only — not c3.
    workloads = ["w_short", "w_long"]
    candidates = ["c1", "c2", "c3"]
    failures = [
        _failure_row(candidate_id="c1", serve_config="c1", workload="w_short", failure_reason="bench_failed"),
        _failure_row(candidate_id="c1", serve_config="c1", workload="w_long", failure_reason="bench_failed"),
        _failure_row(candidate_id="c2", serve_config="c2", workload="w_long", failure_reason="bench_timeout"),
    ]
    summary = compute_failure_summary(failures, candidates=candidates, workloads=workloads)
    assert summary is not None
    assert summary.total_cells == 6
    assert summary.failed_cells == 3
    assert summary.rate_pct == 50.0
    assert set(summary.by_candidate) == {"c1", "c2"}
    assert "c3" not in summary.by_candidate
    assert set(summary.by_workload) == {"w_short", "w_long"}
    assert summary.statuses == {"bench_failed": 2, "bench_timeout": 1}
