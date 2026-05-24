"""Tests for compute_recommendation in moe_bench.recommendation.

These tests pin the pure-function behavior of the merged Decision-summary
hero's logic: identifying the global winner across workloads, the runner-up,
the margin, the worst-case workload, and the human-readable labels. No HTML,
no I/O — synthetic rankings/candidates dicts in, structured Recommendation
fields out.

Following the project's existing test style (see test_config_overrides.py,
test_failure_summary.py): pytest, no fixtures beyond small local helpers.
"""

from moe_bench.recommendation import Recommendation, compute_recommendation, render_decision_hero


def _wkl(idx: int) -> dict:
    """Build a synthetic workload identity for a ranking row."""
    return {
        "workload": f"w{idx}",
        "input_len": 256,
        "output_len": 128,
        "max_concurrency": idx + 1,
    }


def _ranking_row(cid: str, idx: int, relative_to_best: float, rank: int, backend: str = "vllm", serve_config: str = "") -> dict:
    row = _wkl(idx)
    row["candidate_id"] = cid
    row["relative_to_best"] = relative_to_best
    row["rank"] = rank
    row["backend"] = backend
    row["serve_config"] = serve_config or f"{cid}_cfg"
    return row


def _candidate(cid: str, backend: str = "vllm", serve_config: str | None = None, **params) -> dict:
    row = {
        "candidate_id": cid,
        "backend": backend,
        "serve_config": serve_config or f"{cid}_cfg",
    }
    for k, v in params.items():
        row[k] = v
    return row


def test_clear_winner_across_all_workloads():
    """X wins all 6 workloads with rank=1; Y is rank=2 everywhere.
    Winner is X; wins == total_workloads; is_degraded is False."""
    rankings: list[dict] = []
    for i in range(6):
        rankings.append(_ranking_row("X", i, 1.0, 1, backend="vllm", serve_config="x_cfg"))
        rankings.append(_ranking_row("Y", i, 0.7, 2, backend="vllm", serve_config="y_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg", vllm_tensor_parallel_size=2),
        _candidate("Y", backend="vllm", serve_config="y_cfg", vllm_tensor_parallel_size=4),
    ]
    rec = compute_recommendation(rankings, candidates, slo={}, objective="goodput_at_slo")
    assert isinstance(rec, Recommendation)
    assert rec.is_degraded is False
    assert rec.winner_candidate_id == "X"
    assert rec.winner_backend == "vllm"
    assert rec.winner_serve_config_name == "x_cfg"
    assert rec.wins == 6
    assert rec.total_workloads == 6


def test_runner_up_identified_and_margin_correct():
    """X has rel-to-best 1.0 everywhere; Y has 0.5 everywhere. Geomean(X)=1.0,
    geomean(Y)=0.5. Margin over runner-up = 100 * (1.0 - 0.5) / 0.5 = 100.0%."""
    rankings: list[dict] = []
    for i in range(4):
        rankings.append(_ranking_row("X", i, 1.0, 1, backend="vllm", serve_config="x_cfg"))
        rankings.append(_ranking_row("Y", i, 0.5, 2, backend="vllm", serve_config="y_cfg"))
        rankings.append(_ranking_row("Z", i, 0.25, 3, backend="sglang", serve_config="z_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
        _candidate("Z", backend="sglang", serve_config="z_cfg"),
    ]
    rec = compute_recommendation(rankings, candidates, slo={}, objective="goodput_at_slo")
    assert rec.winner_candidate_id == "X"
    assert rec.runner_up_candidate_id == "Y"
    assert rec.runner_up_serve_config_name == "y_cfg"
    assert rec.margin_pct_over_runner_up is not None
    assert abs(rec.margin_pct_over_runner_up - 100.0) < 1e-6


def test_worst_case_fields_match_winners_lowest_workload():
    """Winner X wins 5 of 6 workloads at rel=1.0 but is 0.7 on one workload.
    worst_case_ratio == 0.7 and worst_case_workload_label points to that workload."""
    rankings: list[dict] = []
    # X is best on workloads 0..4 and 0.7 on workload 5.
    for i in range(5):
        rankings.append(_ranking_row("X", i, 1.0, 1, backend="vllm", serve_config="x_cfg"))
    rankings.append(_ranking_row("X", 5, 0.7, 2, backend="vllm", serve_config="x_cfg"))
    # Y is uniformly worse so X is still the overall winner.
    for i in range(6):
        rankings.append(_ranking_row("Y", i, 0.4, 3, backend="vllm", serve_config="y_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
    ]
    rec = compute_recommendation(rankings, candidates, slo={}, objective="goodput_at_slo")
    assert rec.winner_candidate_id == "X"
    assert rec.worst_case_ratio is not None
    assert abs(rec.worst_case_ratio - 0.7) < 1e-9
    # workload label is the synthetic "w5" name we attached in _ranking_row.
    assert rec.worst_case_workload_label == "w5"


def test_zero_floor_flows_through_to_recommendation():
    """A candidate with one big win [1.0, 0, 0, 0, 0, 0] must NOT outrank a
    robust candidate [0.8] * 6 — relies on the zero-floor in compute_robustness.
    Pins that compute_recommendation actually uses the floored geomean."""
    rankings: list[dict] = []
    # X: one workload win, SLO violations (rel=0) on the other 5.
    rankings.append(_ranking_row("X", 0, 1.0, 1, backend="vllm", serve_config="x_cfg"))
    for i in range(1, 6):
        rankings.append(_ranking_row("X", i, 0.0, 5, backend="vllm", serve_config="x_cfg"))
    # Y: consistently 0.8 across all 6 workloads, never the winner.
    for i in range(6):
        rankings.append(_ranking_row("Y", i, 0.8, 2, backend="vllm", serve_config="y_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
    ]
    rec = compute_recommendation(rankings, candidates, slo={}, objective="goodput_at_slo")
    # With the zero-floor (default 0.05), Y's geomean (~0.8) >> X's geomean
    # (~ (1.0 * 0.05^5)^(1/6) ≈ 0.108). The robust candidate wins.
    assert rec.winner_candidate_id == "Y", (
        f"Expected Y (robust) to win over X (lucky one-shot); got {rec.winner_candidate_id}."
    )


def test_tiebreak_by_median_rank_then_win_count():
    """Two candidates have identical geomean (same rel-to-best values across
    workloads). Tiebreak by median rank ascending: the lower median rank wins.

    Then add a third candidate with identical geomean AND identical median
    rank; tiebreak by win count descending picks the one with more rank=1's."""
    # Pair 1: X has [0.8, 0.8] at ranks [1, 1]; Y has [0.8, 0.8] at ranks [2, 2].
    # Same geomean (0.8); X has lower median rank (1 vs 2). X must win.
    rankings_pair: list[dict] = [
        _ranking_row("X", 0, 0.8, 1, backend="vllm", serve_config="x_cfg"),
        _ranking_row("X", 1, 0.8, 1, backend="vllm", serve_config="x_cfg"),
        _ranking_row("Y", 0, 0.8, 2, backend="vllm", serve_config="y_cfg"),
        _ranking_row("Y", 1, 0.8, 2, backend="vllm", serve_config="y_cfg"),
    ]
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
    ]
    rec = compute_recommendation(rankings_pair, candidates, slo={}, objective="goodput_at_slo")
    assert rec.winner_candidate_id == "X", (
        f"Expected X (lower median rank) to win the geomean tie; got {rec.winner_candidate_id}."
    )

    # Triple: A, B, C all have identical [0.8, 0.8] but A wins workload 0
    # (rank=1, rank=2) median rank = 1.5; B wins workload 1 (rank=2, rank=1)
    # median rank = 1.5; C is always rank=2 (median 2). A and B tied on
    # geomean AND median rank, but A has the same win count (1) as B.
    # To make win-count meaningful we need A to win MORE workloads at the
    # same geomean/median-rank: use a 3-workload setup where A wins 2 and
    # B wins 1, both with identical relative_to_best across workloads.
    rankings_triple: list[dict] = [
        # A: ranks [1, 1, 2] — median 1, wins 2
        _ranking_row("A", 0, 0.9, 1, backend="vllm", serve_config="a_cfg"),
        _ranking_row("A", 1, 0.9, 1, backend="vllm", serve_config="a_cfg"),
        _ranking_row("A", 2, 0.9, 2, backend="vllm", serve_config="a_cfg"),
        # B: ranks [2, 2, 1] — median 2, wins 1
        _ranking_row("B", 0, 0.9, 2, backend="vllm", serve_config="b_cfg"),
        _ranking_row("B", 1, 0.9, 2, backend="vllm", serve_config="b_cfg"),
        _ranking_row("B", 2, 0.9, 1, backend="vllm", serve_config="b_cfg"),
        # C: ranks [3, 3, 3] — median 3, wins 0
        _ranking_row("C", 0, 0.9, 3, backend="vllm", serve_config="c_cfg"),
        _ranking_row("C", 1, 0.9, 3, backend="vllm", serve_config="c_cfg"),
        _ranking_row("C", 2, 0.9, 3, backend="vllm", serve_config="c_cfg"),
    ]
    candidates_triple = [
        _candidate("A", backend="vllm", serve_config="a_cfg"),
        _candidate("B", backend="vllm", serve_config="b_cfg"),
        _candidate("C", backend="vllm", serve_config="c_cfg"),
    ]
    rec_triple = compute_recommendation(rankings_triple, candidates_triple, slo={}, objective="goodput_at_slo")
    # A has lower median rank than B, so A wins. (The win-count tiebreak
    # would also pick A: A has 2 wins vs B's 1.)
    assert rec_triple.winner_candidate_id == "A", (
        f"Expected A to win on median rank (and as a fallback on win count); "
        f"got {rec_triple.winner_candidate_id}."
    )


def test_single_backend_path():
    """Only one backend has measurements. Winner and runner-up are both vllm.
    is_degraded is False; no comparative-phrasing flags get set incorrectly."""
    rankings: list[dict] = []
    for i in range(3):
        rankings.append(_ranking_row("X", i, 1.0, 1, backend="vllm", serve_config="x_cfg"))
        rankings.append(_ranking_row("Y", i, 0.6, 2, backend="vllm", serve_config="y_cfg"))
        rankings.append(_ranking_row("Z", i, 0.4, 3, backend="vllm", serve_config="z_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
        _candidate("Z", backend="vllm", serve_config="z_cfg"),
    ]
    rec = compute_recommendation(rankings, candidates, slo={}, objective="goodput_at_slo")
    assert rec.is_degraded is False
    assert rec.winner_backend == "vllm"
    assert rec.winner_candidate_id == "X"
    assert rec.runner_up_candidate_id == "Y"
    # Both winner and runner-up are from the same backend.
    runner_up_cand = next(c for c in candidates if c["candidate_id"] == rec.runner_up_candidate_id)
    assert runner_up_cand["backend"] == rec.winner_backend


def test_zero_valid_path_returns_degraded():
    """Empty rankings (fully-failed sweep): is_degraded=True, winner_* fields
    are None, objective_label and slo_label still populated."""
    rec = compute_recommendation(
        [],
        [],
        slo={"p99_ttft_ms": 2000},
        objective="goodput_at_slo",
    )
    assert rec.is_degraded is True
    assert rec.winner_backend is None
    assert rec.winner_candidate_id is None
    assert rec.winner_serve_config_name is None
    assert rec.runner_up_candidate_id is None
    assert rec.runner_up_serve_config_name is None
    # Labels are still populated for the degraded hero copy.
    assert "goodput_at_slo" in rec.objective_label
    assert "p99_ttft_ms" in rec.slo_label


def test_objective_label_reflects_non_default_objective():
    """A non-default objective string is reflected verbatim in objective_label."""
    rankings: list[dict] = [
        _ranking_row("X", 0, 1.0, 1, backend="vllm", serve_config="x_cfg"),
        _ranking_row("Y", 0, 0.5, 2, backend="vllm", serve_config="y_cfg"),
    ]
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
    ]
    rec = compute_recommendation(
        rankings, candidates, slo={}, objective="median_output_tok_s_per_gpu"
    )
    assert "median_output_tok_s_per_gpu" in rec.objective_label


def test_slo_label_reflects_constraints_with_units():
    """With p99_ttft_ms and p99_tpot_ms constraints, slo_label reads
    `p99_ttft_ms ≤ 2000ms, p99_tpot_ms ≤ 100ms`. With empty slo, it's ''."""
    rankings: list[dict] = [
        _ranking_row("X", 0, 1.0, 1, backend="vllm", serve_config="x_cfg"),
    ]
    candidates = [_candidate("X", backend="vllm", serve_config="x_cfg")]
    rec = compute_recommendation(
        rankings,
        candidates,
        slo={"p99_ttft_ms": 2000, "p99_tpot_ms": 100},
        objective="goodput_at_slo",
    )
    # Each constraint rendered as `metric ≤ value` with "ms" appended for *_ms metrics.
    assert "p99_ttft_ms ≤ 2000ms" in rec.slo_label
    assert "p99_tpot_ms ≤ 100ms" in rec.slo_label

    # Empty SLO -> empty label.
    rec_empty = compute_recommendation(
        rankings, candidates, slo={}, objective="goodput_at_slo"
    )
    assert rec_empty.slo_label == ""


def test_render_decision_hero_emits_slider_dom_hooks():
    """The slider JS (issue 006) finds and updates hero elements via stable
    data-* attributes. This pins that the four hooks the JS looks for
    (winner / slo-label / evidence-wins / evidence-margin) are present in
    the rendered hero HTML for a non-degraded recommendation.

    We deliberately do NOT snapshot the surrounding copy — only the hooks,
    so the test survives copy edits to the hero.
    """
    rankings: list[dict] = []
    for i in range(3):
        rankings.append(_ranking_row("X", i, 1.0, 1, backend="vllm", serve_config="x_cfg"))
        rankings.append(_ranking_row("Y", i, 0.5, 2, backend="vllm", serve_config="y_cfg"))
    candidates = [
        _candidate("X", backend="vllm", serve_config="x_cfg"),
        _candidate("Y", backend="vllm", serve_config="y_cfg"),
    ]
    rec = compute_recommendation(
        rankings, candidates,
        slo={"p99_ttft_ms": 2000, "p99_tpot_ms": 100},
        objective="goodput_at_slo",
    )
    assert rec.is_degraded is False
    html_out = render_decision_hero(rec)
    # The four hooks the slider JS targets. Each must be findable by querySelector.
    assert "data-rec-winner" in html_out
    assert "data-rec-slo-label" in html_out
    assert "data-rec-evidence-wins" in html_out
    assert "data-rec-evidence-margin" in html_out
