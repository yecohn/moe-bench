"""Tests for the top-K caps on the robustness leaderboard and the
candidate detail cards (issue 005).

These tests assert observable behavior of the renderer's HTML output
(string checks on the returned HTML), not internal structure. Mirrors
the style of ``tests/test_robustness_geomean.py``: pytest, no fixtures
beyond small local helpers.
"""
from __future__ import annotations

import re

from moe_bench.html_report import (
    _load_leaderboard_rows,
    _load_top_k_candidate_cards,
    _section_candidate_cards,
    _section_robustness_leaderboard,
)


def _rob_row(cid: str, geomean: float, idx: int) -> dict:
    """A single robustness-leaderboard row, as produced by ``compute_robustness``."""
    return {
        "candidate_id": cid,
        "backend": "vllm",
        "serve_config": f"cfg_{cid}",
        "geomean_relative_to_best": geomean,
        "worst_relative_to_best": geomean * 0.9,
        "median_rank": idx + 1,
        "workload_wins": 0,
        "workloads_evaluated": 6,
        "pareto_appearances": 0,
    }


def _make_robustness(n: int) -> list[dict]:
    """Build N synthetic robustness rows in descending geomean order."""
    return [_rob_row(f"cand{i:02d}", 1.0 - 0.05 * i, i) for i in range(n)]


def test_leaderboard_cap_emits_details_for_oversized_run():
    """Tracer: with 9 candidates and cap=5, the leaderboard HTML must include
    a <details> expander whose summary names the total count."""
    robustness = _make_robustness(9)
    html = _section_robustness_leaderboard(robustness, leaderboard_rows=5)
    assert "<details" in html
    assert "Show all 9" in html


def test_leaderboard_visible_table_truncated_to_top_k():
    """With 9 candidates and cap=5, the table visible before the <details>
    block must list exactly the first 5 candidate ids (by input order)."""
    robustness = _make_robustness(9)
    html = _section_robustness_leaderboard(robustness, leaderboard_rows=5)
    visible, marker, _ = html.partition("<details")
    assert marker, "expected a <details> block in an oversized run"
    visible_ids = set(re.findall(r"cand\d{2}", visible))
    assert visible_ids == {f"cand{i:02d}" for i in range(5)}


def test_leaderboard_no_details_when_total_under_cap():
    """4 candidates and cap=5: no <details> expander at all (no empty
    'show 0 more' element)."""
    robustness = _make_robustness(4)
    html = _section_robustness_leaderboard(robustness, leaderboard_rows=5)
    assert "<details" not in html
    assert "Show all" not in html


def test_load_leaderboard_rows_from_config(tmp_path):
    """``report.leaderboard_rows`` in config.yaml is honored; missing,
    malformed, or non-positive values fall back to the 5 default."""
    import yaml

    # Default when config.yaml is absent.
    assert _load_leaderboard_rows(tmp_path) == 5

    # Default when config.yaml has no report block.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"experiment": {"name": "x"}}))
    assert _load_leaderboard_rows(tmp_path) == 5

    # Honored when configured to a positive int.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"leaderboard_rows": 10}})
    )
    assert _load_leaderboard_rows(tmp_path) == 10

    # Zero falls back to default.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"leaderboard_rows": 0}})
    )
    assert _load_leaderboard_rows(tmp_path) == 5

    # Negative falls back to default.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"leaderboard_rows": -3}})
    )
    assert _load_leaderboard_rows(tmp_path) == 5

    # Non-int string falls back to default.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"leaderboard_rows": "five"}})
    )
    assert _load_leaderboard_rows(tmp_path) == 5


def _make_candidates_and_rob_map(n: int) -> tuple[list[dict], dict[str, dict]]:
    """Build N candidates and a matching robustness-by-cid map in descending
    geomean order, so the sort inside _section_candidate_cards preserves
    the natural ordering of candidate ids (cand00 first).
    """
    candidates = [
        {"candidate_id": f"cand{i:02d}", "backend": "vllm", "serve_config": f"cfg_{i:02d}"}
        for i in range(n)
    ]
    rob_by_cid = {
        f"cand{i:02d}": _rob_row(f"cand{i:02d}", 1.0 - 0.05 * i, i) for i in range(n)
    }
    return candidates, rob_by_cid


def test_candidate_cards_cap_emits_details_for_oversized_run():
    """With 10 candidates and cap=8, the cards section must contain a
    cap-introduced 'Show all' expander naming the total count, and the
    visible (non-expanded) area must contain the first 8 candidate ids."""
    candidates, rob_by_cid = _make_candidates_and_rob_map(10)
    html = _section_candidate_cards(
        measurements=[],
        rankings=[],
        candidates=candidates,
        robustness_by_cid=rob_by_cid,
        top_k=8,
    )
    assert "Show all 10" in html
    # Split on the cap-introduced summary; everything before it is the visible
    # cards region. Per-card <details> blocks (Per-workload ranks, Server
    # command) live before that summary, but they don't contain "Show all".
    visible, marker, _ = html.partition("<details><summary>Show all")
    assert marker, "expected a 'Show all' details block in oversized run"
    visible_ids = set(re.findall(r"cand\d{2}", visible))
    assert visible_ids == {f"cand{i:02d}" for i in range(8)}


def test_candidate_cards_no_show_all_when_total_under_cap():
    """6 candidates and cap=8: no cap-introduced 'Show all' expander.

    The section legitimately uses <details> internally for Per-workload
    ranks and Server command on every card, so we cannot assert that
    '<details' is entirely absent. The cap-specific summary text is the
    right thing to look for.
    """
    candidates, rob_by_cid = _make_candidates_and_rob_map(6)
    html = _section_candidate_cards(
        measurements=[],
        rankings=[],
        candidates=candidates,
        robustness_by_cid=rob_by_cid,
        top_k=8,
    )
    assert "Show all" not in html


def test_load_top_k_candidate_cards_from_config(tmp_path):
    """``report.top_k_candidate_cards`` is honored; missing, malformed,
    or non-positive values fall back to the 8 default."""
    import yaml

    assert _load_top_k_candidate_cards(tmp_path) == 8

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"experiment": {"name": "x"}}))
    assert _load_top_k_candidate_cards(tmp_path) == 8

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"top_k_candidate_cards": 12}})
    )
    assert _load_top_k_candidate_cards(tmp_path) == 12

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"top_k_candidate_cards": 0}})
    )
    assert _load_top_k_candidate_cards(tmp_path) == 8

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"report": {"top_k_candidate_cards": "eight"}})
    )
    assert _load_top_k_candidate_cards(tmp_path) == 8
