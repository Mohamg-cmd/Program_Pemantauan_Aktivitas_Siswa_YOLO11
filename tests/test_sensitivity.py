"""Test sensitivity OAT, Monte Carlo, WP, ablation, dan edge case SAW/AHP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import AHP_MATRIX, CRITERIA, ORIENTATION
from src.ablation import run_ablation
from src.ahp import aggregate_expert_matrices, ahp_weights
from src.monte_carlo import monte_carlo_sensitivity
from src.rank_metrics import compare_rankings
from src.saw import saw_score, zero_diagnostics
from src.sensitivity import DEFAULT_PERTURBATIONS, one_at_a_time
from src.wp import compare_saw_wp, wp_score


@pytest.fixture
def matrix():
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.integers(1, 100, size=(21, 4)), columns=CRITERIA,
                        index=[f"S{i:02d}" for i in range(1, 22)]).astype(float)


@pytest.fixture
def weights():
    return ahp_weights(np.array(AHP_MATRIX), CRITERIA)["weights"]


# ----------------------------------------------------------------- OAT
def test_oat_has_six_perturbations(matrix, weights):
    assert set(DEFAULT_PERTURBATIONS) == {-0.20, -0.10, -0.05, 0.05, 0.10, 0.20}
    s = one_at_a_time(matrix, weights, ORIENTATION)
    assert len(s) == 4 * 6


def test_oat_reports_all_required_metrics(matrix, weights):
    s = one_at_a_time(matrix, weights, ORIENTATION)
    for c in ["criterion_perturbed", "delta", "old_weight", "new_weight",
              "spearman_rho", "kendall_tau", "mean_rank_shift", "max_rank_shift",
              "n_rank_changes", "top1_changed", "top5_overlap", "top10_overlap"]:
        assert c in s.columns, f"kolom hilang: {c}"


def test_oat_correlations_in_range(matrix, weights):
    s = one_at_a_time(matrix, weights, ORIENTATION)
    assert s["spearman_rho"].between(-1, 1).all()
    assert s["kendall_tau"].between(-1, 1).all()
    assert (s["mean_rank_shift"] >= 0).all()
    assert (s["max_rank_shift"] >= s["mean_rank_shift"]).all()


def test_oat_weights_renormalised(matrix, weights):
    """Setelah perturbasi, seluruh bobot tetap berjumlah 1."""
    for crit in weights:
        for d in (-0.2, 0.2):
            nv = weights[crit] * (1 + d)
            others = {k: v for k, v in weights.items() if k != crit}
            scale = (1.0 - nv) / sum(others.values())
            total = nv + sum(v * scale for v in others.values())
            assert total == pytest.approx(1.0)


def test_oat_empty_matrix():
    assert one_at_a_time(pd.DataFrame(), {}, ORIENTATION).empty


# ----------------------------------------------------------------- Monte Carlo
def test_monte_carlo_reproducible(matrix, weights):
    a = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=200, seed=7)
    b = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_monte_carlo_different_seeds_differ(matrix, weights):
    a = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=200, seed=1)
    b = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=200, seed=2)
    assert not a["mean_rank"].equals(b["mean_rank"])


def test_monte_carlo_columns_and_bounds(matrix, weights):
    mc = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=200, seed=42)
    for c in ["student_id", "baseline_rank", "mean_rank", "std_rank", "min_rank",
              "max_rank", "top1_frequency", "top5_frequency", "top10_frequency"]:
        assert c in mc.columns
    assert len(mc) == len(matrix)
    assert mc["top1_frequency"].between(0, 1).all()
    assert (mc["min_rank"] <= mc["mean_rank"]).all()
    assert (mc["mean_rank"] <= mc["max_rank"]).all()
    assert mc["top1_frequency"].sum() == pytest.approx(1.0, abs=1e-6)


def test_monte_carlo_uniform_simplex(matrix, weights):
    mc = monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=100,
                                 seed=3, method="uniform_simplex")
    assert mc["method"].eq("uniform_simplex").all()


def test_monte_carlo_rejects_bad_args(matrix, weights):
    with pytest.raises(ValueError, match="method"):
        monte_carlo_sensitivity(matrix, weights, ORIENTATION, method="nope")
    with pytest.raises(ValueError, match="iterations"):
        monte_carlo_sensitivity(matrix, weights, ORIENTATION, iterations=0)


# ----------------------------------------------------------------- WP
def test_wp_scores_sum_to_one(matrix, weights):
    r = wp_score(matrix, weights, ORIENTATION)
    assert r["V"].sum() == pytest.approx(1.0)
    assert r["rank"].min() == 1


def test_wp_handles_zero(weights):
    m = pd.DataFrame({"Angkat Tangan": [1.0, 2.0], "Berdiri": [1.0, 2.0],
                      "Diam": [0.0, 10.0], "Tidur": [1.0, 2.0]}, index=["S01", "S02"])
    r = wp_score(m, weights, ORIENTATION)
    assert np.isfinite(r["V"]).all()


def test_saw_vs_wp_comparison(matrix, weights):
    c = compare_saw_wp(matrix, weights, ORIENTATION)
    assert c.iloc[0]["method_a"] == "AHP-SAW"
    assert -1 <= c.iloc[0]["spearman_rho"] <= 1
    assert c.iloc[0]["n_entities"] == 21


# ----------------------------------------------------------------- SAW edge
def test_zero_diagnostics_flags_collapse():
    m = pd.DataFrame({"Angkat Tangan": [5.0, 5, 5], "Berdiri": [5.0, 5, 5],
                      "Diam": [10.0, 20, 0], "Tidur": [1.0, 2, 3]},
                     index=["A", "B", "C"])
    d = zero_diagnostics(m, ORIENTATION)
    row = d[d["criterion"] == "Diam"].iloc[0]
    assert row["zero_count"] == 1 and row["epsilon_applied"]
    assert bool(row["discriminative_collapse"])
    assert not bool(d[d["criterion"] == "Tidur"].iloc[0]["epsilon_applied"])


def test_saw_cost_all_zero():
    m = pd.DataFrame({"Angkat Tangan": [1.0, 2], "Berdiri": [1.0, 2],
                      "Diam": [0.0, 0.0], "Tidur": [1.0, 2]}, index=["S01", "S02"])
    r = saw_score(m, {c: 0.25 for c in CRITERIA}, ORIENTATION)
    assert np.isfinite(r["V"]).all()
    assert (r["Diam"] == 1.0).all()


def test_saw_equal_values_tie():
    m = pd.DataFrame({c: [5.0, 5.0] for c in CRITERIA}, index=["S01", "S02"])
    r = saw_score(m, {c: 0.25 for c in CRITERIA}, ORIENTATION)
    assert r["rank"].nunique() == 1


def test_saw_single_entity():
    m = pd.DataFrame({c: [5.0] for c in CRITERIA}, index=["S01"])
    r = saw_score(m, {c: 0.25 for c in CRITERIA}, ORIENTATION)
    assert r.loc["S01", "rank"] == 1


def test_saw_rejects_negative_and_nan():
    w = {c: 0.25 for c in CRITERIA}
    m = pd.DataFrame({c: [1.0, -5.0] for c in CRITERIA}, index=["A", "B"])
    with pytest.raises(ValueError, match="negatif"):
        saw_score(m, w, ORIENTATION)
    m2 = pd.DataFrame({c: [1.0, np.nan] for c in CRITERIA}, index=["A", "B"])
    with pytest.raises(ValueError, match="NaN"):
        saw_score(m2, w, ORIENTATION)


def test_saw_alternative_zero_strategies():
    m = pd.DataFrame({"Angkat Tangan": [5.0, 5, 5], "Berdiri": [5.0, 5, 5],
                      "Diam": [10.0, 20, 0], "Tidur": [1.0, 2, 3]},
                     index=["A", "B", "C"])
    w = {c: 0.25 for c in CRITERIA}
    for strat in ("epsilon", "shift_one", "max_ratio"):
        r = saw_score(m, w, ORIENTATION, zero_strategy=strat)
        assert np.isfinite(r["V"]).all()
    with pytest.raises(ValueError, match="zero_strategy"):
        saw_score(m, w, ORIENTATION, zero_strategy="bogus")


def test_shift_one_avoids_collapse():
    """Regresi terdokumentasi: 'epsilon' meruntuhkan kriteria, 'shift_one' tidak."""
    m = pd.DataFrame({"Angkat Tangan": [5.0] * 4, "Berdiri": [5.0] * 4,
                      "Diam": [10.0, 20, 40, 0], "Tidur": [1.0, 1, 1, 1]},
                     index=list("ABCD"))
    w = {c: 0.25 for c in CRITERIA}
    eps = saw_score(m, w, ORIENTATION, zero_strategy="epsilon")
    shift = saw_score(m, w, ORIENTATION, zero_strategy="shift_one")
    assert eps.loc["A", "Diam"] < 1e-5          # runtuh
    assert shift.loc["A", "Diam"] > 0.05        # tetap bermakna


# ----------------------------------------------------------------- AHP edge
def test_ahp_inconsistent_flagged_not_silent():
    A = np.array([[1, 9, 9, 9], [1/9, 1, 9, 9], [1/9, 1/9, 1, 9], [1/9, 1/9, 1/9, 1]])
    r = ahp_weights(A, CRITERIA)
    assert r["CR"] > 0.10 and not r["consistent"]


def test_ahp_provenance_defaults():
    r = ahp_weights(np.array(AHP_MATRIX), CRITERIA)
    assert r["ahp_source"] == "predefined_model" and r["expert_count"] == 0


def test_ahp_expert_claim_requires_experts():
    with pytest.raises(ValueError, match="expert_count"):
        ahp_weights(np.array(AHP_MATRIX), CRITERIA, source="expert_elicitation")


def test_ahp_invalid_source():
    with pytest.raises(ValueError, match="ahp source"):
        ahp_weights(np.array(AHP_MATRIX), CRITERIA, source="made_up")


def test_ahp_expert_aggregation_geometric():
    A = np.array(AHP_MATRIX)
    g = aggregate_expert_matrices([A, A])
    assert np.allclose(g, A, atol=1e-6)
    assert np.allclose(np.diagonal(g), 1.0)


def test_ahp_criteria_count_mismatch():
    with pytest.raises(ValueError, match="Jumlah kriteria"):
        ahp_weights(np.array(AHP_MATRIX), ["a", "b"])


# ----------------------------------------------------------------- rank metrics
def test_compare_rankings_identical():
    r = pd.Series([1, 2, 3], index=list("abc"))
    out = compare_rankings(r, r)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["n_rank_changes"] == 0 and out["max_rank_shift"] == 0
    assert not out["top1_changed"]


def test_compare_rankings_reversed():
    a = pd.Series([1, 2, 3], index=list("abc"))
    b = pd.Series([3, 2, 1], index=list("abc"))
    out = compare_rankings(a, b)
    assert out["spearman_rho"] == pytest.approx(-1.0)
    assert out["top1_changed"]


def test_compare_rankings_topk_capped():
    a = pd.Series([1, 2], index=list("ab"))
    out = compare_rankings(a, a)
    assert out["top10_overlap"] == 1.0   # tidak melebihi jumlah entitas


def test_compare_rankings_empty():
    out = compare_rankings(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n_entities"] == 0


# ----------------------------------------------------------------- ablation
def test_ablation_variants_and_na():
    det = pd.DataFrame([{"frame": f, "track_id": t, "class_name": "Diam",
                         "confidence": 0.9}
                        for t in (1, 2) for f in range(60)])
    stu = pd.DataFrame([{"frame": f, "track_id": 1, "student_id": "S01",
                         "class_name": "Diam", "confidence": 0.9}
                        for f in range(60)])
    ab = run_ablation(det, 30.0, CRITERIA, 30, student_detections=stu,
                      identity_report={"passed": True})
    assert len(ab) == 4
    a_row = ab[ab["variant"] == "A_detection_only"].iloc[0]
    assert a_row["n_students"] == "NOT_APPLICABLE"
    assert a_row["raw_track_ids"] == "NOT_APPLICABLE"
    b_row = ab[ab["variant"] == "B_detection_tracking"].iloc[0]
    assert b_row["raw_track_ids"] == 2 and b_row["n_students"] == "NOT_APPLICABLE"
    d_row = ab[ab["variant"] == "D_full_student_level_pipeline"].iloc[0]
    assert d_row["n_students"] == 1 and d_row["identity_audit_status"] == "PASSED"


def test_ablation_without_identity_marks_not_run():
    det = pd.DataFrame([{"frame": f, "track_id": 1, "class_name": "Diam",
                         "confidence": 0.9} for f in range(60)])
    ab = run_ablation(det, 30.0, CRITERIA, 30)
    d = ab[ab["variant"] == "D_full_student_level_pipeline"].iloc[0]
    assert d["identity_audit_status"] == "NOT_RUN"
