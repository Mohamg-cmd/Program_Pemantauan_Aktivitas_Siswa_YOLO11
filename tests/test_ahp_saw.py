"""
Smoke test — AHP + SAW + sensitivity dengan data sintetik.
Tidak butuh GPU/YOLO/ultralytics. Jalankan:
    python -m pytest tests/ -v
atau langsung:
    python tests/test_ahp_saw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Tambahkan root ke sys.path supaya `from src.*` bisa jalan
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ahp import ahp_weights
from src.saw import saw_score
from src.sensitivity import one_at_a_time


CRITERIA = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
ORIENTATION = {
    "Angkat Tangan": "benefit",
    "Berdiri": "benefit",
    "Diam": "cost",
    "Tidur": "cost",
}


def test_ahp_consistent_matrix():
    """Matriks contoh dari config.yaml harus konsisten (CR < 0.10)."""
    A = np.array([
        [1.000, 2.000, 3.000, 4.000],
        [0.500, 1.000, 2.000, 3.000],
        [0.333, 0.500, 1.000, 2.000],
        [0.250, 0.333, 0.500, 1.000],
    ])
    r = ahp_weights(A, CRITERIA)
    assert r["consistent"], f"CR = {r['CR']:.4f}"
    assert abs(sum(r["weights"].values()) - 1.0) < 1e-6
    # Angkat Tangan harus bobot terbesar (dominan di matriks)
    assert r["weights"]["Angkat Tangan"] == max(r["weights"].values())
    print(f"  CR = {r['CR']:.4f} ✓")
    print(f"  Weights = {r['weights']}")


def test_ahp_rejects_non_reciprocal():
    """Matriks non-resiprokal harus ditolak."""
    A = np.array([
        [1, 2, 3, 4],
        [0.5, 1, 2, 3],
        [0.333, 0.5, 1, 2],
        [0.25, 0.333, 5.0, 1],   # a_43 * a_34 ≠ 1
    ])
    try:
        ahp_weights(A, CRITERIA)
    except ValueError as e:
        print(f"  Ditolak: {e}")
        return
    raise AssertionError("Seharusnya raise ValueError")


def test_saw_benefit_and_cost():
    """SAW: benefit dinormalisasi ke max, cost ke min/x."""
    matrix = pd.DataFrame(
        {
            "Angkat Tangan": [4.0, 2.0, 8.0],
            "Berdiri":       [10.0, 20.0, 15.0],
            "Diam":          [100.0, 50.0, 80.0],
            "Tidur":         [10.0, 5.0, 20.0],
        },
        index=["S01", "S02", "S03"],
    )
    weights = {"Angkat Tangan": 0.4, "Berdiri": 0.3, "Diam": 0.2, "Tidur": 0.1}
    result = saw_score(matrix, weights, ORIENTATION)

    # Benefit: kolom AT max = 8, jadi S03=1.0, S01=0.5, S02=0.25
    assert abs(result.loc["S03", "Angkat Tangan"] - 1.0) < 1e-9
    assert abs(result.loc["S01", "Angkat Tangan"] - 0.5) < 1e-9
    # Cost: kolom Diam min = 50, jadi S02=1.0, S01=0.5, S03=0.625
    assert abs(result.loc["S02", "Diam"] - 1.0) < 1e-9
    assert abs(result.loc["S01", "Diam"] - 0.5) < 1e-9
    # V harus dalam [0, 1]
    assert (result["V"] >= 0).all() and (result["V"] <= 1).all()
    # Rank 1 harus V tertinggi
    top = result["rank"].idxmin()
    assert result.loc[top, "rank"] == 1
    assert result.loc[top, "V"] == result["V"].max()
    print("  SAW OK")
    print(result.round(4).to_string())


def test_saw_cost_zero_handling():
    """Nilai 0 pada kriteria cost tidak boleh menyebabkan div-by-zero."""
    matrix = pd.DataFrame(
        {"Angkat Tangan": [1.0, 2.0],
         "Berdiri":       [1.0, 2.0],
         "Diam":          [0.0, 10.0],   # zero!
         "Tidur":         [1.0, 2.0]},
        index=["S01", "S02"],
    )
    weights = {"Angkat Tangan": 0.25, "Berdiri": 0.25,
               "Diam": 0.25, "Tidur": 0.25}
    result = saw_score(matrix, weights, ORIENTATION, zero_epsilon=1e-6)
    assert np.isfinite(result["V"]).all()
    # S01 (Diam=0 → epsilon) mendapat r_ij = 1.0 (dia yang paling "kecil")
    assert abs(result.loc["S01", "Diam"] - 1.0) < 1e-9
    print("  Zero handling OK")


def test_saw_weights_must_sum_to_one():
    matrix = pd.DataFrame(
        {c: [1.0, 2.0] for c in CRITERIA}, index=["S01", "S02"]
    )
    bad_weights = {"Angkat Tangan": 0.5, "Berdiri": 0.3,
                   "Diam": 0.3, "Tidur": 0.3}   # sum = 1.4
    try:
        saw_score(matrix, bad_weights, ORIENTATION)
    except ValueError as e:
        print(f"  Ditolak: {e}")
        return
    raise AssertionError("Seharusnya raise ValueError")


def test_sensitivity_end_to_end():
    """Sensitivity dengan bobot AHP + matriks aktivitas kecil."""
    A = np.array([
        [1.000, 2.000, 3.000, 4.000],
        [0.500, 1.000, 2.000, 3.000],
        [0.333, 0.500, 1.000, 2.000],
        [0.250, 0.333, 0.500, 1.000],
    ])
    ahp = ahp_weights(A, CRITERIA)

    rng = np.random.default_rng(42)
    matrix = pd.DataFrame(
        rng.integers(1, 100, size=(10, 4)),
        columns=CRITERIA,
        index=[f"S{i:02d}" for i in range(1, 11)],
    ).astype(float)

    sens = one_at_a_time(matrix, ahp["weights"], ORIENTATION)
    assert not sens.empty
    assert (sens["spearman_rho"] <= 1.0).all()
    assert (sens["spearman_rho"] >= -1.0).all()
    print(f"  {len(sens)} perturbasi diuji")
    print(sens.round(4).to_string(index=False))


if __name__ == "__main__":
    tests = [
        ("AHP consistent matrix",           test_ahp_consistent_matrix),
        ("AHP reject non-reciprocal",       test_ahp_rejects_non_reciprocal),
        ("SAW benefit + cost",              test_saw_benefit_and_cost),
        ("SAW zero handling",               test_saw_cost_zero_handling),
        ("SAW weight sum validation",       test_saw_weights_must_sum_to_one),
        ("Sensitivity end-to-end",          test_sensitivity_end_to_end),
    ]
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        fn()
        print(f"[PASS] {name}")
    print("\n✔  Semua test lulus.")
