"""
Stage 7 — Monte Carlo sensitivity (robustness ranking).

Alih-alih menggeser satu kriteria (OAT), seluruh vektor bobot diambil sampel
berulang kali untuk melihat seberapa sering peringkat seorang siswa bertahan.

Sampling bobot:
  * "dirichlet_around_base"  (DEFAULT) — Dirichlet dengan mean = bobot AHP
    dan konsentrasi `concentration`. Semakin besar konsentrasi, semakin dekat
    sampel ke bobot baseline. Menjawab: "seberapa sensitif ranking terhadap
    ketidakpastian di sekitar judgement AHP?"
  * "uniform_simplex" — Dirichlet(1,...,1), yaitu seragam pada simpleks;
    mengabaikan bobot AHP sama sekali. Berguna sebagai batas atas
    ketidakpastian.

Seluruh vektor yang dihasilkan dijamin w_i >= 0 dan sum(w_i) = 1.
Seed dicatat agar hasil dapat direproduksi persis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .saw import saw_score

VALID_METHODS = {"dirichlet_around_base", "uniform_simplex"}


def monte_carlo_sensitivity(
    matrix: pd.DataFrame,
    base_weights: dict,
    orientation: dict,
    iterations: int = 5000,
    seed: int = 42,
    method: str = "dirichlet_around_base",
    concentration: float = 50.0,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """
    Returns
    -------
    DataFrame per entitas: student_id, baseline_rank, mean_rank, std_rank,
    min_rank, max_rank, top1_frequency, top5_frequency, top10_frequency,
    n_iterations, seed, method.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method harus salah satu {sorted(VALID_METHODS)}")
    if iterations <= 0:
        raise ValueError("iterations harus > 0")
    if matrix is None or matrix.empty:
        return pd.DataFrame()

    criteria = list(matrix.columns)
    base_vec = np.array([base_weights[c] for c in criteria], dtype=float)
    if (base_vec < 0).any():
        raise ValueError("Bobot baseline tidak boleh negatif")

    rng = np.random.default_rng(seed)
    if method == "dirichlet_around_base":
        if concentration <= 0:
            raise ValueError("concentration harus > 0")
        alpha = np.clip(base_vec * concentration, 1e-6, None)
    else:
        alpha = np.ones(len(criteria))

    base = saw_score(matrix, base_weights, orientation, zero_epsilon, zero_strategy)
    baseline_rank = base["rank"].reindex(matrix.index)

    n_ent = len(matrix)
    ranks = np.empty((iterations, n_ent), dtype=float)

    for i in range(iterations):
        w = rng.dirichlet(alpha)
        w = w / w.sum()                     # jaminan numerik sum = 1
        wd = {c: float(v) for c, v in zip(criteria, w)}
        r = saw_score(matrix, wd, orientation, zero_epsilon, zero_strategy)
        ranks[i, :] = r["rank"].reindex(matrix.index).to_numpy()

    top5_k, top10_k = min(5, n_ent), min(10, n_ent)
    out = pd.DataFrame({
        "student_id": matrix.index,
        "baseline_rank": baseline_rank.to_numpy(),
        "mean_rank": ranks.mean(axis=0),
        "std_rank": ranks.std(axis=0, ddof=0),
        "min_rank": ranks.min(axis=0),
        "max_rank": ranks.max(axis=0),
        "top1_frequency": (ranks == 1).mean(axis=0),
        "top5_frequency": (ranks <= top5_k).mean(axis=0),
        "top10_frequency": (ranks <= top10_k).mean(axis=0),
    })
    out["n_iterations"] = iterations
    out["seed"] = seed
    out["method"] = method
    return out.sort_values("mean_rank").reset_index(drop=True)
