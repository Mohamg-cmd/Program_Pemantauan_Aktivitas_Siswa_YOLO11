"""
Utilitas perbandingan ranking — dipakai bersama oleh sensitivity, Monte Carlo,
temporal comparison, dan perbandingan SAW vs WP agar definisi metrik
SERAGAM di seluruh eksperimen (tidak ada dua definisi rank shift).
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def compare_rankings(
    base_rank: pd.Series,
    new_rank: pd.Series,
    top_k: tuple = (1, 5, 10),
) -> dict:
    """
    Bandingkan dua vektor ranking pada entitas yang SAMA.

    delta_rank_i   = |rank_baseline_i - rank_scenario_i|
    mean_rank_shift = mean(delta_rank_i)
    max_rank_shift  = max(delta_rank_i)

    Top-k overlap = |top-k(base) ∩ top-k(new)| / k, dipotong pada jumlah
    entitas yang tersedia (k > n tidak menghasilkan angka menyesatkan).
    """
    idx = base_rank.index.intersection(new_rank.index)
    b = base_rank.reindex(idx).astype(float)
    n_ = new_rank.reindex(idx).astype(float)
    n_ent = len(idx)

    out: dict = {"n_entities": n_ent}
    if n_ent == 0:
        return out

    delta = (b - n_).abs()
    out["mean_rank_shift"] = float(delta.mean())
    out["max_rank_shift"] = float(delta.max())
    out["n_rank_changes"] = int((b != n_).sum())

    if n_ent >= 2 and b.nunique() > 1 and n_.nunique() > 1:
        rho, _ = spearmanr(b, n_)
        tau, _ = kendalltau(b, n_)
        out["spearman_rho"] = float(rho)
        out["kendall_tau"] = float(tau)
    else:
        # Korelasi tidak terdefinisi untuk ranking konstan / entitas tunggal.
        out["spearman_rho"] = float("nan")
        out["kendall_tau"] = float("nan")

    b_sorted = b.sort_values().index.tolist()
    n_sorted = n_.sort_values().index.tolist()
    out["top1_changed"] = bool(b_sorted[0] != n_sorted[0])
    for k in top_k:
        kk = min(k, n_ent)
        overlap = len(set(b_sorted[:kk]) & set(n_sorted[:kk])) / kk
        out[f"top{k}_overlap"] = float(overlap)
    return out
