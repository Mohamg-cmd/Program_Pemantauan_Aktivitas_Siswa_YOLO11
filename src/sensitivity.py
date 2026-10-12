"""
Stage 5 — Sensitivity analysis untuk rank stability.

Metode: One-At-a-Time (OAT). Setiap kriteria digeser bobotnya dengan faktor
delta, sedangkan bobot lain diskala ulang proporsional agar total tetap = 1.

Metrik yang dihitung diperluas dari versi baseline agar memenuhi kebutuhan
pelaporan: old_weight/new_weight, Spearman, Kendall, mean/max rank shift,
jumlah perubahan rank, perubahan peringkat-1, serta overlap top-5/top-10.

CATATAN: modul ini TIDAK menyimpulkan "stabil"/"tidak stabil". Penilaian
tersebut hanya boleh dibuat setelah melihat angka hasil eksekusi nyata.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .rank_metrics import compare_rankings
from .saw import saw_score

DEFAULT_PERTURBATIONS: tuple = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)


def one_at_a_time(
    matrix: pd.DataFrame,
    base_weights: dict,
    orientation: dict,
    perturbations: Sequence[float] = DEFAULT_PERTURBATIONS,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """
    Jalankan sensitivity OAT dan kembalikan ringkasannya.

    Returns
    -------
    DataFrame kolom: criterion_perturbed, delta, old_weight, new_weight,
    spearman_rho, kendall_tau, mean_rank_shift, max_rank_shift,
    n_rank_changes, top1_changed, top5_overlap, top10_overlap, n_entities.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame()

    base = saw_score(matrix, base_weights, orientation, zero_epsilon, zero_strategy)
    base_rank = base["rank"].reindex(matrix.index)

    records: list = []
    for crit in base_weights:
        for delta in perturbations:
            new_val = base_weights[crit] * (1 + delta)
            others = {k: v for k, v in base_weights.items() if k != crit}
            others_sum = sum(others.values())
            remaining = 1.0 - new_val

            # Lewati konfigurasi tidak valid (bobot <= 0 atau >= 1)
            if remaining <= 0 or others_sum <= 0 or new_val <= 0:
                continue

            scale = remaining / others_sum
            new_weights = {k: v * scale for k, v in others.items()}
            new_weights[crit] = new_val

            new = saw_score(matrix, new_weights, orientation, zero_epsilon, zero_strategy)
            new_rank = new["rank"].reindex(matrix.index)

            rec = {
                "criterion_perturbed": crit,
                "delta": delta,
                "old_weight": float(base_weights[crit]),
                "new_weight": float(new_val),
            }
            rec.update(compare_rankings(base_rank, new_rank))
            records.append(rec)

    return pd.DataFrame(records)
