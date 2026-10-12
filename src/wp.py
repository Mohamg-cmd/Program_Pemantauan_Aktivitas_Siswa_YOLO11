"""
Stage 9 (opsional) — Weighted Product (WP) sebagai pembanding agregasi.

WP memakai perkalian berpangkat, bukan penjumlahan:

    S_i = prod_j ( x_ij ^ (w_j * s_j) ),   s_j = +1 (benefit), -1 (cost)
    V_i = S_i / sum_i S_i

Tujuan modul ini adalah MENGUKUR SENSITIVITAS TERHADAP METODE AGREGASI
pada data YOLO11 yang SAMA. Modul ini TIDAK dimaksudkan untuk menyimpulkan
"SAW lebih baik" atau "WP lebih baik" — klaim semacam itu memerlukan
kriteria evaluasi eksternal yang saat ini tidak tersedia.

CATATAN NOL: WP tidak terdefinisi untuk x = 0 pada eksponen negatif dan
menghasilkan S_i = 0 untuk eksponen positif. Nilai nol karenanya diganti
`zero_epsilon`, dan jumlah penggantian dilaporkan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rank_metrics import compare_rankings


def wp_score(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    zero_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Hitung skor WP + ranking. Kolom keluaran: S, V, rank."""
    criteria = list(matrix.columns)
    missing_w = set(criteria) - set(weights)
    missing_o = set(criteria) - set(orientation)
    if missing_w:
        raise ValueError(f"weights tidak lengkap; hilang: {missing_w}")
    if missing_o:
        raise ValueError(f"orientation tidak lengkap; hilang: {missing_o}")

    w = np.array([weights[c] for c in criteria], dtype=float)
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Jumlah bobot harus 1.0, dapat {w.sum():.6f}")

    X = matrix.astype(float).copy()
    if (X.to_numpy() < 0).any():
        raise ValueError("WP tidak terdefinisi untuk nilai negatif")
    X = X.replace(0, zero_epsilon)

    signs = np.array(
        [1.0 if orientation[c] == "benefit" else -1.0 for c in criteria], dtype=float
    )
    for c in criteria:
        if orientation[c] not in {"benefit", "cost"}:
            raise ValueError(f"orientation[{c!r}] tidak valid: {orientation[c]!r}")

    exponents = w * signs
    S = np.exp((np.log(X.to_numpy()) * exponents).sum(axis=1))  # stabil numerik
    total = S.sum()
    V = S / total if total > 0 else np.zeros_like(S)

    out = pd.DataFrame({"S": S, "V": V}, index=X.index)
    out["rank"] = out["V"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("V", ascending=False)


def compare_saw_wp(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """
    Bandingkan ranking AHP-SAW vs AHP-WP pada matriks yang sama.
    Satu baris ringkasan metrik kesepakatan ranking.
    """
    from .saw import saw_score

    if matrix is None or matrix.empty:
        return pd.DataFrame()
    s = saw_score(matrix, weights, orientation, zero_epsilon, zero_strategy)
    p = wp_score(matrix, weights, orientation, zero_epsilon)
    rec = {"method_a": "AHP-SAW", "method_b": "AHP-WP"}
    rec.update(compare_rankings(
        s["rank"].reindex(matrix.index), p["rank"].reindex(matrix.index)
    ))
    return pd.DataFrame([rec])
