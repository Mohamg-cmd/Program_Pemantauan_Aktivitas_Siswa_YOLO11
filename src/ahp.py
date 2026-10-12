"""
Stage 3 — Analytic Hierarchy Process (AHP).

Menghitung vektor prioritas kriteria dari matriks pairwise
comparison dengan metode eigenvector, lengkap dengan konsistensi
(lambda_max, CI, CR). Random Index (RI) mengikuti tabel Saaty
standar (n ≤ 10).

Referensi:
  Saaty, T. L. (1977). A scaling method for priorities in
  hierarchical structures. J. Math. Psych., 15(3), 234–281.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Random Index (Saaty, 1977; nilai baris n)
RI_TABLE: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def _validate_matrix(A: np.ndarray) -> None:
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"Matriks harus persegi, dapat {A.shape}")
    if (A <= 0).any():
        raise ValueError("Seluruh entri matriks harus positif")
    # Cek resiprositas dengan toleransi kecil
    reciprocal_err = np.abs(A * A.T - 1.0)
    np.fill_diagonal(reciprocal_err, 0.0)
    if reciprocal_err.max() > 1e-3:
        max_pos = np.unravel_index(np.argmax(reciprocal_err), reciprocal_err.shape)
        raise ValueError(
            f"Matriks tidak resiprokal pada posisi {max_pos}: "
            f"a_ij * a_ji = {A[max_pos] * A[max_pos[::-1]]:.4f} (harus 1.0)"
        )


VALID_AHP_SOURCES = {"predefined_model", "researcher_constructed_supervisor_reviewed", "expert_elicitation", "literature"}


def aggregate_expert_matrices(matrices, method: str = "geometric") -> np.ndarray:
    """
    Gabungkan beberapa matriks pakar menjadi satu (AIJ).

    `geometric` = Aggregation of Individual Judgements dengan rerata
    geometrik — metode baku pada AHP karena mempertahankan resiprositas.
    Dipakai HANYA bila data pakar benar-benar tersedia; tidak ada data
    pakar yang dikarang di sini.
    """
    arrs = [np.asarray(m, dtype=float) for m in matrices]
    if not arrs:
        raise ValueError("Tidak ada matriks pakar untuk digabung")
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"Ukuran matriks pakar tidak seragam: {shapes}")
    if method != "geometric":
        raise ValueError("method yang didukung saat ini hanya 'geometric'")
    stack = np.stack(arrs, axis=0)
    if (stack <= 0).any():
        raise ValueError("Seluruh entri matriks pakar harus positif")
    return np.exp(np.log(stack).mean(axis=0))


def ahp_weights(
    matrix,
    criteria: Sequence[str],
    source: str = "predefined_model",
    expert_count: int = 0,
    aggregation_method: Optional[str] = None,
) -> dict:
    """
    Turunkan bobot kriteria + statistik konsistensi.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Matriks pairwise comparison resiprokal.
    criteria : sequence of str
        Nama kriteria; urutannya harus sesuai baris/kolom matriks.

    Returns
    -------
    dict dengan kunci:
        weights       : {kriteria: bobot}, jumlah = 1
        weight_vector : np.ndarray
        lambda_max    : eigenvalue terbesar
        CI            : consistency index
        RI            : random index
        CR            : consistency ratio (CI/RI)
        consistent    : bool, True jika CR ≤ 0.10
    """
    if source not in VALID_AHP_SOURCES:
        raise ValueError(
            f"ahp source harus salah satu {sorted(VALID_AHP_SOURCES)}, bukan {source!r}"
        )
    if source == "expert_elicitation" and expert_count <= 0:
        raise ValueError(
            "source='expert_elicitation' membutuhkan expert_count > 0. "
            "Jangan mengklaim elisitasi pakar tanpa data pakar."
        )
    A = np.asarray(matrix, dtype=float)
    _validate_matrix(A)
    n = A.shape[0]
    if n != len(criteria):
        raise ValueError(
            f"Jumlah kriteria ({len(criteria)}) tidak sama dengan "
            f"ukuran matriks ({n})"
        )

    # Metode eigenvector: prioritas = eigenvector terkait λ_max
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    # Matriks resiprokal positif menjamin akar Perron real (Perron-Frobenius);
    # bagian imajiner yang besar menandakan input rusak.
    if abs(float(eigvals[max_idx].imag)) > 1e-6:
        raise ValueError(
            f"Eigenvalue dominan tidak real (imag={eigvals[max_idx].imag:.2e}); "
            f"matriks pairwise kemungkinan rusak."
        )
    w = np.abs(eigvecs[:, max_idx].real)
    w = w / w.sum()

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    return {
        "weights": {c: float(v) for c, v in zip(criteria, w)},
        "weight_vector": w,
        "lambda_max": lambda_max,
        "CI": float(CI),
        "RI": float(RI),
        "CR": float(CR),
        "consistent": bool(CR <= 0.10),
        "ahp_source": source,
        "expert_count": int(expert_count),
        "aggregation_method": aggregation_method,
        "n_criteria": int(n),
        "matrix": A.tolist(),
    }
