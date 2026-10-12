"""
Stage 4 — Simple Additive Weighting (SAW).

Normalisasi:
  benefit : r_ij = x_ij / max_i(x_ij)
  cost    : r_ij = min_i(x_ij) / x_ij   (dengan zero-handling)

Skor akhir: V_i = sum_j w_j * r_ij, dengan sum_j w_j = 1.

Interpretasi V_i adalah *decision-support score* di bawah kriteria
observasional yang didefinisikan — BUKAN ukuran kecerdasan, karakter,
atau kondisi psikologis siswa.

--------------------------------------------------------------------
PERINGATAN METODOLOGIS — ZERO HANDLING PADA KRITERIA COST
--------------------------------------------------------------------
Aturan cost r_ij = min/x_ij mengasumsikan min > 0. Bila ADA SATU saja
entitas dengan nilai 0, maka min menjadi `zero_epsilon` (default 1e-6)
dan SELURUH entitas lain runtuh ke nilai ~1e-7, sehingga kriteria itu
kehilangan daya pembedanya.

Contoh terukur (kolom cost [10, 20, 40]):
    tanpa nol      -> [1.000, 0.500, 0.250]   (diskriminatif)
    +1 entitas nol -> [1e-7 , 5e-8 , 2.5e-8]  (praktis seragam / mati)

Pada data kelas nyata `Sleeping = 0` sangat lazim, sehingga situasi ini
kemungkinan besar terjadi pada setiap run. `zero_diagnostics()` melaporkan
berapa nol per kriteria agar efek ini WAJIB dilaporkan di manuskrip dan
tidak tersembunyi. Strategi alternatif tersedia lewat `zero_strategy`,
namun default dipertahankan agar hasil tetap kompatibel dengan baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_ZERO_STRATEGIES = {"epsilon", "shift_one", "max_ratio"}


def zero_diagnostics(
    matrix: pd.DataFrame,
    orientation: dict,
    zero_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Laporkan pemakaian zero-handling per kriteria.

    Kolom: criterion, orientation, zero_count, epsilon_applied,
           min_nonzero, discriminative_collapse
    `discriminative_collapse=True` menandai kriteria cost yang daya
    pembedanya hancur karena ada nilai nol — wajib disebut di manuskrip.
    """
    recs = []
    for c in matrix.columns:
        col = matrix[c].astype(float)
        zc = int((col == 0).sum())
        mode = orientation.get(c, "benefit")
        applied = bool(zc > 0 and mode == "cost")
        nz = col[col > 0]
        recs.append({
            "criterion": c,
            "orientation": mode,
            "zero_count": zc,
            "epsilon_applied": applied,
            "epsilon_value": zero_epsilon if applied else None,
            "min_nonzero": float(nz.min()) if len(nz) else None,
            "discriminative_collapse": applied and len(nz) > 0,
        })
    return pd.DataFrame(recs)


def saw_score(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """
    Hitung skor SAW dan ranking.

    Parameters
    ----------
    zero_epsilon : float
        Pengganti nilai 0 pada kriteria cost sebelum pembagian. WAJIB
        dilaporkan di manuskrip (lihat peringatan di docstring modul).
    zero_strategy : {"epsilon", "shift_one", "max_ratio"}
        "epsilon"   — perilaku baseline: 0 -> zero_epsilon (DEFAULT).
        "shift_one" — normalisasi memakai (1 + x): r = (1+min)/(1+x).
                      Menghindari keruntuhan skala, mengubah interpretasi.
        "max_ratio" — r = 1 - x/max(x). Linier terbalik, tetap di [0,1].
        Dua strategi terakhir DISEDIAKAN tetapi bukan default; bila dipakai
        harus dinyatakan eksplisit di manuskrip karena mengubah rumus.
    """
    if zero_strategy not in VALID_ZERO_STRATEGIES:
        raise ValueError(
            f"zero_strategy harus salah satu {sorted(VALID_ZERO_STRATEGIES)}, "
            f"bukan {zero_strategy!r}"
        )

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
    if (w < 0).any():
        raise ValueError("Bobot tidak boleh negatif")

    X = matrix.astype(float).copy()
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Matriks memuat NaN/inf — periksa agregasi temporal.")
    if (X.to_numpy() < 0).any():
        raise ValueError("Matriks memuat nilai negatif — tidak valid untuk SAW.")

    R = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)

    for c in criteria:
        col = X[c]
        mode = orientation[c]
        if mode == "benefit":
            mx = col.max()
            R[c] = col / mx if mx > 0 else 0.0
        elif mode == "cost":
            if zero_strategy == "epsilon":
                col_safe = col.replace(0, zero_epsilon)
                mn = col_safe.min()
                R[c] = mn / col_safe
            elif zero_strategy == "shift_one":
                shifted = 1.0 + col
                R[c] = shifted.min() / shifted
            else:  # max_ratio
                mx = col.max()
                R[c] = 1.0 - (col / mx) if mx > 0 else 1.0
        else:
            raise ValueError(
                f"orientation[{c!r}] harus 'benefit' atau 'cost', bukan {mode!r}"
            )

    V = (R * w).sum(axis=1)
    if not np.isfinite(V).all():
        raise ValueError("Skor SAW tidak finit — periksa zero handling.")

    result = R.copy()
    result["V"] = V
    result["rank"] = V.rank(method="min", ascending=False).astype(int)
    return result.sort_values("V", ascending=False)
