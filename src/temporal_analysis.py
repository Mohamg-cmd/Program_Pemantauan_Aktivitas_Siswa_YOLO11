"""
Stage 3 — Eksperimen FREQUENCY vs DURATION.

Kedua representasi WAJIB berasal dari video, deteksi, dan asosiasi identitas
yang SAMA; detector/tracker tidak dijalankan ulang. Yang dibandingkan murni
pengaruh pilihan representasi temporal terhadap ranking akhir.

Bila validasi identitas gagal, tidak ada ranking yang dihasilkan dan modul ini
mengembalikan status NOT_AVAILABLE — bukan angka palsu.
"""
from __future__ import annotations

import pandas as pd

from .rank_metrics import compare_rankings
from .saw import saw_score


def compare_representations(
    matrices: dict,
    weights: dict,
    orientation: dict,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    matrices : {"duration": DataFrame, "frequency": DataFrame}
        Keluaran aggregate.build_both_representations().

    Returns
    -------
    Satu baris perbandingan: Spearman, Kendall, mean/max rank displacement,
    jumlah rank berubah, top-5 dan top-10 overlap.
    """
    need = {"duration", "frequency"}
    if not need.issubset(matrices):
        raise ValueError(f"matrices harus memuat {need}, dapat {set(matrices)}")

    dur, freq = matrices["duration"], matrices["frequency"]
    if dur is None or freq is None or dur.empty or freq.empty:
        return pd.DataFrame([{
            "representation_a": "duration", "representation_b": "frequency",
            "status": "NOT_AVAILABLE",
            "reason": "matriks aktivitas kosong / identity validation gagal",
        }])

    rs = saw_score(dur, weights, orientation, zero_epsilon, zero_strategy)["rank"]
    rf = saw_score(freq, weights, orientation, zero_epsilon, zero_strategy)["rank"]

    rec = {
        "representation_a": "duration",
        "representation_b": "frequency",
        "status": "OK",
    }
    rec.update(compare_rankings(rs.reindex(dur.index), rf.reindex(freq.index)))
    return pd.DataFrame([rec])


def per_entity_rank_table(
    matrices: dict,
    weights: dict,
    orientation: dict,
    zero_epsilon: float = 1e-6,
    zero_strategy: str = "epsilon",
) -> pd.DataFrame:
    """Tabel per siswa: skor & rank pada kedua representasi + selisihnya."""
    dur, freq = matrices["duration"], matrices["frequency"]
    if dur.empty or freq.empty:
        return pd.DataFrame()
    a = saw_score(dur, weights, orientation, zero_epsilon, zero_strategy)
    b = saw_score(freq, weights, orientation, zero_epsilon, zero_strategy)
    out = pd.DataFrame({
        "student_id": dur.index,
        "V_duration": a["V"].reindex(dur.index).to_numpy(),
        "rank_duration": a["rank"].reindex(dur.index).to_numpy(),
        "V_frequency": b["V"].reindex(dur.index).to_numpy(),
        "rank_frequency": b["rank"].reindex(dur.index).to_numpy(),
    })
    out["rank_displacement"] = (out["rank_duration"] - out["rank_frequency"]).abs()
    return out.sort_values("rank_duration").reset_index(drop=True)
