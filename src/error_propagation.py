"""
Stage 13 — Analisis propagasi error (DESKRIPTIF).

Rantai: Detection -> Tracking -> Identity -> Temporal profile -> SAW -> Rank.

Karena ground truth deteksi/trajektori tidak tersedia, modul ini TIDAK
menghitung error rate. Yang dihasilkan adalah rantai KUANTITAS TERUKUR
(hasil eksekusi nyata) yang dipisahkan secara tegas dari EFEK TERSIMPULKAN
(interpretasi yang memerlukan asumsi).

Kolom `evidence_type` menandai setiap baris:
  measured  — dihitung langsung dari keluaran pipeline
  inferred  — interpretasi/estimasi, WAJIB dibaca sebagai dugaan
"""
from __future__ import annotations

import pandas as pd


def build_propagation_table(
    student_audit: pd.DataFrame,
    activity_matrix: pd.DataFrame,
    saw_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rantai per siswa: fragmen track -> frame teramati -> durasi -> skor -> rank.
    Seluruh baris di sini berjenis `measured`.
    """
    if student_audit is None or student_audit.empty or activity_matrix.empty:
        return pd.DataFrame()

    rows = []
    for sid in activity_matrix.index:
        aud = student_audit[student_audit["student_id"] == sid]
        rec = {
            "student_id": sid,
            "evidence_type": "measured",
            "track_fragments": int(aud["n_track_fragments"].iloc[0]) if not aud.empty else 0,
            "observed_frames": int(aud["total_frames"].iloc[0]) if not aud.empty else 0,
            "duration_sec": float(aud["total_duration_sec"].iloc[0]) if not aud.empty else 0.0,
            "mean_identity_confidence": (
                float(aud["mean_identity_confidence"].iloc[0]) if not aud.empty else 0.0
            ),
            "unassigned_frames": int(aud["unassigned_frames"].iloc[0]) if not aud.empty else 0,
            "duplicate_track_frames": (
                int(aud["duplicate_track_frames"].iloc[0]) if not aud.empty else 0
            ),
        }
        for c in activity_matrix.columns:
            rec[f"x[{c}]"] = float(activity_matrix.loc[sid, c])
        if sid in saw_result.index:
            rec["V"] = float(saw_result.loc[sid, "V"])
            rec["rank"] = int(saw_result.loc[sid, "rank"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("rank" if "rank" in rows[0] else "student_id")


def scenario_delta(
    baseline_saw: pd.DataFrame,
    scenario_saw: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Perubahan skor/rank antar skenario (measured)."""
    idx = baseline_saw.index.intersection(scenario_saw.index)
    if len(idx) == 0:
        return pd.DataFrame()
    return pd.DataFrame({
        "student_id": idx,
        "scenario": scenario_name,
        "evidence_type": "measured",
        "V_baseline": baseline_saw.loc[idx, "V"].to_numpy(),
        "V_scenario": scenario_saw.loc[idx, "V"].to_numpy(),
        "rank_baseline": baseline_saw.loc[idx, "rank"].to_numpy(),
        "rank_scenario": scenario_saw.loc[idx, "rank"].to_numpy(),
        "delta_V": (scenario_saw.loc[idx, "V"] - baseline_saw.loc[idx, "V"]).to_numpy(),
        "delta_rank": (
            scenario_saw.loc[idx, "rank"] - baseline_saw.loc[idx, "rank"]
        ).abs().to_numpy(),
    }).sort_values("delta_rank", ascending=False).reset_index(drop=True)


def qualitative_notes() -> pd.DataFrame:
    """
    Catatan efek TERSIMPULKAN — eksplisit ditandai `inferred`, tanpa angka.
    Ini mencegah pembaca menyalahartikan interpretasi sebagai hasil ukur.
    """
    notes = [
        ("Detection -> Tracking",
         "Deteksi yang terlewat memperpanjang celah trajektori dan memicu "
         "fragmentasi ID BoT-SORT."),
        ("Tracking -> Identity",
         "Fragmentasi menaikkan jumlah fragmen per siswa; lapisan fixed-seat "
         "menggabungkannya kembali sehingga jumlah siswa tetap terjaga."),
        ("Identity -> Temporal",
         "Frame unassigned mengurangi durasi/frekuensi teramati; ketiadaan data "
         "TIDAK sama dengan aktivitas bernilai nol."),
        ("Temporal -> SAW",
         "Durasi yang kurang tercatat pada kriteria cost dapat menaikkan skor "
         "secara keliru karena normalisasi min/x."),
        ("SAW -> Rank",
         "Selisih skor yang kecil dapat mengubah rank; besarnya perubahan "
         "terlihat pada analisis sensitivitas dan Monte Carlo."),
    ]
    return pd.DataFrame(
        [{"stage": a, "evidence_type": "inferred", "note": b} for a, b in notes]
    )
