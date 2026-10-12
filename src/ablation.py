"""
Stage 10 — Studi ablasi.

Varian:
  A. detection_only              — hanya statistik deteksi
  B. detection_tracking          — + asosiasi trajektori (track)
  C. detection_tracking_temporal — + agregasi temporal TANPA lapisan identitas
                                   (entitas = track, BUKAN siswa)
  D. full_student_level_pipeline — + fixed-seat identity + penggabungan fragmen

Tujuan: menunjukkan kontribusi tracking, pemrosesan temporal, dan lapisan
identitas. Metrik yang TIDAK valid untuk suatu varian dilaporkan sebagai
NOT_APPLICABLE, bukan diisi angka.

Khususnya: varian A-C TIDAK memiliki identitas siswa, sehingga
`n_students` di sana adalah NOT_APPLICABLE. Jumlah track pada varian B/C
TIDAK boleh dibaca sebagai jumlah siswa.
"""
from __future__ import annotations

import pandas as pd

from .aggregate import build_activity_matrix
from .detect_track import summarise_tracks

NA = "NOT_APPLICABLE"


def run_ablation(
    detections: pd.DataFrame,
    fps: float,
    classes: list,
    min_track_frames: int = 30,
    student_detections: pd.DataFrame = None,
    identity_report: dict = None,
    representation: str = "duration",
) -> pd.DataFrame:
    """
    Jalankan keempat varian pada deteksi yang SAMA dan rangkum statistiknya.
    Tidak ada detektor yang dijalankan ulang.
    """
    recs: list = []

    n_det = int(len(detections)) if detections is not None else 0
    n_tracks = (
        int(detections["track_id"].nunique())
        if detections is not None and not detections.empty else 0
    )

    # ---- A. detection_only ---------------------------------------------
    recs.append({
        "variant": "A_detection_only",
        "detections": n_det,
        "raw_track_ids": NA,
        "retained_entities": NA,
        "mean_coverage_ratio": NA,
        "n_students": NA,
        "identity_audit_status": NA,
        "temporal_aggregation": "no",
        "note": "Hanya deteksi per-frame; tanpa asosiasi temporal apa pun.",
    })

    # ---- B. detection_tracking -----------------------------------------
    cov = NA
    if detections is not None and not detections.empty:
        ts = summarise_tracks(fps=fps, detections_df=detections)
        cov = round(float(ts["coverage_ratio"].mean()), 6) if not ts.empty else NA
    recs.append({
        "variant": "B_detection_tracking",
        "detections": n_det,
        "raw_track_ids": n_tracks,
        "retained_entities": n_tracks,
        "mean_coverage_ratio": cov,
        "n_students": NA,
        "identity_audit_status": NA,
        "temporal_aggregation": "no",
        "note": "Track ID mentah. Jumlah track BUKAN jumlah siswa.",
    })

    # ---- C. detection_tracking_temporal (track-level) -------------------
    retained_c, note_c = NA, ""
    try:
        m = build_activity_matrix(
            detections_csv="", fps=fps, output_csv="",
            min_track_frames=min_track_frames, representation=representation,
            entity_column="track_id", classes=classes,
            strict_classes=False, detections_df=detections,
        )
        retained_c = int(len(m))
        note_c = ("Agregasi temporal pada track. Satu siswa terfragmentasi "
                  "masih dihitung sebagai beberapa entitas.")
    except Exception as e:
        note_c = f"gagal: {type(e).__name__}: {e}"
    recs.append({
        "variant": "C_detection_tracking_temporal",
        "detections": n_det,
        "raw_track_ids": n_tracks,
        "retained_entities": retained_c,
        "mean_coverage_ratio": cov,
        "n_students": NA,
        "identity_audit_status": NA,
        "temporal_aggregation": "yes",
        "note": note_c,
    })

    # ---- D. full_student_level_pipeline ---------------------------------
    retained_d, n_students, status, note_d = NA, NA, NA, ""
    if student_detections is not None and not student_detections.empty:
        try:
            md = build_activity_matrix(
                detections_csv="", fps=fps, output_csv="",
                min_track_frames=min_track_frames, representation=representation,
                entity_column="student_id", classes=classes,
                strict_classes=False, detections_df=student_detections,
            )
            retained_d = int(len(md))
            n_students = int(md.index.nunique())
            note_d = ("Fragmen track digabung menjadi identitas siswa "
                      "berbasis kursi tetap.")
        except Exception as e:
            note_d = f"gagal: {type(e).__name__}: {e}"
        if identity_report:
            status = "PASSED" if identity_report.get("passed") else "FAILED"
    else:
        note_d = ("Tidak dijalankan: deteksi tingkat-siswa tidak tersedia "
                  "(student_seats.csv belum ada atau validasi identitas gagal).")
        status = "NOT_RUN"

    recs.append({
        "variant": "D_full_student_level_pipeline",
        "detections": n_det,
        "raw_track_ids": n_tracks,
        "retained_entities": retained_d,
        "mean_coverage_ratio": cov,
        "n_students": n_students,
        "identity_audit_status": status,
        "temporal_aggregation": "yes",
        "note": note_d,
    })

    return pd.DataFrame(recs)
