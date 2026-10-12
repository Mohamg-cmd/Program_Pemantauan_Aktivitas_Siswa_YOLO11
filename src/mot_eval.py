"""
Stage 26 (future-proof) — Antarmuka evaluasi MOT formal.

STATUS SAAT INI: ground-truth trajectory TIDAK tersedia di paket ini,
sehingga HOTA/IDF1/MOTA TIDAK DAPAT dan TIDAK AKAN dihitung. Modul ini
sengaja hanya menyediakan antarmuka + validasi format, supaya metrik dapat
dihitung begitu anotasi trajektori tersedia.

Tidak ada nilai metrik yang dikarang di modul ini.

Format yang diharapkan nanti:
  ground truth : frame, gt_id, x1, y1, x2, y2
  prediksi     : frame, track_id, x1, y1, x2, y2
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

GT_REQUIRED = ["frame", "gt_id", "x1", "y1", "x2", "y2"]
PRED_REQUIRED = ["frame", "track_id", "x1", "y1", "x2", "y2"]

UNAVAILABLE_MESSAGE = (
    "WARNING:\nFormal MOT metrics unavailable because trajectory ground truth "
    "is absent.\nHOTA/IDF1/MOTA TIDAK dihitung dan TIDAK boleh dilaporkan.\n"
    "Sediakan CSV ground-truth trajectory (frame, gt_id, bbox) lalu panggil "
    "evaluate_mot(gt_csv=..., pred_csv=...)."
)


class GroundTruthUnavailable(RuntimeError):
    """Ground-truth trajektori tidak tersedia."""


def validate_trajectory_csv(path: str, required: list, kind: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise GroundTruthUnavailable(f"{kind} tidak ditemukan: {path}")
    df = pd.read_csv(p)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{kind} ({path}) kehilangan kolom: {missing}. Wajib: {required}")
    return df


def mot_status(gt_csv: Optional[str] = None) -> dict:
    """Status ketersediaan evaluasi MOT — dipakai untuk menandai sheet Excel."""
    available = bool(gt_csv) and Path(gt_csv).exists()
    return {
        "ground_truth_available": available,
        "gt_path": gt_csv or "NOT_AVAILABLE",
        "HOTA": "NOT_AVAILABLE",
        "IDF1": "NOT_AVAILABLE",
        "MOTA": "NOT_AVAILABLE",
        "status": "READY_TO_COMPUTE" if available else "NOT_AVAILABLE",
        "note": (
            "Ground truth tersedia; hitung dengan pustaka MOT resmi "
            "(mis. TrackEval) memakai evaluate_mot()."
            if available else UNAVAILABLE_MESSAGE.replace("\n", " ")
        ),
    }


def evaluate_mot(gt_csv: Optional[str] = None, pred_csv: Optional[str] = None) -> dict:
    """
    Hitung metrik MOT formal — HANYA bila ground truth tersedia.

    Implementasi metrik sengaja TIDAK disertakan: HOTA/IDF1/MOTA harus
    dihitung dengan implementasi rujukan yang tervalidasi (mis. TrackEval)
    agar angkanya dapat dipertanggungjawabkan secara ilmiah. Menuliskan
    ulang metrik ini dari nol berisiko menghasilkan angka yang salah namun
    tampak meyakinkan.
    """
    if not gt_csv or not Path(gt_csv).exists():
        raise GroundTruthUnavailable(UNAVAILABLE_MESSAGE)
    gt = validate_trajectory_csv(gt_csv, GT_REQUIRED, "ground truth")
    pred = validate_trajectory_csv(pred_csv, PRED_REQUIRED, "prediksi") if pred_csv else None
    return {
        "status": "READY_TO_COMPUTE",
        "n_gt_rows": len(gt),
        "n_gt_ids": int(gt["gt_id"].nunique()),
        "n_pred_rows": int(len(pred)) if pred is not None else 0,
        "n_pred_tracks": int(pred["track_id"].nunique()) if pred is not None else 0,
        "HOTA": "NOT_COMPUTED_USE_TRACKEVAL",
        "IDF1": "NOT_COMPUTED_USE_TRACKEVAL",
        "MOTA": "NOT_COMPUTED_USE_TRACKEVAL",
        "note": "Gunakan TrackEval pada file terformat MOTChallenge untuk angka resmi.",
    }
