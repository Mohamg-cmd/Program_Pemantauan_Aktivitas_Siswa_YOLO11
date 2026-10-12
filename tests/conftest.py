"""Fixture bersama untuk seluruh test. Tidak butuh GPU/YOLO/video."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CRITERIA = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
ORIENTATION = {
    "Angkat Tangan": "benefit", "Berdiri": "benefit",
    "Diam": "cost", "Tidur": "cost",
}
AHP_MATRIX = [
    [1.000, 2.000, 3.000, 4.000],
    [0.500, 1.000, 2.000, 3.000],
    [0.333, 0.500, 1.000, 2.000],
    [0.250, 0.333, 0.500, 1.000],
]


def make_seat_grid(n: int = 21, pitch: int = 200, half: int = 70) -> pd.DataFrame:
    """
    Geometri kursi SINTETIK untuk pengujian perangkat lunak saja.
    BUKAN koordinat kelas sungguhan dan tidak boleh dipakai sebagai data.
    """
    rows = []
    for i in range(n):
        r, c = divmod(i, 5)
        x, y = 100 + c * pitch, 100 + r * pitch
        rows.append({
            "student_id": f"S{i+1:02d}", "seat_id": f"seat_{i+1:02d}",
            "seat_x": x, "seat_y": y,
            "roi_x1": x - half, "roi_y1": y - half,
            "roi_x2": x + half, "roi_y2": y + half,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def seats_csv(tmp_path) -> str:
    p = tmp_path / "student_seats.csv"
    make_seat_grid().to_csv(p, index=False)
    return str(p)


@pytest.fixture
def seats(seats_csv):
    from src.identity import load_seat_map
    return load_seat_map(seats_csv, 21)


def det_row(frame, track_id, seat, cls="Diam", conf=0.9, dx=0.0, dy=0.0, w=80, h=120):
    """Deteksi dengan titik bottom-center tepat di (seat_x+dx, seat_y+dy)."""
    cx, by = seat.seat_x + dx, seat.seat_y + dy
    return {"frame": frame, "track_id": track_id, "class_id": 2, "class_name": cls,
            "confidence": conf, "x1": cx - w / 2, "y1": by - h,
            "x2": cx + w / 2, "y2": by}
