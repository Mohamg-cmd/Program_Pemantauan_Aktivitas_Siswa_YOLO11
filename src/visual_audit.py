"""
Stage — Audit identitas VISUAL (opsional).

Membuat output/identity_audit.mp4 dengan overlay:

    S07 | track_15 | seat_07 | ID=0.94 | Standing

Status khusus ditandai AMBIGUOUS / UNASSIGNED / DUPLICATE agar pengguna dapat
memeriksa secara visual bahwa track_15, track_37, dan track_82 memang orang
yang sama. Video ini TIDAK PERNAH diwajibkan pada smoke test dan hanya
dibuat bila diminta eksplisit.

Membutuhkan OpenCV dan video sumber.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .identity import (STATUS_AMBIGUOUS, STATUS_ASSIGNED, STATUS_DUPLICATE,
                       STATUS_UNASSIGNED)

_COLORS = {
    STATUS_ASSIGNED: (0, 180, 0),
    STATUS_AMBIGUOUS: (0, 165, 255),
    STATUS_DUPLICATE: (0, 0, 255),
    STATUS_UNASSIGNED: (128, 128, 128),
}


def render_identity_audit_video(
    video_path: str,
    assignments: pd.DataFrame,
    output_path: str,
    seats: Optional[pd.DataFrame] = None,
    max_frames: Optional[int] = None,
    draw_seats: bool = True,
) -> str:
    """Render video audit. Mengembalikan path keluaran."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "OpenCV (opencv-python) dibutuhkan untuk video audit identitas."
        ) from e

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
    if assignments is None or assignments.empty:
        raise ValueError("Tidak ada assignment identitas untuk divisualisasikan.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    by_frame = {f: g for f, g in assignments.groupby("frame")}
    font, idx = cv2.FONT_HERSHEY_SIMPLEX, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and idx >= max_frames):
                break

            if draw_seats and seats is not None:
                for s in seats.itertuples():
                    cv2.rectangle(frame, (int(s.roi_x1), int(s.roi_y1)),
                                  (int(s.roi_x2), int(s.roi_y2)), (90, 90, 90), 1)
                    cv2.putText(frame, str(s.seat_id),
                                (int(s.roi_x1) + 2, int(s.roi_y1) + 14),
                                font, 0.4, (90, 90, 90), 1, cv2.LINE_AA)

            for r in by_frame.get(idx, pd.DataFrame()).itertuples():
                status = getattr(r, "assignment_status", STATUS_UNASSIGNED)
                color = _COLORS.get(status, (200, 200, 200))
                x1, y1, x2, y2 = (int(getattr(r, c)) for c in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if status == STATUS_ASSIGNED:
                    label = (f"{r.student_id} | track_{r.track_id} | {r.seat_id} | "
                             f"ID={r.identity_confidence:.2f} | "
                             f"{getattr(r, 'class_name', '?')}")
                elif status == STATUS_DUPLICATE:
                    label = f"DUPLICATE | track_{r.track_id}"
                elif status == STATUS_AMBIGUOUS:
                    label = f"AMBIGUOUS | track_{r.track_id}"
                else:
                    label = f"UNASSIGNED | track_{r.track_id}"

                (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                ty = max(0, y1 - th - 4)
                cv2.rectangle(frame, (x1, ty), (x1 + tw + 4, ty + th + 4), color, -1)
                cv2.putText(frame, label, (x1 + 2, ty + th),
                            font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f"frame {idx}", (10, h - 12),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return output_path
