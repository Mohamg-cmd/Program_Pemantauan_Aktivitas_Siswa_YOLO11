"""
Seat calibration tool — buat data/student_seats.csv dari frame referensi video.

Koordinat kursi TIDAK dapat ditebak oleh siapa pun: koordinat bergantung pada
penempatan kamera, resolusi, dan tata letak kelas Anda. Karena itu tidak ada
koordinat bawaan di repositori ini; Anda menandainya sendiri satu kali.

Pemakaian
---------
    python tools/calibrate_seats.py --video data/classroom.mp4 \
                                    --output data/student_seats.csv

Kontrol (GUI OpenCV):
    seret kiri   : gambar ROI persegi untuk satu kursi
    ENTER / SPACE: simpan ROI sebagai kursi berikutnya (S01, S02, ...)
    u            : batalkan kursi terakhir
    f / b        : maju / mundur 30 frame (cari frame yang jelas)
    s            : simpan CSV dan keluar
    q / ESC      : keluar tanpa menyimpan

Pusat kursi diambil dari titik TENGAH-BAWAH ROI, konsisten dengan titik
jangkar bottom-center yang dipakai src/identity.py saat pencocokan.

Mode tanpa GUI (--from-json) disediakan untuk lingkungan headless.
Modul ini TIDAK PERNAH diimpor oleh unit test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HEADER = ["student_id", "seat_id", "seat_x", "seat_y",
          "roi_x1", "roi_y1", "roi_x2", "roi_y2"]


def _row(i: int, x1: float, y1: float, x2: float, y2: float) -> dict:
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    return {
        "student_id": f"S{i:02d}",
        "seat_id": f"seat_{i:02d}",
        # pusat kursi = tengah-bawah ROI (sejajar dengan anchor bottom-center)
        "seat_x": round((x1 + x2) / 2.0, 2),
        "seat_y": round(y2, 2),
        "roi_x1": round(x1, 2), "roi_y1": round(y1, 2),
        "roi_x2": round(x2, 2), "roi_y2": round(y2, 2),
    }


def save_rows(rows: list, output: str, expected: int) -> None:
    if not rows:
        print("Tidak ada kursi yang ditandai; tidak ada yang disimpan.")
        return
    df = pd.DataFrame(rows, columns=HEADER)
    if len(df) != expected:
        print(f"WARNING: {len(df)} kursi ditandai, diharapkan {expected}. "
              f"Pipeline akan menolak sampai jumlahnya cocok atau "
              f"identity.expected_students disesuaikan.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Tersimpan {len(df)} kursi -> {output}")


def from_json(json_path: str, output: str, expected: int) -> None:
    """Konversi ROI dari JSON [[x1,y1,x2,y2], ...] (alur headless)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = [_row(i, *box) for i, box in enumerate(data, start=1)]
    save_rows(rows, output, expected)


def interactive(video: str, output: str, expected: int, frame_index: int) -> None:
    try:
        import cv2
    except ImportError:
        sys.exit("ERROR: opencv-python dibutuhkan. Pasang: pip install opencv-python")

    if not Path(video).exists():
        sys.exit(f"ERROR: video tidak ditemukan: {video}")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"ERROR: tidak dapat membuka video: {video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"ERROR: tidak dapat membaca frame {frame_index}")

    rows: list = []
    win = "Seat calibration - ENTER=simpan  u=undo  f/b=geser frame  s=selesai  q=keluar"
    print(__doc__)

    while True:
        disp = frame.copy()
        for r in rows:
            cv2.rectangle(disp, (int(r["roi_x1"]), int(r["roi_y1"])),
                          (int(r["roi_x2"]), int(r["roi_y2"])), (0, 200, 0), 2)
            cv2.putText(disp, r["student_id"],
                        (int(r["roi_x1"]), max(12, int(r["roi_y1"]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(disp, f"{len(rows)}/{expected} kursi ditandai",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        box = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
        if box and box[2] > 0 and box[3] > 0:
            x, y, w, h = box
            rows.append(_row(len(rows) + 1, x, y, x + w, y + h))
            print(f"  + {rows[-1]['student_id']} seat_x={rows[-1]['seat_x']} "
                  f"seat_y={rows[-1]['seat_y']}")

        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            print("Dibatalkan."); break
        if k == ord("u") and rows:
            print(f"  - batal {rows.pop()['student_id']}")
        elif k == ord("s"):
            save_rows(rows, output, expected); break
        elif k in (ord("f"), ord("b")):
            frame_index = max(0, frame_index + (30 if k == ord("f") else -30))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, f2 = cap.read()
            if ok:
                frame = f2
                print(f"  frame -> {frame_index}")
        if len(rows) >= expected:
            print(f"{expected} kursi lengkap. Tekan 's' untuk menyimpan.")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Kalibrasi kursi tetap untuk identitas siswa")
    p.add_argument("--video", help="Path video kelas")
    p.add_argument("--output", default="data/student_seats.csv")
    p.add_argument("--expected", type=int, default=21, help="Jumlah siswa (default 21)")
    p.add_argument("--frame", type=int, default=0, help="Indeks frame referensi")
    p.add_argument("--from-json", help="Alur headless: JSON berisi [[x1,y1,x2,y2], ...]")
    a = p.parse_args()

    if a.from_json:
        from_json(a.from_json, a.output, a.expected)
    elif a.video:
        interactive(a.video, a.output, a.expected, a.frame)
    else:
        p.error("berikan --video (mode GUI) atau --from-json (mode headless)")


if __name__ == "__main__":
    main()
