"""
Stage 1 — YOLO11 detection + BoT-SORT tracking (ReID opsional).

Menghasilkan CSV per-frame: frame, track_id, class_id, class_name,
confidence, x1, y1, x2, y2.

PENTING — SEMANTIK track_id
---------------------------
`track_id` adalah PENGENAL TRAJEKTORI, BUKAN identitas siswa. Satu siswa
dapat menghasilkan banyak track_id akibat fragmentasi/oklusi/reasosiasi.
Pemetaan ke siswa dilakukan oleh src/identity.py melalui kursi tetap.
Jangan pernah membaca `track_15` sebagai "siswa 15".

CATATAN ReID
------------
Versi baseline mengklaim "BoT-SORT + ReID" pada docstring, tetapi tidak
pernah mengaktifkannya: `botsort.yaml` bawaan Ultralytics memakai
`with_reid: False`. Di sini ReID dapat dinyalakan secara EKSPLISIT lewat
config; bila dinyalakan, file tracker kustom dibuat agar status ReID
tercatat jujur di provenance. ReID tetap hanya cue SEKUNDER — identitas
siswa tetap ditentukan oleh kursi tetap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DETECTION_COLUMNS = [
    "frame", "track_id", "class_id", "class_name",
    "confidence", "x1", "y1", "x2", "y2",
]


def resolve_tracker_config(
    tracker: str = "botsort.yaml",
    with_reid: bool = False,
    reid_model: Optional[str] = None,
    output_dir: str = "output",
) -> str:
    """
    Kembalikan path konfigurasi tracker yang dipakai.

    Bila `with_reid=True`, tulis salinan botsort.yaml dengan `with_reid: true`
    supaya klaim ReID pada manuskrip benar-benar tercermin di konfigurasi
    yang dieksekusi. Bila berkas dasar tidak ditemukan, kembalikan nama
    tracker apa adanya dan biarkan Ultralytics menanganinya.
    """
    if not with_reid:
        return tracker
    try:
        import ultralytics
        base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
        if not base.exists():
            return tracker
        text = base.read_text(encoding="utf-8")
        if "with_reid" in text:
            text = "\n".join(
                "with_reid: True" if ln.strip().startswith("with_reid:") else ln
                for ln in text.splitlines()
            )
        else:
            text += "\nwith_reid: True\n"
        if reid_model:
            if "model:" in text:
                text = "\n".join(
                    f"model: {reid_model}" if ln.strip().startswith("model:") else ln
                    for ln in text.splitlines()
                )
            else:
                text += f"model: {reid_model}\n"
        out = Path(output_dir) / "botsort_reid.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return str(out)
    except Exception:
        return tracker


def run_detection_tracking(
    model_path: str,
    video_path: str,
    output_csv: str,
    tracker: str = "botsort.yaml",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    device: str = "0",
    save_video: bool = False,
    verbose: bool = False,
    with_reid: bool = False,
    reid_model: Optional[str] = None,
    keep_untracked: bool = False,
) -> pd.DataFrame:
    """
    Jalankan pipeline detect+track pada satu video.

    keep_untracked : bool
        Bila True, deteksi tanpa track ID tetap disimpan (track_id = -1).
        Berguna karena asosiasi berbasis kursi tidak selalu membutuhkan
        track ID; versi baseline membuang frame semacam itu seluruhnya.
    """
    # Import di dalam fungsi supaya modul lain (AHP/SAW/identity) tetap bisa
    # di-unit-test tanpa dependency ultralytics.
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"ERROR:\nBobot model tidak ditemukan: {model_path}\n"
            f"Letakkan bobot YOLO11 Anda atau perbarui source.model_path."
        )
    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"ERROR:\nVideo tidak ditemukan: {video_path}\n"
            f"Letakkan video kelas atau perbarui source.video_path."
        )

    tracker_cfg = resolve_tracker_config(
        tracker, with_reid, reid_model, str(Path(output_csv).parent)
    )

    model = YOLO(model_path)
    class_names = model.names  # {0: 'Angkat Tangan', ...}

    rows: list = []
    stream: Iterable = model.track(
        source=video_path, tracker=tracker_cfg, conf=conf, iou=iou,
        imgsz=imgsz, device=device, persist=True, stream=True,
        save=save_video, verbose=verbose,
    )

    for frame_idx, r in enumerate(stream):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        has_ids = r.boxes.id is not None
        if not has_ids and not keep_untracked:
            continue

        ids = (r.boxes.id.int().cpu().tolist() if has_ids
               else [-1] * len(r.boxes))
        clses = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()

        for tid, cid, cf, box in zip(ids, clses, confs, xyxy):
            rows.append({
                "frame": frame_idx,
                "track_id": int(tid),
                "class_id": int(cid),
                "class_name": class_names[int(cid)],
                "confidence": float(cf),
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
            })

    df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def summarise_tracks(
    detections_csv: Optional[str] = None,
    fps: float = 30.0,
    detections_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Ringkasan per track_id (diperluas dari versi baseline).

    Kolom: track_id, n_frames, first_frame, last_frame, span_frames,
    coverage_ratio, duration_sec, mean_confidence, median_confidence,
    dominant_activity, class_switch_count.

        coverage_ratio = observed_frames / span_frames

    coverage_ratio yang rendah menandakan trajektori berlubang (indikator
    fragmentasi). INGAT: jumlah baris di sini adalah jumlah TRACK, bukan
    jumlah siswa.
    """
    cols = ["track_id", "n_frames", "first_frame", "last_frame", "span_frames",
            "coverage_ratio", "duration_sec", "mean_confidence",
            "median_confidence", "dominant_activity", "class_switch_count"]

    if detections_df is not None:
        df = detections_df.copy()
    else:
        if not detections_csv or not Path(detections_csv).exists():
            raise FileNotFoundError(
                f"ERROR:\ndetections.csv tidak ditemukan: {detections_csv}\n"
                f"Jalankan tahap deteksi terlebih dahulu, atau hilangkan --skip-detect."
            )
        df = pd.read_csv(detections_csv)

    if df.empty:
        return pd.DataFrame(columns=cols)
    if fps <= 0:
        raise ValueError(f"fps harus > 0, dapat {fps}")

    recs = []
    for tid, g in df.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        first, last = int(g["frame"].min()), int(g["frame"].max())
        span = last - first + 1
        nf = int(g["frame"].nunique())
        seq = g["class_name"].tolist()
        recs.append({
            "track_id": int(tid),
            "n_frames": nf,
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "coverage_ratio": round(nf / span, 6) if span > 0 else 0.0,
            "duration_sec": nf / fps,
            "mean_confidence": float(g["confidence"].mean())
                if "confidence" in g else float("nan"),
            "median_confidence": float(g["confidence"].median())
                if "confidence" in g else float("nan"),
            "dominant_activity": g["class_name"].value_counts().idxmax()
                if "class_name" in g else None,
            "class_switch_count": sum(1 for a, b in zip(seq, seq[1:]) if a != b),
        })
    return pd.DataFrame(recs, columns=cols)
