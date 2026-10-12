"""
Stage 2 — Agregasi temporal.

Mengubah deteksi per-frame menjadi matriks ENTITAS x KRITERIA.

Perbedaan penting terhadap versi baseline:

  * Entitas default sekarang adalah `student_id` (hasil lapisan identitas),
    bukan `track_id`. Mode `track_level` tetap tersedia HANYA untuk debugging
    dan keluarannya tidak boleh disebut "siswa".
  * Frame UNIK yang dihitung, bukan baris CSV -- sehingga dua fragmen track
    dari siswa yang sama pada frame yang sama tidak lagi dihitung dua kali.
  * Daftar kelas TIDAK lagi di-hardcode; diambil dari config agar tidak
    pernah desync dengan `criteria`.
  * Kelas yang tidak dikenal TIDAK lagi dibuang diam-diam.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

# Urutan kanonis kelas BAWAAN (kompatibilitas mundur). config.yaml adalah
# sumber kebenaran yang sesungguhnya.
CLASSES: list[str] = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]

VALID_REPRESENTATIONS = {"duration", "frequency"}


class AggregationError(RuntimeError):
    """Agregasi temporal tidak dapat diselesaikan dengan aman."""


def _resolve_classes(classes: Optional[Sequence[str]]) -> list[str]:
    return list(classes) if classes else list(CLASSES)


def check_class_coverage(
    detections: pd.DataFrame,
    classes: Sequence[str],
    strict: bool = True,
) -> list[str]:
    """
    Pastikan `class_name` pada deteksi cocok dengan kriteria yang dikonfigurasi.

    Versi baseline membuang kelas tak dikenal TANPA SUARA, sehingga seluruh
    deteksi bisa berubah menjadi matriks nol dan tetap ikut di-ranking.
    """
    if detections.empty or "class_name" not in detections.columns:
        return []
    found = set(detections["class_name"].dropna().astype(str).unique())
    unknown = sorted(found - set(classes))
    if unknown:
        msg = (
            f"ERROR:\nclass_name pada deteksi tidak ada di daftar kriteria.\n"
            f"  Tidak dikenal : {unknown}\n"
            f"  Dikonfigurasi : {list(classes)}\n"
            f"Nama kelas model YOLO11 harus sama persis dengan `criteria` di "
            f"config.yaml. Bila dibiarkan, deteksi tersebut hilang dari matriks."
        )
        if strict:
            raise AggregationError(msg)
        print(msg.replace("ERROR:", "WARNING:"))
    return unknown


def build_activity_matrix(
    detections_csv: str,
    fps: float,
    output_csv: str,
    identity_map_csv: Optional[str] = None,
    min_track_frames: int = 30,
    representation: str = "duration",
    entity_column: str = "track_id",
    classes: Optional[Sequence[str]] = None,
    strict_classes: bool = True,
    detections_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Bangun matriks entitas x kelas aktivitas.

    Catatan durasi
    --------------
    Durasi dihitung dari jumlah frame UNIK per (entitas, kelas) dibagi fps,
    sehingga deteksi ganda pada frame yang sama tidak menggelembungkan durasi.
    """
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"representation harus 'duration' atau 'frequency', bukan {representation!r}"
        )
    if fps <= 0:
        raise ValueError(f"fps harus > 0, dapat {fps}")

    cls = _resolve_classes(classes)
    df = detections_df.copy() if detections_df is not None else pd.read_csv(detections_csv)

    if df.empty:
        raise AggregationError(
            "Tidak ada deteksi untuk diagregasi (detections kosong). "
            "Periksa tahap deteksi/tracking sebelum melanjutkan."
        )

    check_class_coverage(df, cls, strict=strict_classes)

    if entity_column == "student_id":
        if "student_id" not in df.columns:
            raise AggregationError(
                "ERROR:\nMode student_level membutuhkan kolom `student_id` yang "
                "dihasilkan src/identity.py. Jalankan lapisan identitas dahulu.\n"
                "Pipeline TIDAK akan diam-diam memakai track_id sebagai siswa."
            )
        df = df[df["student_id"].notna()].copy()
        if df.empty:
            raise AggregationError(
                "Tidak ada deteksi dengan student_id valid setelah asosiasi identitas."
            )
        df["entity"] = df["student_id"].astype(str)
    elif entity_column == "track_id":
        if identity_map_csv and Path(identity_map_csv).exists():
            raise AggregationError(
                "ERROR:\nidentity_map.csv (track_id -> student_id) tidak lagi "
                "didukung: pemetaan satu-ke-satu track->siswa keliru karena "
                "BoT-SORT memecah satu siswa menjadi banyak track.\n"
                "Gunakan identity.mode: student_level dengan data/student_seats.csv."
            )
        df["entity"] = "track_" + df["track_id"].astype(str)
    else:
        raise ValueError(
            f"entity_column harus 'student_id' atau 'track_id', bukan {entity_column!r}"
        )

    frames_per_entity = df.groupby("entity")["frame"].nunique()
    keep = frames_per_entity[frames_per_entity >= min_track_frames].index
    dropped = sorted(set(frames_per_entity.index) - set(keep))
    if dropped:
        print(
            f"INFO: {len(dropped)} entitas dibuang karena < min_track_frames="
            f"{min_track_frames} frame unik: {dropped[:10]}"
            f"{' ...' if len(dropped) > 10 else ''}"
        )
    df = df[df["entity"].isin(keep)]

    if df.empty:
        raise AggregationError(
            f"Tidak ada entitas yang lulus min_track_frames={min_track_frames}. "
            f"Turunkan ambangnya atau periksa hasil tracking/identitas."
        )

    freq = (
        df.groupby(["entity", "class_name"])["frame"]
          .nunique()                      # <- bukan size(): anti double-count
          .unstack(fill_value=0)
    )
    for c in cls:
        if c not in freq.columns:
            freq[c] = 0
    freq = freq[cls]

    matrix = freq / fps if representation == "duration" else freq.astype(float)
    matrix = matrix.astype(float)
    matrix.index.name = "entity"
    matrix = matrix.sort_index()

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(output_csv)
    return matrix


def build_both_representations(
    detections_df: pd.DataFrame,
    fps: float,
    entity_column: str = "student_id",
    classes: Optional[Sequence[str]] = None,
    min_track_frames: int = 30,
    strict_classes: bool = True,
) -> dict:
    """
    Bangun matriks `duration` DAN `frequency` dari deteksi yang SAMA.

    Detector/tracker tidak dijalankan dua kali -- kedua representasi wajib
    memakai video, deteksi, dan asosiasi identitas yang identik.
    """
    return {
        rep: build_activity_matrix(
            detections_csv="", fps=fps, output_csv="",
            min_track_frames=min_track_frames, representation=rep,
            entity_column=entity_column, classes=classes,
            strict_classes=strict_classes, detections_df=detections_df,
        )
        for rep in ("duration", "frequency")
    }


def activity_transitions(
    detections_df: pd.DataFrame,
    entity_column: str = "student_id",
) -> pd.DataFrame:
    """
    Statistik transisi kelas aktivitas per entitas.

    HANYA untuk analisis temporal/debugging. TIDAK dipakai sebagai kriteria
    ranking kecuali diminta eksplisit.
    """
    cols = ["student_id", "n_class_switches", "dominant_activity", "transition_counts"]
    if detections_df is None or detections_df.empty:
        return pd.DataFrame(columns=cols)
    if entity_column not in detections_df.columns:
        return pd.DataFrame(columns=cols)

    recs = []
    for ent, g in detections_df.dropna(subset=[entity_column]).groupby(entity_column):
        g = g.sort_values("frame")
        seq = g["class_name"].tolist()
        trans: dict = {}
        switches = 0
        for a, b in zip(seq, seq[1:]):
            if a != b:
                switches += 1
                trans[f"{a}->{b}"] = trans.get(f"{a}->{b}", 0) + 1
        recs.append({
            "student_id": ent,
            "n_class_switches": switches,
            "dominant_activity": g["class_name"].value_counts().idxmax(),
            "transition_counts": ";".join(f"{k}={v}" for k, v in sorted(trans.items())),
        })
    return pd.DataFrame(recs, columns=cols).sort_values("student_id").reset_index(drop=True)
