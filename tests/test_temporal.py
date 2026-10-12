"""Test agregasi temporal, transisi, dan perbandingan frequency-vs-duration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import AHP_MATRIX, CRITERIA, ORIENTATION, det_row
from src.aggregate import (AggregationError, activity_transitions,
                           build_activity_matrix, build_both_representations,
                           check_class_coverage)
from src.ahp import ahp_weights
from src.identity import (IdentityConfig, assign_track_to_seat,
                          build_student_level_detections)
from src.temporal_analysis import compare_representations, per_entity_rank_table

FPS = 30.0


def _det(student, frames, cls="Diam", track=1):
    return pd.DataFrame([{"frame": f, "track_id": track, "student_id": student,
                          "class_name": cls, "confidence": 0.9} for f in frames])


def test_duration_uses_unique_frames_not_rows():
    """Dua deteksi pada frame yang sama TIDAK boleh menggandakan durasi."""
    rows = [{"frame": f, "track_id": 1, "student_id": "S01", "class_name": "Diam",
             "confidence": 0.9} for f in range(30) for _ in range(2)]
    m = build_activity_matrix("", FPS, "", detections_df=pd.DataFrame(rows),
                              entity_column="student_id", min_track_frames=10)
    assert m.loc["S01", "Diam"] == pytest.approx(30 / FPS)


def test_frequency_representation():
    m = build_activity_matrix("", FPS, "", detections_df=_det("S01", range(60)),
                              entity_column="student_id", min_track_frames=10,
                              representation="frequency")
    assert m.loc["S01", "Diam"] == 60.0


def test_duration_equals_frequency_over_fps():
    d = _det("S01", range(90))
    a = build_activity_matrix("", FPS, "", detections_df=d, entity_column="student_id",
                              min_track_frames=10, representation="duration")
    b = build_activity_matrix("", FPS, "", detections_df=d, entity_column="student_id",
                              min_track_frames=10, representation="frequency")
    assert a.loc["S01", "Diam"] == pytest.approx(b.loc["S01", "Diam"] / FPS)


def test_unknown_class_raises_not_silent():
    """Regresi: kelas tak dikenal dulu dibuang diam-diam menjadi baris nol."""
    d = _det("S01", range(60), cls="Hand Raising")
    with pytest.raises(AggregationError, match="tidak ada di daftar kriteria"):
        build_activity_matrix("", FPS, "", detections_df=d,
                              entity_column="student_id", min_track_frames=10,
                              classes=CRITERIA)


def test_unknown_class_warning_mode():
    d = _det("S01", range(60), cls="Hand Raising")
    m = build_activity_matrix("", FPS, "", detections_df=d, entity_column="student_id",
                              min_track_frames=10, classes=CRITERIA,
                              strict_classes=False)
    assert (m.loc["S01"] == 0).all()   # tetap nol, tetapi sudah diperingatkan


def test_check_class_coverage_lists_unknown():
    d = _det("S01", range(5), cls="Sleeping")
    assert check_class_coverage(d, CRITERIA, strict=False) == ["Sleeping"]


def test_student_level_requires_student_id():
    d = pd.DataFrame([{"frame": f, "track_id": 1, "class_name": "Diam",
                       "confidence": 0.9} for f in range(60)])
    with pytest.raises(AggregationError, match="student_id"):
        build_activity_matrix("", FPS, "", detections_df=d,
                              entity_column="student_id", min_track_frames=10)


def test_track_level_entities_are_prefixed():
    d = pd.DataFrame([{"frame": f, "track_id": 15, "class_name": "Diam",
                       "confidence": 0.9} for f in range(60)])
    m = build_activity_matrix("", FPS, "", detections_df=d,
                              entity_column="track_id", min_track_frames=10)
    assert list(m.index) == ["track_15"], "entitas track tidak boleh tampak seperti siswa"


def test_legacy_identity_map_rejected(tmp_path):
    p = tmp_path / "identity_map.csv"
    p.write_text("track_id,student_id\n1,S01\n", encoding="utf-8")
    d = pd.DataFrame([{"frame": f, "track_id": 1, "class_name": "Diam",
                       "confidence": 0.9} for f in range(60)])
    with pytest.raises(AggregationError, match="tidak lagi didukung"):
        build_activity_matrix("", FPS, "", detections_df=d, entity_column="track_id",
                              identity_map_csv=str(p), min_track_frames=10)


def test_min_track_frames_filter():
    d = pd.concat([_det("S01", range(60)), _det("S02", range(5), track=2)])
    m = build_activity_matrix("", FPS, "", detections_df=d, entity_column="student_id",
                              min_track_frames=30)
    assert list(m.index) == ["S01"]


def test_empty_detections_raises():
    with pytest.raises(AggregationError, match="kosong"):
        build_activity_matrix("", FPS, "", detections_df=pd.DataFrame(),
                              entity_column="student_id")


def test_all_entities_filtered_raises():
    with pytest.raises(AggregationError, match="min_track_frames"):
        build_activity_matrix("", FPS, "", detections_df=_det("S01", range(5)),
                              entity_column="student_id", min_track_frames=1000)


def test_invalid_representation_and_fps():
    d = _det("S01", range(60))
    with pytest.raises(ValueError, match="representation"):
        build_activity_matrix("", FPS, "", detections_df=d, representation="bogus",
                              entity_column="student_id")
    with pytest.raises(ValueError, match="fps"):
        build_activity_matrix("", 0.0, "", detections_df=d, entity_column="student_id")


def test_both_representations_from_same_detections():
    d = _det("S01", range(60))
    both = build_both_representations(d, FPS, min_track_frames=10, classes=CRITERIA)
    assert set(both) == {"duration", "frequency"}
    assert both["frequency"].loc["S01", "Diam"] == 60.0


def test_activity_transitions():
    d = pd.DataFrame(
        [{"frame": f, "track_id": 1, "student_id": "S01",
          "class_name": "Diam" if f < 10 else "Berdiri", "confidence": 0.9}
         for f in range(20)])
    t = activity_transitions(d)
    assert t.iloc[0]["n_class_switches"] == 1
    assert "Diam->Berdiri" in t.iloc[0]["transition_counts"]


def test_transitions_empty_safe():
    assert activity_transitions(pd.DataFrame()).empty


def test_compare_representations_runs():
    rng = np.random.default_rng(1)
    idx = [f"S{i:02d}" for i in range(1, 12)]
    dur = pd.DataFrame(rng.integers(1, 90, size=(11, 4)), columns=CRITERIA,
                       index=idx).astype(float)
    both = {"duration": dur, "frequency": dur * FPS}
    w = ahp_weights(np.array(AHP_MATRIX), CRITERIA)["weights"]
    out = compare_representations(both, w, ORIENTATION)
    assert out.iloc[0]["status"] == "OK"
    # skala linier seragam tidak boleh mengubah ranking
    assert out.iloc[0]["spearman_rho"] == pytest.approx(1.0)
    assert out.iloc[0]["n_rank_changes"] == 0


def test_compare_representations_not_available():
    out = compare_representations(
        {"duration": pd.DataFrame(), "frequency": pd.DataFrame()},
        {c: 0.25 for c in CRITERIA}, ORIENTATION)
    assert out.iloc[0]["status"] == "NOT_AVAILABLE"


def test_per_entity_rank_table():
    rng = np.random.default_rng(2)
    idx = [f"S{i:02d}" for i in range(1, 8)]
    dur = pd.DataFrame(rng.integers(1, 90, size=(7, 4)), columns=CRITERIA,
                       index=idx).astype(float)
    w = ahp_weights(np.array(AHP_MATRIX), CRITERIA)["weights"]
    t = per_entity_rank_table({"duration": dur, "frequency": dur * 2}, w, ORIENTATION)
    assert len(t) == 7
    assert "rank_displacement" in t.columns


def test_end_to_end_identity_to_matrix(seats):
    """Integrasi: fragmen track -> siswa -> matriks aktivitas."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = [det_row(f, 15 if f <= 100 else 37, s07, cls="Berdiri")
            for f in range(1, 181)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats,
                             IdentityConfig(expected_students=21))
    sl = build_student_level_detections(a)
    m = build_activity_matrix("", FPS, "", detections_df=sl,
                              entity_column="student_id", min_track_frames=30,
                              classes=CRITERIA)
    assert list(m.index) == ["S07"]
    assert m.loc["S07", "Berdiri"] == pytest.approx(180 / FPS)
