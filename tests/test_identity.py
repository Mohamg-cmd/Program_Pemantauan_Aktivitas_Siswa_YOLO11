"""
Test lapisan identitas — inti perbaikan track_id != student_id.
Tanpa GPU/YOLO/video. Tidak pernah mengimpor GUI kalibrasi.
"""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import det_row, make_seat_grid
from src.identity import (STATUS_ASSIGNED, STATUS_DUPLICATE, STATUS_UNASSIGNED,
                          IdentityConfig, IdentityConfigError,
                          assign_track_to_seat, bottom_center,
                          build_student_level_detections, load_seat_map,
                          long_gap_audit, merge_track_fragments,
                          validate_seat_assignment)

FPS = 30.0


def cfg(**kw):
    base = dict(expected_students=21, max_seat_distance=150.0,
                temporal_smoothing_window=15)
    base.update(kw)
    return IdentityConfig(**base)


# ---------------------------------------------------------------- skenario inti
def test_three_fragments_become_one_student(seats):
    """
    SKENARIO SPESIFIKASI:
      frame   1-100 -> track 15
      frame 101-150 -> track 37
      frame 151-200 -> track 82
    Ketiganya pada kursi S07 => SATU siswa, 200 frame, 200/FPS detik.
    """
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = []
    for f in range(1, 201):
        tid = 15 if f <= 100 else (37 if f <= 150 else 82)
        rows.append(det_row(f, tid, s07))

    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    _, audit = merge_track_fragments(a, FPS, cfg())

    assert list(audit["student_id"]) == ["S07"], "harus tepat SATU identitas siswa"
    row = audit.iloc[0]
    assert row["n_track_fragments"] == 3
    assert row["total_frames"] == 200
    assert row["total_duration_sec"] == pytest.approx(200 / FPS)
    assert set(row["assigned_track_ids"].split(";")) == {"15", "37", "82"}


def test_fragments_never_become_separate_students(seats):
    """track_15 dan track_37 TIDAK boleh menjadi S07 dan S08."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = [det_row(f, 15 if f <= 100 else 37, s07) for f in range(1, 151)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    assigned = a[a.assignment_status == STATUS_ASSIGNED]
    assert set(assigned["student_id"].unique()) == {"S07"}
    assert assigned["student_id"].nunique() == 1


def test_student_id_never_derived_from_track_id(seats):
    """track_15 tidak boleh memetakan ke 'S15' hanya karena angkanya sama."""
    s03 = seats[seats.student_id == "S03"].iloc[0]
    rows = [det_row(f, 15, s03) for f in range(1, 61)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    got = a[a.assignment_status == STATUS_ASSIGNED]["student_id"].unique()
    assert list(got) == ["S03"], "identitas harus dari KURSI, bukan dari track_id"
    assert "S15" not in got


def test_simultaneous_tracks_on_same_seat(seats):
    """Dua track aktif pada kursi yang sama: satu primary, satu duplicate."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = []
    for f in range(1, 61):
        rows.append(det_row(f, 15, s07, conf=0.95))
        rows.append(det_row(f, 99, s07, conf=0.50, dx=30, dy=25))
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())

    per_frame = a[a.assignment_status == STATUS_ASSIGNED].groupby("frame").size()
    assert (per_frame == 1).all(), "satu kursi hanya boleh satu track per frame"
    assert (a.assignment_status == STATUS_DUPLICATE).sum() == 60

    # tidak boleh double-count aktivitas
    assert len(build_student_level_detections(a)) == 60


def test_no_double_counting_across_fragments(seats):
    """Fragmen tumpang tindih pada frame yang sama tidak menggandakan durasi."""
    s05 = seats[seats.student_id == "S05"].iloc[0]
    rows = []
    for f in range(1, 101):
        rows.append(det_row(f, 1, s05))
        if f > 50:
            rows.append(det_row(f, 2, s05, dx=15))
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    _, audit = merge_track_fragments(a, FPS, cfg())
    assert audit.iloc[0]["total_frames"] == 100, "frame unik, bukan jumlah baris"


def test_boundary_track_does_not_flicker_between_two_students(seats):
    """Track-level resolution must not alternate S07/S08 frame-by-frame."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    s08 = seats[seats.student_id == "S08"].iloc[0]
    mid = (s07.seat_x + s08.seat_x) / 2
    rows = [{**det_row(f, 5, s07), "x1": mid + (8 if f % 2 else -8) - 40,
             "x2": mid + (8 if f % 2 else -8) + 40} for f in range(1, 81)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    assigned = a[a.assignment_status == STATUS_ASSIGNED]
    if not assigned.empty:
        assert assigned.student_id.nunique() == 1
    # A truly ambiguous boundary track is allowed to be marked ambiguous,
    # but it must never oscillate between two student IDs.
    assert not set(a.student_id.dropna().astype(str)) - {"S07", "S08"}


def test_far_detection_is_unassigned_not_forced(seats):
    """Deteksi jauh dari semua kursi tidak boleh dipaksakan ke siswa mana pun."""
    rows = [{"frame": f, "track_id": 7, "class_id": 2, "class_name": "Diam",
             "confidence": 0.9, "x1": 9000, "y1": 9000, "x2": 9080, "y2": 9120}
            for f in range(1, 41)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    assert (a.assignment_status == STATUS_UNASSIGNED).all()
    assert a["student_id"].isna().all()


def test_bottom_center_used_not_bbox_center(seats):
    """Siswa berdiri: bbox memanjang ke atas, jangkar tetap di kursi."""
    import numpy as np
    x1, y1, x2, y2 = np.array([100.]), np.array([0.]), np.array([180.]), np.array([300.])
    ax, ay = bottom_center(x1, y1, x2, y2)
    assert ax[0] == 140.0 and ay[0] == 300.0     # bukan pusat (150.0)

    s01 = seats[seats.student_id == "S01"].iloc[0]
    standing = [det_row(f, 3, s01, h=260) for f in range(1, 41)]   # bbox tinggi
    a = assign_track_to_seat(pd.DataFrame(standing), seats, cfg())
    assert (a[a.assignment_status == STATUS_ASSIGNED]["student_id"] == "S01").all()



def test_long_gap_strong_fixed_seat_evidence_is_auto_classified(seats):
    """Long gaps are diagnostic; strong evidence on both fragments is auto-classified."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = []
    for f in range(1, 41):
        rows.append(det_row(f, 15, s07))
    for f in range(242, 282):  # 201-frame gap
        rows.append(det_row(f, 37, s07))

    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg(reassociation_gap_frames=90))
    t2s, _ = merge_track_fragments(a, FPS, cfg(reassociation_gap_frames=90), seats=seats)
    long_df = t2s[t2s["long_gap"]]

    assert len(long_df) == 1
    row = long_df.iloc[0]
    assert row["student_id"] == "S07"
    assert row["prev_track_id"] == 15
    assert row["long_gap_status"] == "strong_fixed_seat_evidence"
    assert row["long_gap_evidence_score"] >= 0.90

    audit = long_gap_audit(t2s, cfg(reassociation_gap_frames=90))
    assert len(audit) == 1
    assert audit.loc[0, "long_gap_status"] == "strong_fixed_seat_evidence"


def test_long_gap_weak_evidence_remains_for_review(seats):
    """Long-gap links with weak seat evidence must remain explicitly reviewable."""
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = []
    for f in range(1, 11):
        rows.append(det_row(f, 15, s07, dx=0, dy=0))
    for f in range(242, 252):
        # Move the second fragment far enough from the seat to weaken evidence.
        rows.append(det_row(f, 37, s07, dx=80, dy=0))

    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg(reassociation_gap_frames=90))
    t2s, _ = merge_track_fragments(a, FPS, cfg(reassociation_gap_frames=90), seats=seats)
    long_df = t2s[t2s["long_gap"]]

    assert len(long_df) == 1
    assert long_df.iloc[0]["long_gap_status"] == "needs_visual_review"


def test_validation_distinguishes_long_gap_review_status(seats):
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = []
    for f in range(1, 41):
        rows.append(det_row(f, 15, s07))
    for f in range(242, 282):
        rows.append(det_row(f, 37, s07))
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg(reassociation_gap_frames=90))
    rep = validate_seat_assignment(a, seats, cfg(reassociation_gap_frames=90))
    assert rep["n_long_gap_fragments"] == 1
    assert rep["n_long_gap_auto_verified"] == 1
    assert rep["n_long_gap_needs_review"] == 0


# ---------------------------------------------------------------- validasi
def test_validation_passes_normal_case(seats):
    rows = []
    for i, s in enumerate(seats.itertuples(), start=1):
        rows += [det_row(f, i, s) for f in range(1, 41)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    rep = validate_seat_assignment(a, seats, cfg())
    assert rep["passed"]
    assert rep["n_assigned_students"] == 21
    assert rep["expected_students"] == 21


def test_validation_reports_fragments(seats):
    s07 = seats[seats.student_id == "S07"].iloc[0]
    rows = [det_row(f, 15 if f <= 50 else 37, s07) for f in range(1, 101)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    rep = validate_seat_assignment(a, seats, cfg())
    assert rep["n_fragmented_students"] == 1
    assert rep["n_track_fragments_merged"] == 1
    assert rep["n_unique_track_ids"] == 2


def test_validation_fails_on_empty():
    seats = make_seat_grid()
    rep = validate_seat_assignment(pd.DataFrame(), seats, cfg())
    assert not rep["passed"]
    assert rep["errors"]


# ---------------------------------------------------------------- seat map
def test_missing_seats_file_raises():
    with pytest.raises(IdentityConfigError, match="requires"):
        load_seat_map("tidak/ada/student_seats.csv", 21)


def test_duplicate_student_id_rejected(tmp_path):
    df = make_seat_grid()
    df.loc[1, "student_id"] = "S01"
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    with pytest.raises(IdentityConfigError, match="Duplicate student_id"):
        load_seat_map(str(p), 21)


def test_duplicate_seat_id_rejected(tmp_path):
    df = make_seat_grid()
    df.loc[1, "seat_id"] = "seat_01"
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    with pytest.raises(IdentityConfigError, match="Duplicate seat_id"):
        load_seat_map(str(p), 21)


def test_wrong_student_count_rejected(tmp_path):
    df = make_seat_grid(24)
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    with pytest.raises(IdentityConfigError, match="Expected 21"):
        load_seat_map(str(p), 21)


def test_missing_column_rejected(tmp_path):
    df = make_seat_grid().drop(columns=["roi_x2"])
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    with pytest.raises(IdentityConfigError, match="kolom wajib"):
        load_seat_map(str(p), 21)


def test_malformed_csv_rejected(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("student_id,seat_id,seat_x,seat_y,roi_x1,roi_y1,roi_x2,roi_y2\n"
                 "S01,seat_01,abc,def,0,0,10,10\n", encoding="utf-8")
    with pytest.raises(IdentityConfigError, match="non-numerik"):
        load_seat_map(str(p), 1)


def test_invalid_roi_rejected(tmp_path):
    df = make_seat_grid()
    df.loc[0, "roi_x2"] = df.loc[0, "roi_x1"] - 5
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    with pytest.raises(IdentityConfigError, match="ROI tidak valid"):
        load_seat_map(str(p), 21)


# ---------------------------------------------------------------- config
def test_spatial_weight_cannot_be_zero():
    with pytest.raises(IdentityConfigError, match="weight_spatial"):
        IdentityConfig(weight_spatial=0.0)


def test_unknown_config_key_rejected():
    with pytest.raises(IdentityConfigError, match="tidak dikenal"):
        IdentityConfig.from_dict({"typo_key": 1})


def test_smoothing_window_forced_odd():
    assert IdentityConfig(temporal_smoothing_window=14).temporal_smoothing_window == 15


def test_reid_weight_excluded_when_unavailable():
    c = IdentityConfig()
    ws, wt, wr = c.active_weights(reid_available=False)
    assert wr == 0.0 and abs(ws + wt - 1.0) < 1e-9


# ---------------------------------------------------------------- edge
def test_empty_detections_no_crash(seats):
    a = assign_track_to_seat(pd.DataFrame(), seats, cfg())
    assert a.empty
    t, s = merge_track_fragments(a, FPS, cfg())
    assert t.empty and s.empty


def test_single_student_only(seats):
    s01 = seats[seats.student_id == "S01"].iloc[0]
    rows = [det_row(f, 1, s01) for f in range(1, 41)]
    a = assign_track_to_seat(pd.DataFrame(rows), seats, cfg())
    _, audit = merge_track_fragments(a, FPS, cfg())
    assert len(audit) == 1
    rep = validate_seat_assignment(a, seats, cfg())
    assert rep["n_assigned_students"] == 1
    assert any("21" in w for w in rep["warnings"])   # diperingatkan, tidak crash


def test_zero_fps_rejected(seats):
    s01 = seats[seats.student_id == "S01"].iloc[0]
    a = assign_track_to_seat(pd.DataFrame([det_row(1, 1, s01)]), seats, cfg())
    with pytest.raises(ValueError, match="fps"):
        merge_track_fragments(a, 0.0, cfg())
