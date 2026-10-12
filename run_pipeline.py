"""
run_pipeline.py — orchestrator end-to-end:

  video -> YOLO11 -> BoT-SORT(+ReID opsional) -> FIXED-SEAT IDENTITY
        -> merge fragmen track -> agregasi temporal -> AHP -> SAW
        -> sensitivity -> laporan Excel

Cara pakai:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --skip-detect
    python run_pipeline.py --config config.yaml --identity-audit-only
    python run_pipeline.py --config config.yaml --temporal-both
    python run_pipeline.py --config config.yaml --sensitivity
    python run_pipeline.py --config config.yaml --monte-carlo
    python run_pipeline.py --config config.yaml --track-level   # DEBUG saja

ATURAN KESELAMATAN IDENTITAS: ranking student-level TIDAK PERNAH dihasilkan
bila validasi identitas gagal. Tidak ada fallback diam-diam ke track_id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.aggregate import (activity_transitions, build_activity_matrix,
                           build_both_representations)
from src.ahp import ahp_weights
from src.detect_track import run_detection_tracking, summarise_tracks
from src.identity import (IdentityConfig, IdentityConfigError, assign_track_to_seat,
                          build_student_level_detections, format_identity_audit,
                          fragment_warnings, load_seat_map, long_gap_audit,
                          merge_track_fragments, validate_seat_assignment)
from src.report import write_report
from src.saw import saw_score, zero_diagnostics
from src.sensitivity import one_at_a_time

BANNER = (
    "NOTE: BoT-SORT track IDs are trajectory identifiers, not absolute "
    "student identities."
)


def _cfg(cfg: dict, path: str, default=None):
    """Ambil nilai config bersarang dengan aman: _cfg(cfg,'identity.mode')."""
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def main(cfg_path: str, args: argparse.Namespace) -> int:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(_cfg(cfg, "output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = out_dir / "detections.csv"
    report_xlsx = out_dir / "yolo11_classroom_research_report.xlsx"

    criteria = _cfg(cfg, "criteria", [])
    orientation = _cfg(cfg, "orientation", {})
    fps_cfg = _cfg(cfg, "temporal.fps", "auto")
    if isinstance(fps_cfg, str) and fps_cfg.strip().lower() == "auto":
        from src.provenance import video_info
        vi = video_info(_cfg(cfg, "source.video_path"))
        actual_fps = vi.get("video_fps")
        if actual_fps in (None, "NOT_AVAILABLE"):
            print(
                "ERROR:\n`temporal.fps=auto` tetapi FPS video tidak dapat dibaca. "
                "Isi temporal.fps secara eksplisit atau periksa source.video_path.",
                file=sys.stderr,
            )
            return 2
        fps = float(actual_fps)
    else:
        fps = float(fps_cfg)
    min_track_frames = int(_cfg(cfg, "temporal.min_track_frames", 30))
    representation = _cfg(cfg, "temporal.representation", "duration")
    strict_classes = bool(_cfg(cfg, "temporal.strict_classes", True))
    zero_eps = float(_cfg(cfg, "saw.zero_epsilon", 1e-6))
    zero_strategy = _cfg(cfg, "saw.zero_strategy", "epsilon")

    print(BANNER)

    # ---------- Stage 1: detect + track ----------------------------------
    need_detect = not args.skip_detect and (
        not detections_csv.exists() or _cfg(cfg, "detection.force_detect", False)
    )
    if need_detect:
        print(f"[1] YOLO11 + BoT-SORT <- {_cfg(cfg,'source.video_path')}")
        run_detection_tracking(
            model_path=_cfg(cfg, "source.model_path"),
            video_path=_cfg(cfg, "source.video_path"),
            output_csv=str(detections_csv),
            tracker=_cfg(cfg, "tracking.tracker", "botsort.yaml"),
            conf=_cfg(cfg, "detection.conf", 0.25),
            iou=_cfg(cfg, "detection.iou", 0.5),
            imgsz=_cfg(cfg, "detection.imgsz", 640),
            device=str(_cfg(cfg, "detection.device", "0")),
            save_video=_cfg(cfg, "detection.save_video", False),
            with_reid=_cfg(cfg, "tracking.with_reid", False),
            reid_model=_cfg(cfg, "tracking.reid_model"),
            keep_untracked=_cfg(cfg, "detection.keep_untracked", False),
        )
    else:
        print(f"[1] Memakai deteksi yang sudah ada: {detections_csv}")

    if not detections_csv.exists():
        print(f"ERROR:\ndetections.csv tidak ditemukan di {detections_csv}.\n"
              f"Jalankan tanpa --skip-detect terlebih dahulu.", file=sys.stderr)
        return 2

    detections = pd.read_csv(detections_csv)
    track_summary = summarise_tracks(fps=fps, detections_df=detections)
    track_summary.to_csv(out_dir / "track_summary.csv", index=False)
    print(f"    Track summary -> {out_dir/'track_summary.csv'} "
          f"({len(track_summary)} track ID unik — BUKAN jumlah siswa)")

    # peringatan fps
    from src.provenance import check_fps_consistency
    warn = check_fps_consistency(fps, _cfg(cfg, "source.video_path"))
    if warn:
        print(warn)

    # ---------- Stage 1.5: IDENTITAS -------------------------------------
    mode = "track_level" if args.track_level else _cfg(cfg, "identity.mode", "student_level")
    identity_assignments = track_to_student = student_audit = None
    student_detections = None
    identity_report: dict = {}
    identity_ok = False

    if mode == "student_level":
        print("\n[2] Fixed-seat identity association")
        id_cfg_raw = dict(_cfg(cfg, "identity", {}) or {})
        seats_csv = id_cfg_raw.pop("seats_csv", None)
        id_cfg_raw.pop("mode", None)
        try:
            icfg = IdentityConfig.from_dict({**id_cfg_raw, "mode": "student_level"})
            seats = load_seat_map(seats_csv, icfg.expected_students)
        except IdentityConfigError as e:
            # Pesan bersih, bukan traceback: konfigurasi identitas yang rusak
            # TIDAK PERNAH dipulihkan diam-diam ke track_id.
            print(f"\n{e}", file=sys.stderr)
            print("\nStudent-level ranking blocked because identity validation "
                  "has not passed.", file=sys.stderr)
            return 2

        identity_assignments = assign_track_to_seat(detections, seats, icfg)
        identity_assignments.to_csv(out_dir / "identity_assignments.csv", index=False)

        track_to_student, student_audit = merge_track_fragments(
            identity_assignments, fps, icfg,
            total_video_frames=int(detections["frame"].max()) + 1 if not detections.empty else 0,
            seats=seats)
        track_to_student.to_csv(out_dir / "track_to_student_summary.csv", index=False)
        student_audit.to_csv(out_dir / "student_identity_audit.csv", index=False)

        # Compact long-gap audit. This is kept as CSV (not embedded into the
        # large Excel workbook) so the raw evidence remains inspectable without
        # inflating the final report.
        long_gap_df = long_gap_audit(track_to_student, icfg)
        long_gap_df.to_csv(out_dir / "long_gap_audit.csv", index=False)

        identity_report = validate_seat_assignment(
            identity_assignments, seats, icfg,
            total_video_frames=int(detections["frame"].max()) + 1 if not detections.empty else 0)
        print(format_identity_audit(identity_report))
        for m in fragment_warnings(track_to_student, icfg)[:20]:
            print(m)
        print(
            f"    Long-gap audit -> {out_dir / 'long_gap_audit.csv'} "
            f"({identity_report.get('n_long_gap_fragments', 0)} total, "
            f"{identity_report.get('n_long_gap_auto_verified', 0)} strong fixed-seat, "
            f"{identity_report.get('n_long_gap_needs_review', 0)} visual review)"
        )

        identity_ok = bool(identity_report.get("passed"))
        student_detections = build_student_level_detections(identity_assignments)

        if _cfg(cfg, "experiments.identity_audit_video", False):
            try:
                from src.visual_audit import render_identity_audit_video
                p = render_identity_audit_video(
                    _cfg(cfg, "source.video_path"), identity_assignments,
                    str(out_dir / "identity_audit.mp4"), seats=seats)
                print(f"    Video audit identitas -> {p}")
            except Exception as e:
                print(f"WARNING: video audit identitas dilewati: {e}")

        if args.identity_audit_only:
            print("\nSelesai (--identity-audit-only): tidak ada ranking yang dihitung.")
            return 0 if identity_ok else 1

        if not identity_ok:
            print("\nStudent-level ranking blocked because identity validation "
                  "has not passed.", file=sys.stderr)
            print("Jalankan --track-level untuk debugging, atau perbaiki "
                  "data/student_seats.csv.", file=sys.stderr)
            write_report(
                output_xlsx=str(report_xlsx), config_snapshot=cfg,
                track_summary=track_summary,
                identity_assignments=identity_assignments,
                track_to_student=track_to_student,
                student_identity_audit=student_audit,
            )
            print(f"Laporan audit (tanpa ranking) -> {report_xlsx}")
            return 1
    else:
        print("\n[2] MODE track_level (DEBUG). Entitas adalah track_id, "
              "BUKAN siswa. Hasilnya tidak boleh dilaporkan sebagai siswa.")

    # ---------- Stage 3: agregasi temporal --------------------------------
    entity_col = "student_id" if mode == "student_level" else "track_id"
    agg_source = student_detections if mode == "student_level" else detections
    print(f"\n[3] Agregasi temporal ({representation}), fps={fps}, "
          f"entitas={entity_col}")

    matrices = build_both_representations(
        agg_source, fps=fps, entity_column=entity_col, classes=criteria,
        min_track_frames=min_track_frames, strict_classes=strict_classes,
    ) if (args.temporal_both or _cfg(cfg, "experiments.temporal_comparison", False)) else None

    matrix = (matrices[representation] if matrices else build_activity_matrix(
        detections_csv="", fps=fps, output_csv="", min_track_frames=min_track_frames,
        representation=representation, entity_column=entity_col, classes=criteria,
        strict_classes=strict_classes, detections_df=agg_source))

    # Nama berkas mengikuti JENIS ENTITAS. Mode track_level TIDAK PERNAH menulis
    # ke berkas ber-awalan `student_`, supaya hasil debugging track tidak dapat
    # tertukar dengan hasil tingkat-siswa.
    prefix = "student" if mode == "student_level" else "track"
    if matrices:
        matrices["duration"].to_csv(out_dir / f"{prefix}_activity_matrix_duration.csv")
        matrices["frequency"].to_csv(out_dir / f"{prefix}_activity_matrix_frequency.csv")
    matrix.to_csv(out_dir / f"{prefix}_activity_matrix.csv")
    print(matrix.round(2).to_string())

    transitions = activity_transitions(agg_source, entity_column=entity_col)
    transitions.to_csv(out_dir / f"{prefix}_activity_transitions.csv", index=False)

    # ---------- Stage 4: AHP ----------------------------------------------
    print("\n[4] AHP — bobot kriteria")
    ahp = ahp_weights(
        np.array(_cfg(cfg, "ahp.matrix")), criteria,
        source=_cfg(cfg, "ahp.source", "predefined_model"),
        expert_count=int(_cfg(cfg, "ahp.expert_count", 0)),
        aggregation_method=_cfg(cfg, "ahp.aggregation_method"),
    )
    for c, w in ahp["weights"].items():
        print(f"    w[{c}] = {w:.4f}")
    print(f"    lambda_max={ahp['lambda_max']:.4f} CI={ahp['CI']:.4f} "
          f"RI={ahp['RI']:.2f} CR={ahp['CR']:.4f} -> consistent={ahp['consistent']}")
    if ahp["ahp_source"] == "predefined_model" and ahp["expert_count"] == 0:
        print("    CATATAN: matriks AHP adalah CONTOH (bukan elisitasi pakar).")
    if not ahp["consistent"]:
        print("WARNING:\nAHP CR > 0.10.")
        if _cfg(cfg, "ahp.block_on_inconsistent", True) and not args.force_inconsistent:
            print("Ranking final diblokir karena CR > 0.10. Revisi matriks "
                  "pairwise, atau jalankan ulang dengan --force-inconsistent.",
                  file=sys.stderr)
            write_report(output_xlsx=str(report_xlsx), config_snapshot=cfg,
                         ahp_result=ahp, track_summary=track_summary,
                         identity_assignments=identity_assignments,
                         track_to_student=track_to_student,
                         student_identity_audit=student_audit,
                         activity_duration=matrix)
            return 1

    # ---------- Stage 5: SAW ----------------------------------------------
    print("\n[5] SAW — skor & ranking")
    zero_diag = zero_diagnostics(matrix, orientation, zero_eps)
    if bool(zero_diag["discriminative_collapse"].any()):
        hit = zero_diag.loc[zero_diag["discriminative_collapse"], "criterion"].tolist()
        print(f"WARNING:\nKriteria cost {hit} memuat nilai nol. Dengan "
              f"zero_strategy='epsilon', daya pembeda kriteria tersebut runtuh "
              f"(entitas lain -> ~0). Laporkan hal ini di manuskrip.")
    saw = saw_score(matrix, ahp["weights"], orientation, zero_eps, zero_strategy)
    print(saw.round(4).to_string())

    # ---------- Stage 6: sensitivity + eksperimen --------------------------
    print("\n[6] Sensitivity (OAT)")
    sens = one_at_a_time(
        matrix, ahp["weights"], orientation,
        perturbations=_cfg(cfg, "sensitivity.perturbations",
                           (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)),
        zero_epsilon=zero_eps, zero_strategy=zero_strategy)
    sens.to_csv(out_dir / "sensitivity_oat.csv", index=False)
    print(sens.round(4).to_string(index=False))

    mc_df = temporal_cmp = wp_df = ablation_df = err_df = None

    if args.monte_carlo or _cfg(cfg, "monte_carlo.enabled", False):
        print("\n[7] Monte Carlo sensitivity")
        from src.monte_carlo import monte_carlo_sensitivity
        mc_df = monte_carlo_sensitivity(
            matrix, ahp["weights"], orientation,
            iterations=int(_cfg(cfg, "monte_carlo.iterations", 5000)),
            seed=int(_cfg(cfg, "monte_carlo.seed", 42)),
            method=_cfg(cfg, "monte_carlo.method", "dirichlet_around_base"),
            concentration=float(_cfg(cfg, "monte_carlo.concentration", 50.0)),
            zero_epsilon=zero_eps, zero_strategy=zero_strategy)
        mc_df.to_csv(out_dir / "sensitivity_montecarlo.csv", index=False)
        print(mc_df.head(10).round(4).to_string(index=False))

    if matrices and _cfg(cfg, "experiments.temporal_comparison", False):
        from src.temporal_analysis import compare_representations
        temporal_cmp = compare_representations(
            matrices, ahp["weights"], orientation, zero_eps, zero_strategy)
        temporal_cmp.to_csv(out_dir / "temporal_comparison.csv", index=False)

    if _cfg(cfg, "experiments.saw_vs_wp", False):
        from src.wp import compare_saw_wp
        wp_df = compare_saw_wp(matrix, ahp["weights"], orientation,
                               zero_eps, zero_strategy)
        wp_df.to_csv(out_dir / "ahp_saw_vs_wp.csv", index=False)

    if _cfg(cfg, "experiments.ablation", False):
        from src.ablation import run_ablation
        ablation_df = run_ablation(
            detections, fps, criteria, min_track_frames,
            student_detections=student_detections,
            identity_report=identity_report, representation=representation)
        ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)

    if _cfg(cfg, "experiments.error_propagation", False) and student_audit is not None:
        from src.error_propagation import build_propagation_table, qualitative_notes
        err_df = build_propagation_table(student_audit, matrix, saw)
        if err_df is not None and not err_df.empty:
            err_df = pd.concat([err_df, qualitative_notes()], ignore_index=True)
            err_df.to_csv(out_dir / "error_propagation.csv", index=False)

    # detector evaluation (opsional, butuh data.yaml)
    det_metrics = None
    if _cfg(cfg, "experiments.detector_evaluation", False):
        from src.evaluate_detector import (DetectorEvalUnavailable,
                                           evaluate_detector, not_available_frame)
        try:
            res = evaluate_detector(
                model_path=_cfg(cfg, "source.model_path"),
                data_yaml=_cfg(cfg, "source.data_yaml"),
                split=_cfg(cfg, "detector_evaluation.split", "test"),
                imgsz=_cfg(cfg, "detection.imgsz", 640),
                device=str(_cfg(cfg, "detection.device", "0")),
                batch=int(_cfg(cfg, "detector_evaluation.batch", 16)),
                output_json=str(out_dir / "detection_metrics.json"),
                output_csv=str(out_dir / "detection_metrics.csv"))
            det_metrics = pd.DataFrame(
                [{"scope": "overall", "class": "ALL", **res["overall"]}]
                + [{"scope": "per_class", **pc} for pc in res["per_class"]])
        except DetectorEvalUnavailable as e:
            print(e)
            det_metrics = not_available_frame([str(e)])

    from src.mot_eval import mot_status
    mot_df = pd.DataFrame([mot_status(_cfg(cfg, "mot_evaluation.ground_truth_csv"))])

    # ---------- Laporan ----------------------------------------------------
    from src.provenance import build_provenance
    write_report(
        output_xlsx=str(report_xlsx), config_snapshot=cfg, ahp_result=ahp,
        saw_result=saw, sensitivity_df=sens,
        provenance_df=build_provenance(cfg, {
            "identity_mode": mode,
            "identity_validation_passed": identity_ok if mode == "student_level" else "NOT_APPLICABLE",
            "n_raw_track_ids": len(track_summary),
            "n_ranked_entities": len(matrix),
        }),
        detection_metrics=det_metrics, track_summary=track_summary,
        identity_assignments=identity_assignments, track_to_student=track_to_student,
        student_identity_audit=student_audit,
        activity_duration=matrices["duration"] if matrices else matrix,
        activity_frequency=matrices["frequency"] if matrices else None,
        activity_transitions_df=transitions, zero_handling=zero_diag,
        ablation_df=ablation_df, temporal_comparison=temporal_cmp,
        monte_carlo_df=mc_df, saw_vs_wp=wp_df, error_propagation_df=err_df,
        mot_evaluation=mot_df,
    )
    print(f"\nOK  Laporan Excel: {report_xlsx}")
    if mode == "track_level":
        print("INGAT: hasil ini berbasis track_id dan TIDAK merepresentasikan siswa.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLO11 classroom analytics pipeline")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-detect", action="store_true",
                   help="Lewati detect+track; gunakan detections.csv yang ada")
    p.add_argument("--identity-audit-only", action="store_true",
                   help="Hanya jalankan audit identitas, tanpa ranking")
    p.add_argument("--temporal-both", action="store_true",
                   help="Hitung representasi duration DAN frequency")
    p.add_argument("--sensitivity", action="store_true",
                   help="(selalu aktif) analisis sensitivity OAT")
    p.add_argument("--monte-carlo", action="store_true",
                   help="Paksa jalankan Monte Carlo meski dinonaktifkan di config")
    p.add_argument("--track-level", action="store_true",
                   help="DEBUG: agregasi pada track_id, bukan siswa")
    p.add_argument("--force-inconsistent", action="store_true",
                   help="Lanjutkan meski CR > 0.10 (harus disebut di manuskrip)")
    a = p.parse_args()
    try:
        sys.exit(main(a.config, a))
    except (IdentityConfigError, FileNotFoundError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pesan ringkas; traceback penuh dengan --debug
        if "--debug" in sys.argv:
            raise
        print(f"\nERROR:\n{type(e).__name__}: {e}", file=sys.stderr)
        print("Jalankan ulang dengan --debug untuk traceback lengkap.", file=sys.stderr)
        sys.exit(2)
