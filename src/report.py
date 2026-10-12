"""
Stage 14 — Laporan Excel penelitian multi-sheet.

19 sheet target. Eksperimen yang TIDAK dijalankan ditulis sebagai
NOT_RUN / NOT_AVAILABLE beserta alasannya — TIDAK PERNAH diisi angka dummy.

Bila laporan dibangun dari data sintetik/mock, seluruh sheet diberi penanda
TEST ONLY melalui argumen `test_only=True` agar tidak pernah tertukar dengan
hasil penelitian sebenarnya.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

SHEET_ORDER = [
    "Config", "Provenance", "Detection_Metrics", "Track_Summary",
    "Identity_Assignments", "Track_to_Student", "Student_Identity_Audit",
    "Activity_Duration", "Activity_Frequency", "Activity_Transitions",
    "AHP_Weights", "AHP_Consistency", "SAW_Normalized", "SAW_Ranking",
    "Zero_Handling", "Ablation", "Temporal_Comparison", "Sensitivity_OAT",
    "Sensitivity_MonteCarlo", "AHP_SAW_vs_WP", "Error_Propagation", "MOT_Evaluation",
]

MAX_ROWS_PER_SHEET = 200_000   # jaga ukuran berkas tetap wajar


def not_run(reason: str, experiment: str = "") -> pd.DataFrame:
    """Bingkai penanda baku untuk eksperimen yang tidak dijalankan."""
    return pd.DataFrame([{
        "experiment": experiment or "(tidak ditentukan)",
        "status": "NOT_RUN",
        "value": "NOT_AVAILABLE",
        "reason": reason,
    }])


def _flatten_config(cfg: dict, prefix: str = "") -> list:
    out: list = []
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_config(v, key))
        else:
            out.append((key, str(v)))
    return out


def _autoformat(ws, df: pd.DataFrame, index: bool, decimals: int = 4) -> None:
    """Freeze panes, autofilter, lebar kolom, dan presisi desimal konsisten."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    n_idx = df.index.nlevels if index else 0
    n_cols = n_idx + len(df.columns)
    if n_cols == 0 or ws.max_row < 1:
        return

    header_fill = PatternFill("solid", fgColor="DDE5F0")
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.freeze_panes = ws.cell(row=2, column=n_idx + 1)
    if ws.max_row > 1:
        ws.auto_filter.ref = (
            f"A1:{get_column_letter(n_cols)}{min(ws.max_row, MAX_ROWS_PER_SHEET)}"
        )

    fmt = "0." + "0" * decimals if decimals > 0 else "0"
    headers = ([str(x) for x in (df.index.names if index else [])] +
               [str(c) for c in df.columns])
    for i, name in enumerate(headers, start=1):
        longest = len(str(name))
        col_vals = (df.index.get_level_values(i - 1) if index and i <= n_idx
                    else df[df.columns[i - 1 - n_idx]])
        try:
            sample = col_vals.astype(str).head(200)
            longest = max(longest, int(sample.str.len().max() or 0))
        except Exception:
            pass
        ws.column_dimensions[get_column_letter(i)].width = min(max(longest + 3, 10), 52)

        if i > n_idx:
            series = df[df.columns[i - 1 - n_idx]]
            if pd.api.types.is_float_dtype(series):
                for r in range(2, min(ws.max_row, MAX_ROWS_PER_SHEET) + 1):
                    ws.cell(row=r, column=i).number_format = fmt


def write_report(
    output_xlsx: str,
    activity_matrix: Optional[pd.DataFrame] = None,
    ahp_result: Optional[dict] = None,
    saw_result: Optional[pd.DataFrame] = None,
    sensitivity_df: Optional[pd.DataFrame] = None,
    config_snapshot: Optional[dict] = None,
    *,
    provenance_df: Optional[pd.DataFrame] = None,
    detection_metrics: Optional[pd.DataFrame] = None,
    track_summary: Optional[pd.DataFrame] = None,
    identity_assignments: Optional[pd.DataFrame] = None,
    track_to_student: Optional[pd.DataFrame] = None,
    student_identity_audit: Optional[pd.DataFrame] = None,
    activity_duration: Optional[pd.DataFrame] = None,
    activity_frequency: Optional[pd.DataFrame] = None,
    activity_transitions_df: Optional[pd.DataFrame] = None,
    zero_handling: Optional[pd.DataFrame] = None,
    ablation_df: Optional[pd.DataFrame] = None,
    temporal_comparison: Optional[pd.DataFrame] = None,
    monte_carlo_df: Optional[pd.DataFrame] = None,
    saw_vs_wp: Optional[pd.DataFrame] = None,
    error_propagation_df: Optional[pd.DataFrame] = None,
    mot_evaluation: Optional[pd.DataFrame] = None,
    test_only: bool = False,
) -> str:
    """
    Tulis laporan Excel. Argumen posisional lama dipertahankan agar pemanggilan
    versi baseline tetap berfungsi.
    """
    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    sheets: dict = {}
    indexed: set = set()

    # 1. Config
    if config_snapshot is not None:
        sheets["Config"] = pd.DataFrame(
            _flatten_config(config_snapshot), columns=["key", "value"]
        )
    else:
        sheets["Config"] = not_run("config snapshot tidak diberikan", "Config")

    # 2. Provenance
    sheets["Provenance"] = (
        provenance_df if provenance_df is not None and not provenance_df.empty
        else not_run("provenance tidak dikumpulkan", "Provenance")
    )

    # 3. Detection metrics
    sheets["Detection_Metrics"] = (
        detection_metrics if detection_metrics is not None and not detection_metrics.empty
        else not_run(
            "model.val() tidak dijalankan: bobot model dan/atau data.yaml "
            "dengan split test belum tersedia.", "Detection_Metrics")
    )

    # 4-7. tracking + identitas
    def _or_na(df, name, reason):
        return df if df is not None and not df.empty else not_run(reason, name)

    sheets["Track_Summary"] = _or_na(
        track_summary, "Track_Summary", "tidak ada deteksi/track.")
    sheets["Identity_Assignments"] = _or_na(
        identity_assignments, "Identity_Assignments",
        "lapisan identitas tidak dijalankan (mode track_level atau "
        "student_seats.csv belum tersedia).")
    sheets["Track_to_Student"] = _or_na(
        track_to_student, "Track_to_Student", "asosiasi identitas tidak tersedia.")
    sheets["Student_Identity_Audit"] = _or_na(
        student_identity_audit, "Student_Identity_Audit",
        "audit identitas tidak tersedia.")

    # 8-10. temporal
    dur = activity_duration if activity_duration is not None else activity_matrix
    if dur is not None and not dur.empty:
        sheets["Activity_Duration"] = dur; indexed.add("Activity_Duration")
    else:
        sheets["Activity_Duration"] = not_run(
            "matriks durasi tidak tersedia.", "Activity_Duration")
    if activity_frequency is not None and not activity_frequency.empty:
        sheets["Activity_Frequency"] = activity_frequency
        indexed.add("Activity_Frequency")
    else:
        sheets["Activity_Frequency"] = not_run(
            "representasi frequency tidak dihitung "
            "(aktifkan temporal.representations).", "Activity_Frequency")
    sheets["Activity_Transitions"] = _or_na(
        activity_transitions_df, "Activity_Transitions",
        "statistik transisi tidak dihitung.")

    # 11-12. AHP
    if ahp_result:
        sheets["AHP_Weights"] = pd.DataFrame({
            "criterion": list(ahp_result["weights"].keys()),
            "weight": list(ahp_result["weights"].values()),
        })
        sheets["AHP_Consistency"] = pd.DataFrame({
            "metric": ["lambda_max", "CI", "RI", "CR", "consistent",
                       "ahp_source", "expert_count", "aggregation_method",
                       "n_criteria"],
            "value": [
                ahp_result.get("lambda_max"), ahp_result.get("CI"),
                ahp_result.get("RI"), ahp_result.get("CR"),
                ahp_result.get("consistent"),
                ahp_result.get("ahp_source", "NOT_AVAILABLE"),
                ahp_result.get("expert_count", "NOT_AVAILABLE"),
                ahp_result.get("aggregation_method") or "NOT_APPLICABLE",
                ahp_result.get("n_criteria", "NOT_AVAILABLE"),
            ],
        })
    else:
        sheets["AHP_Weights"] = not_run("AHP tidak dijalankan.", "AHP_Weights")
        sheets["AHP_Consistency"] = not_run("AHP tidak dijalankan.", "AHP_Consistency")

    # 13-14. SAW
    if saw_result is not None and not saw_result.empty:
        sheets["SAW_Normalized"] = saw_result; indexed.add("SAW_Normalized")
        rank_cols = [c for c in ("V", "rank") if c in saw_result.columns]
        ranking = saw_result[rank_cols].copy()
        ranking.index.name = saw_result.index.name or "entity"
        sheets["SAW_Ranking"] = ranking.sort_values("rank") if "rank" in rank_cols else ranking
        indexed.add("SAW_Ranking")
    else:
        sheets["SAW_Normalized"] = not_run(
            "SAW tidak dijalankan atau ranking student-level diblokir oleh "
            "validasi identitas.", "SAW_Normalized")
        sheets["SAW_Ranking"] = not_run(
            "ranking tidak dihasilkan.", "SAW_Ranking")

    sheets["Zero_Handling"] = _or_na(
        zero_handling, "Zero_Handling", "diagnostik zero-handling tidak dihitung.")

    # 16-22. eksperimen
    sheets["Ablation"] = _or_na(
        ablation_df, "Ablation", "ablation tidak diaktifkan (experiments.ablation).")
    sheets["Temporal_Comparison"] = _or_na(
        temporal_comparison, "Temporal_Comparison",
        "perbandingan frequency-vs-duration tidak diaktifkan.")
    sheets["Sensitivity_OAT"] = _or_na(
        sensitivity_df, "Sensitivity_OAT", "sensitivity OAT tidak dijalankan.")
    sheets["Sensitivity_MonteCarlo"] = _or_na(
        monte_carlo_df, "Sensitivity_MonteCarlo",
        "Monte Carlo dinonaktifkan (monte_carlo.enabled=false).")
    sheets["AHP_SAW_vs_WP"] = _or_na(
        saw_vs_wp, "AHP_SAW_vs_WP", "pembanding WP tidak diaktifkan.")
    sheets["Error_Propagation"] = _or_na(
        error_propagation_df, "Error_Propagation",
        "analisis propagasi error tidak diaktifkan.")
    sheets["MOT_Evaluation"] = _or_na(
        mot_evaluation, "MOT_Evaluation",
        "WARNING: Formal MOT metrics unavailable because trajectory ground "
        "truth is absent.")

    # ---- tulis ----------------------------------------------------------
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        if test_only:
            pd.DataFrame([{
                "NOTICE": "TEST ONLY — SYNTHETIC/MOCK DATA",
                "detail": ("Laporan ini dibuat dari data sintetik untuk "
                           "verifikasi perangkat lunak. Angka di dalamnya BUKAN "
                           "hasil penelitian dan tidak boleh dikutip."),
            }]).to_excel(writer, sheet_name="TEST_ONLY_NOTICE", index=False)

        for name in SHEET_ORDER:
            df = sheets.get(name)
            if df is None:
                continue
            use_index = name in indexed
            if len(df) > MAX_ROWS_PER_SHEET:
                df = df.head(MAX_ROWS_PER_SHEET)
            title = f"{name}_TESTONLY"[:31] if test_only else name[:31]
            df.to_excel(writer, sheet_name=title, index=use_index)
            _autoformat(writer.sheets[title], df, use_index)

    return output_xlsx
