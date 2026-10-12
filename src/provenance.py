"""
Stage 21 — Reproducibility / provenance.

Mengumpulkan kondisi lingkungan dan konfigurasi eksekusi. Informasi yang
TIDAK dapat dibaca dari environment dilaporkan sebagai NOT_AVAILABLE —
tidak pernah ditebak atau dikarang.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

NA = "NOT_AVAILABLE"


def _safe(fn, default: str = NA):
    try:
        v = fn()
        return default if v in (None, "") else v
    except Exception:
        return default


def _git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else NA
    except Exception:
        return NA


def environment_info() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None) or NA
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else NA
        )
    except Exception:
        info["torch_version"] = NA
        info["cuda_available"] = NA
        info["cuda_version"] = NA
        info["gpu"] = NA
    for mod, key in (("ultralytics", "ultralytics_version"),
                     ("pandas", "pandas_version"),
                     ("numpy", "numpy_version"),
                     ("scipy", "scipy_version")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            info[key] = NA
    return info


def video_info(video_path: Optional[str]) -> dict:
    """Baca FPS/resolusi/jumlah frame dari video bila OpenCV tersedia."""
    out = {"video_path": video_path or NA, "video_fps": NA,
           "video_frame_count": NA, "video_resolution": NA}
    if not video_path or not Path(video_path).exists():
        return out
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            out["video_fps"] = round(float(fps), 6) if fps and fps > 0 else NA
            out["video_frame_count"] = int(n) if n and n > 0 else NA
            out["video_resolution"] = f"{int(w)}x{int(h)}" if w and h else NA
        cap.release()
    except Exception:
        pass
    return out


def build_provenance(cfg: dict, extra: Optional[dict] = None) -> pd.DataFrame:
    """Tabel provenance dua kolom (key, value) untuk sheet Excel."""
    src = cfg.get("source", {})
    rec: dict = {}
    rec.update(environment_info())
    rec.update(video_info(src.get("video_path")))
    rec["model_path"] = src.get("model_path", NA)
    rec["data_yaml"] = src.get("data_yaml") or NA

    det = cfg.get("detection", {})
    rec["detection_conf"] = det.get("conf", NA)
    rec["detection_iou"] = det.get("iou", NA)
    rec["detection_imgsz"] = det.get("imgsz", NA)
    rec["detection_device"] = det.get("device", NA)

    trk = cfg.get("tracking", {})
    rec["tracker_config"] = trk.get("tracker", NA)
    rec["tracker_with_reid"] = trk.get("with_reid", NA)

    tmp = cfg.get("temporal", {})
    rec["configured_fps"] = tmp.get("fps", NA)
    rec["min_track_frames"] = tmp.get("min_track_frames", NA)

    for k, v in (cfg.get("identity", {}) or {}).items():
        rec[f"identity.{k}"] = v

    ahp = cfg.get("ahp", {})
    rec["ahp_source"] = ahp.get("source", NA)
    rec["ahp_expert_count"] = ahp.get("expert_count", NA)
    rec["ahp_matrix"] = str(ahp.get("matrix", NA))

    rec["saw_zero_epsilon"] = cfg.get("saw", {}).get("zero_epsilon", NA)
    rec["saw_zero_strategy"] = cfg.get("saw", {}).get("zero_strategy", NA)
    rec["sensitivity_perturbations"] = str(
        cfg.get("sensitivity", {}).get("perturbations", NA)
    )
    mc = cfg.get("monte_carlo", {})
    rec["monte_carlo_enabled"] = mc.get("enabled", NA)
    rec["monte_carlo_iterations"] = mc.get("iterations", NA)
    rec["monte_carlo_seed"] = mc.get("seed", NA)

    if extra:
        rec.update(extra)

    return pd.DataFrame(
        [{"key": k, "value": NA if v is None else str(v)} for k, v in rec.items()]
    )


def check_fps_consistency(configured_fps: float, video_path: Optional[str]) -> Optional[str]:
    """
    Peringatkan bila fps config berbeda dari fps video sebenarnya.
    Ketidakcocokan fps membuat SELURUH konversi durasi salah secara sistematis.
    """
    vi = video_info(video_path)
    actual = vi.get("video_fps")
    if actual in (NA, None):
        return None
    if abs(float(actual) - float(configured_fps)) > 0.01:
        return (
            f"WARNING:\nfps pada config ({configured_fps}) berbeda dari fps video "
            f"sebenarnya ({actual}). Seluruh nilai durasi akan bias. "
            f"Samakan temporal.fps dengan fps video."
        )
    return None
