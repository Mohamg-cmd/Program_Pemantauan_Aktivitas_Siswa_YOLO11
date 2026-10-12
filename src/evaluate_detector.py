"""
Stage 11 — Evaluasi detector memakai Ultralytics `model.val()` yang SEBENARNYA.

Modul ini TIDAK PERNAH mengarang precision/recall/mAP. Bila `data.yaml`
atau bobot model tidak tersedia, fungsi mengembalikan status NOT_AVAILABLE
beserta alasannya, dan sheet Excel terkait ditandai demikian.

Perintah:
    python -m src.evaluate_detector --config config.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


class DetectorEvalUnavailable(RuntimeError):
    """Evaluasi detector tidak dapat dijalankan (input belum tersedia)."""


def evaluation_status(model_path: Optional[str], data_yaml: Optional[str]) -> dict:
    reasons = []
    if not model_path or not Path(model_path).exists():
        reasons.append(f"model weights tidak ditemukan: {model_path or 'NOT_SET'}")
    if not data_yaml or not Path(data_yaml).exists():
        reasons.append(f"data.yaml tidak ditemukan: {data_yaml or 'NOT_SET'}")
    return {
        "available": not reasons,
        "reasons": reasons,
        "model_path": model_path or "NOT_AVAILABLE",
        "data_yaml": data_yaml or "NOT_AVAILABLE",
    }


def evaluate_detector(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    device: str = "0",
    batch: int = 16,
    conf: Optional[float] = None,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> dict:
    """
    Jalankan `model.val()` nyata dan kembalikan metrik keseluruhan + per kelas.

    Returns dict dengan kunci: status, overall{precision, recall, mAP50,
    mAP50-95}, per_class[...], plus metadata run.
    """
    st = evaluation_status(model_path, data_yaml)
    if not st["available"]:
        raise DetectorEvalUnavailable(
            "ERROR:\nEvaluasi detector tidak dapat dijalankan.\n  - "
            + "\n  - ".join(st["reasons"])
            + "\nSediakan bobot model dan data.yaml (dengan split test) lalu ulangi."
        )

    from ultralytics import YOLO  # import ditunda agar tes tanpa GPU tetap jalan

    model = YOLO(model_path)
    kwargs = dict(data=data_yaml, split=split, imgsz=imgsz, device=device, batch=batch)
    if conf is not None:
        kwargs["conf"] = conf
    m = model.val(**kwargs)

    box = m.box
    names = getattr(m, "names", None) or getattr(model, "names", {})
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }

    per_class = []
    class_indices = list(getattr(box, "ap_class_index", []) or [])
    for i, ci in enumerate(class_indices):
        try:
            p, r, ap50, ap = box.class_result(i)
        except Exception:
            continue
        per_class.append({
            "class": names.get(int(ci), str(ci)) if isinstance(names, dict) else str(ci),
            "precision": float(p), "recall": float(r),
            "mAP50": float(ap50), "mAP50-95": float(ap),
            "support": "NOT_AVAILABLE",   # tidak diekspos seragam oleh Ultralytics
        })

    result = {
        "status": "OK",
        "overall": overall,
        "per_class": per_class,
        "model_path": model_path, "data_yaml": data_yaml, "split": split,
        "imgsz": imgsz, "device": str(device), "batch": batch,
        "conf": conf if conf is not None else "ULTRALYTICS_DEFAULT",
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if output_csv:
        rows = [{"scope": "overall", "class": "ALL", **overall}]
        rows += [{"scope": "per_class", **pc} for pc in per_class]
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    return result


def not_available_frame(reasons: list) -> pd.DataFrame:
    """Bingkai penanda untuk Excel bila evaluasi tidak dijalankan."""
    return pd.DataFrame([{
        "scope": "overall", "class": "ALL",
        "precision": "NOT_AVAILABLE", "recall": "NOT_AVAILABLE",
        "mAP50": "NOT_AVAILABLE", "mAP50-95": "NOT_AVAILABLE",
        "status": "NOT_RUN", "reason": "; ".join(reasons) or "tidak dijalankan",
    }])


if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description="Evaluasi detector YOLO11 (model.val)")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    src_cfg, det = cfg.get("source", {}), cfg.get("detection", {})
    out = Path(cfg.get("output_dir", "output"))
    try:
        res = evaluate_detector(
            model_path=src_cfg.get("model_path"), data_yaml=src_cfg.get("data_yaml"),
            split=cfg.get("detector_evaluation", {}).get("split", "test"),
            imgsz=det.get("imgsz", 640), device=str(det.get("device", "0")),
            batch=cfg.get("detector_evaluation", {}).get("batch", 16),
            output_json=str(out / "detection_metrics.json"),
            output_csv=str(out / "detection_metrics.csv"),
        )
        print(json.dumps(res["overall"], indent=2))
    except DetectorEvalUnavailable as e:
        print(e)
