"""Recompute final AHP -> SAW/WP -> sensitivity -> Monte Carlo from an
already verified student-level activity matrix.

This tool does not run detection/tracking. It is intended for the final
research run after identity validation has passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.ahp import ahp_weights
from src.saw import saw_score, zero_diagnostics
from src.sensitivity import one_at_a_time
from src.monte_carlo import monte_carlo_sensitivity
from src.wp import compare_saw_wp, wp_score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--matrix", required=True, help="student_activity_matrix_duration.csv")
    ap.add_argument("--out", default="output_final_ahp")
    args = ap.parse_args()

    root = Path(args.config).resolve().parent
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    criteria = cfg["criteria"]
    orientation = cfg["orientation"]
    ahpcfg = cfg["ahp"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix, index_col=0)
    matrix.index.name = "student_id"
    matrix = matrix[criteria].astype(float)

    ahp = ahp_weights(
        np.asarray(ahpcfg["matrix"], dtype=float),
        criteria,
        source=ahpcfg.get("source", "predefined_model"),
        expert_count=int(ahpcfg.get("expert_count", 0)),
        aggregation_method=ahpcfg.get("aggregation_method"),
    )
    pd.DataFrame({"criterion": criteria, "weight": [ahp["weights"][c] for c in criteria]}).to_csv(
        out / "ahp_weights_final.csv", index=False
    )
    pd.DataFrame({
        "metric": ["lambda_max", "CI", "RI", "CR", "consistent", "ahp_source", "expert_count"],
        "value": [ahp["lambda_max"], ahp["CI"], ahp["RI"], ahp["CR"], ahp["consistent"], ahp["ahp_source"], ahp["expert_count"]],
    }).to_csv(out / "ahp_consistency_final.csv", index=False)
    pd.DataFrame(ahp["matrix"], index=criteria, columns=criteria).to_csv(out / "ahp_pairwise_matrix_final.csv")

    zero_diagnostics(matrix, orientation, float(cfg["saw"]["zero_epsilon"])).to_csv(
        out / "zero_handling_final.csv", index=False
    )

    zero_eps = float(cfg["saw"]["zero_epsilon"])
    zero_strategy = cfg["saw"]["zero_strategy"]
    saw = saw_score(matrix, ahp["weights"], orientation, zero_eps, zero_strategy)
    saw.to_csv(out / "ranking_saw_final.csv")

    wp = wp_score(matrix, ahp["weights"], orientation, zero_eps)
    wp.to_csv(out / "ranking_wp_final.csv")
    compare_saw_wp(matrix, ahp["weights"], orientation, zero_eps, zero_strategy).to_csv(
        out / "ahp_saw_vs_wp_final.csv", index=False
    )

    sens = one_at_a_time(
        matrix, ahp["weights"], orientation,
        perturbations=cfg["sensitivity"]["perturbations"],
        zero_epsilon=zero_eps, zero_strategy=zero_strategy,
    )
    sens.to_csv(out / "sensitivity_oat_final.csv", index=False)

    mc = monte_carlo_sensitivity(
        matrix, ahp["weights"], orientation,
        iterations=int(cfg["monte_carlo"]["iterations"]),
        seed=int(cfg["monte_carlo"]["seed"]),
        method=cfg["monte_carlo"]["method"],
        concentration=float(cfg["monte_carlo"]["concentration"]),
        zero_epsilon=zero_eps, zero_strategy=zero_strategy,
    )
    mc.to_csv(out / "sensitivity_montecarlo_final.csv", index=False)

    print("AHP FINAL")
    print(pd.DataFrame({"criterion": criteria, "weight": [ahp["weights"][c] for c in criteria]}).to_string(index=False))
    print(f"lambda_max={ahp['lambda_max']:.10f}; CI={ahp['CI']:.10f}; RI={ahp['RI']:.2f}; CR={ahp['CR']:.10f}; consistent={ahp['consistent']}")
    print("\nSAW RANKING")
    print(saw[["V", "rank"]].to_string())
    print(f"\nOutputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
