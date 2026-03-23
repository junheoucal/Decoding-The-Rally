"""
Build vren_edgelist.csv: edges (hitter_location -> hit_land_location) with
in-system, out-of-system, and overall success rates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "Raw Data" / "dataset_full.csv"
OUT_PATH = SCRIPT_DIR / "vren_edgelist.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Rows that define an edge need both endpoints
    loc = df["hitter_location"].notna() & df["hit_land_location"].notna()
    edge_df = df.loc[loc].copy()

    team = edge_df["team"].astype(str).str.strip()
    winning = edge_df["winning_team"].astype(str).str.strip()
    win_reason = edge_df["win_reason"].fillna("").astype(str).str.strip()
    success = (winning == team) & (win_reason != "")

    set_type = edge_df["set_type"]
    is_in = set_type == "in"
    is_out = set_type == "out"
    set_nonempty = set_type.notna() & (set_type.astype(str).str.strip() != "")

    tmp = edge_df.copy()
    tmp["_in"] = is_in.astype(int)
    tmp["_out"] = is_out.astype(int)
    tmp["_succ_in"] = (success & is_in).astype(int)
    tmp["_succ_out"] = (success & is_out).astype(int)
    tmp["_succ_any"] = (success & set_nonempty).astype(int)
    tmp["_set_nonempty"] = set_nonempty.astype(int)

    out = (
        tmp.groupby(["hitter_location", "hit_land_location"], as_index=False)
        .agg(
            in_denom=("_in", "sum"),
            in_num=("_succ_in", "sum"),
            out_denom=("_out", "sum"),
            out_num=("_succ_out", "sum"),
            all_denom=("_set_nonempty", "sum"),
            all_num=("_succ_any", "sum"),
        )
    )

    out["in_system_success"] = np.where(
        out["in_denom"] > 0, out["in_num"] / out["in_denom"], np.nan
    )
    out["out_system_success"] = np.where(
        out["out_denom"] > 0, out["out_num"] / out["out_denom"], np.nan
    )
    out["success rate"] = np.where(
        out["all_denom"] > 0, out["all_num"] / out["all_denom"], np.nan
    )

    final = out[
        [
            "hitter_location",
            "hit_land_location",
            "in_system_success",
            "out_system_success",
            "success rate",
        ]
    ]
    final.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(final)} edges to {OUT_PATH}")


if __name__ == "__main__":
    main()
