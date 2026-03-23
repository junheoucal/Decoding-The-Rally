"""
Build a game-level table from rally-level data.

Each output row is one game. Games are separated when rally resets to 1,
but consecutive rows with rally == 1 are treated as the same game.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "Raw Data" / "dataset_full.csv"
OUT_PATH = SCRIPT_DIR / "vren_game_features.csv"


def is_nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def serving_team_from_row_team(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map({"a": "b", "b": "a"})


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def team_metrics(
    game_df: pd.DataFrame,
    team: str,
    set_locations: list[str],
    serve_types: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}

    team_rows = game_df["team_norm"] == team

    # Pass in-rate for this team's receptions.
    pass_nonempty = team_rows & game_df["pass_rating_nonempty"]
    pass_in = pass_nonempty & (game_df["pass_rating_norm"] == "in")
    out["pass_in_pct"] = safe_pct(int(pass_in.sum()), int(pass_nonempty.sum()))

    # Set-location distribution for this team's set attempts.
    set_denom_mask = team_rows & game_df["set_type_nonempty"]
    set_denom = int(set_denom_mask.sum())
    for loc in set_locations:
        loc_mask = set_denom_mask & (game_df["set_location_norm"] == loc)
        out[f"set_location_{slug(loc)}_pct"] = safe_pct(int(loc_mask.sum()), set_denom)

    # Serving metrics: serving team is opposite of row team.
    serve_denom_mask = (game_df["serving_team"] == team) & game_df["serve_type_nonempty"]
    serve_denom = int(serve_denom_mask.sum())

    for serve in serve_types:
        serve_mask = serve_denom_mask & (game_df["serve_type_norm"] == serve)
        out[f"serve_type_{slug(serve)}_pct"] = safe_pct(int(serve_mask.sum()), serve_denom)

    ace_mask = (
        serve_denom_mask
        & (game_df["win_reason_norm"] == "ace")
        & (game_df["winning_team_norm"] == team)
    )
    out["ace_pct"] = safe_pct(int(ace_mask.sum()), serve_denom)

    service_error_mask = serve_denom_mask & (game_df["win_reason_norm"] == "serve_error")
    out["service_error_pct"] = safe_pct(int(service_error_mask.sum()), serve_denom)

    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Normalized helper columns.
    df["team_norm"] = df["team"].astype(str).str.strip().str.lower()
    df["winning_team_norm"] = df["winning_team"].astype(str).str.strip().str.lower()
    df["pass_rating_norm"] = df["pass_rating"].fillna("").astype(str).str.strip().str.lower()
    df["set_type_nonempty"] = is_nonempty(df["set_type"])
    df["set_location_norm"] = df["set_location"].fillna("").astype(str).str.strip().str.lower()
    df["serve_type_nonempty"] = is_nonempty(df["serve_type"])
    df["serve_type_norm"] = df["serve_type"].fillna("").astype(str).str.strip().str.lower()
    df["pass_rating_nonempty"] = is_nonempty(df["pass_rating"])
    df["win_reason_norm"] = df["win_reason"].fillna("").astype(str).str.strip().str.lower()
    df["serving_team"] = serving_team_from_row_team(df["team"])

    # Identify games: new game only when rally changes from non-1 to 1.
    rally_num = pd.to_numeric(df["rally"], errors="coerce")
    game_start = (rally_num == 1) & (rally_num.shift(1) != 1)
    game_id = game_start.cumsum()
    if int(game_id.min()) == 0:
        game_id = game_id + 1
    df["game_id"] = game_id.astype(int)

    set_locations = sorted(
        df.loc[df["set_type_nonempty"] & is_nonempty(df["set_location"]), "set_location_norm"]
        .dropna()
        .unique()
        .tolist()
    )
    serve_types = sorted(
        df.loc[df["serve_type_nonempty"], "serve_type_norm"]
        .dropna()
        .unique()
        .tolist()
    )

    rows: list[dict[str, object]] = []
    for gid, gdf in df.groupby("game_id", sort=True):
        row: dict[str, object] = {"game_id": int(gid)}

        a_metrics = team_metrics(gdf, "a", set_locations, serve_types)
        b_metrics = team_metrics(gdf, "b", set_locations, serve_types)

        for k, v in a_metrics.items():
            row[f"team_a_{k}"] = v
        for k, v in b_metrics.items():
            row[f"team_b_{k}"] = v

        # Game winner taken from the last non-empty winning_team in the game.
        winners = gdf.loc[is_nonempty(gdf["winning_team"]), "winning_team_norm"]
        row["winning_team"] = winners.iloc[-1] if not winners.empty else np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("game_id").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} games to {OUT_PATH}")


if __name__ == "__main__":
    main()
