"""
Generate Tableau-ready CSV exports from the raw VREN-style dataset.

Exports written to: Cleaned Data/Analytical Exports/

Included exports:
  Q1) system_vs_outcome.csv
      - Relationship between in/out-of-system state and terminal point outcomes.
  Q2) hit_block_efficiency.csv
      - Attack-ending contacts summarized by hit_type and block pressure.
  Q3) serve_tradeoffs.csv
      - Serve type vs ace/error/win rates and average rally length.
  Momentum) momentum_system_runs.csv
      - Rally-level "in-system share" and max consecutive in-system run length.

Design goals:
  - Keep outputs easy to relate/join in Tableau.
  - Preserve provenance via source_line (line number in raw CSV, header excluded).
  - Avoid relying on any previously "cleaned" artifacts; this reads only raw data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RAW_PATH = REPO_ROOT / "Raw Data" / "dataset_full.csv"
OUT_DIR = REPO_ROOT / "Cleaned Data" / "Analytical Exports"


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (_norm(series) != "")


def _add_game_id_from_rally_resets(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw file stacks multiple games but does not include an explicit game_id.

    This function derives a game_id by detecting rally resets back to 1:
      - A new game begins when rally == 1 and the previous rally != 1.
      - Consecutive rows with rally == 1 are treated as the same game.
    """
    rally_num = pd.to_numeric(df["rally"], errors="coerce")
    game_start = (rally_num == 1) & (rally_num.shift(1) != 1)
    game_id = game_start.cumsum()
    if int(game_id.min()) == 0:
        game_id = game_id + 1
    df = df.copy()
    df["game_id"] = game_id.astype(int)
    return df


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_q1_system_vs_outcome(df: pd.DataFrame) -> Path:
    """
    Output: system_vs_outcome.csv

    Grain: (set_type, win_reason) aggregated over terminal rows.

    Implementation choices:
      - Terminal rally outcome rows are identified by (win_reason != "" AND winning_team in {a,b}).
      - A rally's 'set_type' is taken as the last non-empty set_type (in/out) observed in that rally.
        (This is a transparent rule that behaves well when earlier rows have missing set_type.)
    """
    set_type = _norm(df["set_type"])
    win_reason = _norm(df["win_reason"])
    winning_team = _norm(df["winning_team"])

    is_terminal = (win_reason != "") & winning_team.isin(["a", "b"])
    terminal = df.loc[is_terminal, ["game_id", "rally", "win_reason", "winning_team"]].copy()
    terminal["win_reason"] = win_reason.loc[is_terminal]
    terminal["winning_team"] = winning_team.loc[is_terminal]

    has_set_state = set_type.isin(["in", "out"])
    st = (
        df.loc[has_set_state, ["game_id", "rally", "round", "set_type"]]
        .assign(
            round_num=pd.to_numeric(df.loc[has_set_state, "round"], errors="coerce"),
            set_type_norm=set_type.loc[has_set_state],
        )
        .sort_values(["game_id", "rally", "round_num"])
    )
    # Last non-empty set_type per rally
    last_state = st.groupby(["game_id", "rally"], as_index=False).tail(1)
    last_state = last_state[["game_id", "rally", "set_type_norm"]].rename(
        columns={"set_type_norm": "rally_set_type"}
    )

    terminal = terminal.merge(last_state, on=["game_id", "rally"], how="left")

    out = (
        terminal.groupby(["rally_set_type", "win_reason"], dropna=False)
        .size()
        .reset_index(name="rallies")
        .sort_values(["rally_set_type", "rallies"], ascending=[True, False])
    )
    out["pct_within_set_type"] = out["rallies"] / out.groupby("rally_set_type")[
        "rallies"
    ].transform("sum")

    out_path = OUT_DIR / "system_vs_outcome.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_q2_hit_block_efficiency(df: pd.DataFrame) -> Path:
    """
    Output: hit_block_efficiency.csv

    Grain: (hit_type, num_blockers, block_touch, win_reason) aggregated over rows that:
      - have a non-empty hit_type AND non-empty win_reason AND non-empty winning_team

    Adds:
      - attempts: number of rows in that group
      - points_won: count where row team == winning_team
      - point_win_rate: points_won / attempts
    """
    hit_type = _norm(df["hit_type"])
    win_reason = _norm(df["win_reason"])
    winning_team = _norm(df["winning_team"])
    team = _norm(df["team"])
    block_touch = _norm(df["block_touch"])

    num_blockers = pd.to_numeric(df["num_blockers"], errors="coerce")

    mask = (hit_type != "") & (win_reason != "") & winning_team.isin(["a", "b"]) & team.isin(
        ["a", "b"]
    )
    sub = pd.DataFrame(
        {
            "hit_type": hit_type.loc[mask],
            "num_blockers": num_blockers.loc[mask],
            "block_touch": block_touch.loc[mask].replace({"": np.nan}),
            "win_reason": win_reason.loc[mask],
            "team": team.loc[mask],
            "winning_team": winning_team.loc[mask],
        }
    )
    sub["won_point"] = (sub["team"] == sub["winning_team"]).astype(int)

    out = (
        sub.groupby(["hit_type", "num_blockers", "block_touch", "win_reason"], dropna=False)
        .agg(attempts=("won_point", "size"), points_won=("won_point", "sum"))
        .reset_index()
    )
    out["point_win_rate"] = np.where(out["attempts"] > 0, out["points_won"] / out["attempts"], np.nan)
    out = out.sort_values(["attempts", "hit_type"], ascending=[False, True])

    out_path = OUT_DIR / "hit_block_efficiency.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_q3_serve_tradeoffs(df: pd.DataFrame) -> Path:
    """
    Output: serve_tradeoffs.csv

    Grain: serve_type aggregated over rows with non-empty serve_type.

    Definitions:
      - serving_team is the opposite of row team (per VREN convention: row team is typically receiver/possessor).
      - ace counts only when win_reason == "ace" and winning_team == serving_team
      - service_error counts when win_reason == "serve_error"
      - server_point_win_rate: winning_team == serving_team
      - rally_max_round: max numeric round within (game_id, rally)

    Note: Serve_type appears on a few non-round-1 rows in the raw file; this export includes them
    because it's intended to reflect what's in the raw data. You can filter round==1 in Tableau if desired.
    """
    serve_type = _norm(df["serve_type"])
    win_reason = _norm(df["win_reason"])
    winning_team = _norm(df["winning_team"])
    team = _norm(df["team"])

    round_num = pd.to_numeric(df["round"], errors="coerce")

    rally_len = (
        pd.DataFrame(
            {
                "game_id": df["game_id"],
                "rally": df["rally"],
                "round_num": round_num,
            }
        )
        .groupby(["game_id", "rally"], as_index=False)["round_num"]
        .max()
        .rename(columns={"round_num": "rally_max_round"})
    )

    serve_rows = serve_type != ""
    serves = df.loc[serve_rows, ["game_id", "rally", "round", "team", "serve_type"]].copy()
    serves["serve_type"] = serve_type.loc[serve_rows]
    serves["team"] = team.loc[serve_rows]
    serves["round_num"] = round_num.loc[serve_rows]
    serves["serving_team"] = serves["team"].map({"a": "b", "b": "a"}).fillna("")

    terminal = (win_reason != "") & winning_team.isin(["a", "b"])
    term = (
        pd.DataFrame(
            {
                "game_id": df.loc[terminal, "game_id"],
                "rally": df.loc[terminal, "rally"],
                "win_reason": win_reason.loc[terminal],
                "winning_team": winning_team.loc[terminal],
            }
        )
        .drop_duplicates(["game_id", "rally"])
    )

    serves = serves.merge(term, on=["game_id", "rally"], how="left").merge(
        rally_len, on=["game_id", "rally"], how="left"
    )

    serves["is_ace_by_server"] = (
        (serves["win_reason"] == "ace") & (serves["winning_team"] == serves["serving_team"])
    ).astype(int)
    serves["is_service_error"] = (serves["win_reason"] == "serve_error").astype(int)
    serves["server_won_point"] = (serves["winning_team"] == serves["serving_team"]).astype(int)

    out = (
        serves.groupby(["serve_type"], dropna=False)
        .agg(
            serve_rows=("serve_type", "size"),
            ace=("is_ace_by_server", "sum"),
            service_error=("is_service_error", "sum"),
            points_won=("server_won_point", "sum"),
            avg_rally_max_round=("rally_max_round", "mean"),
            pct_serve_rows_round1=("round_num", lambda s: float((s == 1).mean())),
        )
        .reset_index()
    )
    out["ace_rate"] = out["ace"] / out["serve_rows"]
    out["service_error_rate"] = out["service_error"] / out["serve_rows"]
    out["server_point_win_rate"] = out["points_won"] / out["serve_rows"]
    out = out.sort_values("serve_rows", ascending=False)

    out_path = OUT_DIR / "serve_tradeoffs.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_momentum_system_runs(df: pd.DataFrame) -> Path:
    """
    Output: momentum_system_runs.csv

    Grain: rally (game_id, rally).

    Features (based on set_type sequence within rally):
      - n_set_rows: number of rows with set_type in {in,out}
      - in_share / out_share
      - max_in_run: maximum consecutive run length of set_type == in

    Joins:
      - winning_team per rally (from any terminal row for that rally)
    """
    set_type = _norm(df["set_type"])
    win_reason = _norm(df["win_reason"])
    winning_team = _norm(df["winning_team"])
    team = _norm(df["team"])
    round_num = pd.to_numeric(df["round"], errors="coerce")

    st_mask = set_type.isin(["in", "out"])
    st = pd.DataFrame(
        {
            "game_id": df.loc[st_mask, "game_id"],
            "rally": df.loc[st_mask, "rally"],
            "round_num": round_num.loc[st_mask],
            "set_type": set_type.loc[st_mask],
            "team": team.loc[st_mask],
        }
    ).sort_values(["game_id", "rally", "round_num"])

    def max_in_run(values: Iterable[str]) -> int:
        best = 0
        cur = 0
        for v in values:
            if v == "in":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    # Overall (both teams pooled) rally features
    agg = (
        st.groupby(["game_id", "rally"], as_index=False)
        .agg(
            n_set_rows=("set_type", "size"),
            n_in=("set_type", lambda x: int((x == "in").sum())),
            n_out=("set_type", lambda x: int((x == "out").sum())),
            max_in_run=("set_type", max_in_run),
        )
    )
    agg["in_share"] = np.where(agg["n_set_rows"] > 0, agg["n_in"] / agg["n_set_rows"], np.nan)
    agg["out_share"] = np.where(agg["n_set_rows"] > 0, agg["n_out"] / agg["n_set_rows"], np.nan)

    # Per-team in/out shares (within the same rally)
    team_agg = (
        st[st["team"].isin(["a", "b"])]
        .groupby(["game_id", "rally", "team"], as_index=False)
        .agg(
            n_set_rows=("set_type", "size"),
            n_in=("set_type", lambda x: int((x == "in").sum())),
            n_out=("set_type", lambda x: int((x == "out").sum())),
            max_in_run=("set_type", max_in_run),
        )
    )
    team_agg["in_share"] = np.where(
        team_agg["n_set_rows"] > 0, team_agg["n_in"] / team_agg["n_set_rows"], np.nan
    )
    team_agg["out_share"] = np.where(
        team_agg["n_set_rows"] > 0, team_agg["n_out"] / team_agg["n_set_rows"], np.nan
    )

    wide = (
        team_agg.pivot(index=["game_id", "rally"], columns="team")
        .sort_index(axis=1, level=1)
    )
    # Flatten columns like ('in_share','a') -> 'team_a_in_share'
    wide.columns = [f"team_{t}_{m}" for (m, t) in wide.columns.to_list()]
    wide = wide.reset_index()
    agg = agg.merge(wide, on=["game_id", "rally"], how="left")

    terminal = (win_reason != "") & winning_team.isin(["a", "b"])
    term = (
        pd.DataFrame(
            {
                "game_id": df.loc[terminal, "game_id"],
                "rally": df.loc[terminal, "rally"],
                "winning_team": winning_team.loc[terminal],
            }
        )
        .drop_duplicates(["game_id", "rally"])
    )
    out = agg.merge(term, on=["game_id", "rally"], how="left").sort_values(
        ["game_id", "rally"]
    )

    out_path = OUT_DIR / "momentum_system_runs.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    _ensure_out_dir()
    df = pd.read_csv(RAW_PATH)
    df = df.reset_index(drop=True)
    df["source_line"] = df.index + 2  # header excluded
    df = _add_game_id_from_rally_resets(df)

    written = [
        export_q1_system_vs_outcome(df),
        export_q2_hit_block_efficiency(df),
        export_q3_serve_tradeoffs(df),
        export_momentum_system_runs(df),
    ]

    print("Wrote exports:")
    for p in written:
        print(f"- {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

