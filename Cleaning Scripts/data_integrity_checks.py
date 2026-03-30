"""
Scan the raw VREN-style dataset for simple consistency issues.

Outputs:
  - Cleaning Scripts/integrity_flagged_rows.csv  (row-level flags, includes source line number)
  - Cleaning Scripts/integrity_summary.csv       (counts by issue_code)

Notes:
  - The raw CSV is treated as an event log; many blanks are structural ("not applicable").
  - These checks are meant to *flag* potential issues, not automatically "fix" them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "Raw Data" / "dataset_full.csv"

FLAGGED_OUT_PATH = SCRIPT_DIR / "integrity_flagged_rows.csv"
SUMMARY_OUT_PATH = SCRIPT_DIR / "integrity_summary.csv"


def _norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


@dataclass(frozen=True)
class Issue:
    code: str
    description: str


ISSUES: dict[str, Issue] = {
    "blocked_missing_block_fields": Issue(
        code="blocked_missing_block_fields",
        description='hit_type == "blocked" but num_blockers and/or block_touch is missing',
    ),
    "blocked_outcome_block_touch_no": Issue(
        code="blocked_outcome_block_touch_no",
        description='win_reason/lose_reason == "blocked" but block_touch == "no"',
    ),
    "serve_type_not_round1": Issue(
        code="serve_type_not_round1",
        description="serve_type present on a row where round != 1",
    ),
}


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Keep source CSV line numbers for easy lookup (header is line 1).
    df = df.reset_index(drop=True)
    df["source_line"] = df.index + 2

    # Normalize relevant categorical fields.
    for col in [
        "hit_type",
        "win_reason",
        "lose_reason",
        "block_touch",
        "serve_type",
        "team",
        "winning_team",
    ]:
        if col in df.columns:
            df[col] = _norm(df[col])

    # Numeric parsing (leave NaN if blank/invalid).
    round_num = pd.to_numeric(df.get("round"), errors="coerce")
    num_blockers = pd.to_numeric(df.get("num_blockers"), errors="coerce")

    hit_type = df.get("hit_type", pd.Series([""] * len(df)))
    win_reason = df.get("win_reason", pd.Series([""] * len(df)))
    lose_reason = df.get("lose_reason", pd.Series([""] * len(df)))
    block_touch = df.get("block_touch", pd.Series([""] * len(df)))
    serve_type = df.get("serve_type", pd.Series([""] * len(df)))

    def nonempty(s: pd.Series) -> pd.Series:
        return s.notna() & (_norm(s) != "")

    flags: list[pd.DataFrame] = []

    # 1) hit_type == blocked but missing block metadata
    is_blocked = hit_type == "blocked"
    missing_num_blockers = num_blockers.isna()
    missing_block_touch = (block_touch == "") | block_touch.isna()
    m1 = is_blocked & (missing_num_blockers | missing_block_touch)
    if m1.any():
        out = df.loc[m1].copy()
        out["issue_code"] = ISSUES["blocked_missing_block_fields"].code
        out["issue_description"] = ISSUES["blocked_missing_block_fields"].description
        flags.append(out)

    # 2) blocked outcome but block_touch says "no"
    outcome_blocked = (win_reason == "blocked") | (lose_reason == "blocked")
    m2 = outcome_blocked & (block_touch == "no")
    if m2.any():
        out = df.loc[m2].copy()
        out["issue_code"] = ISSUES["blocked_outcome_block_touch_no"].code
        out["issue_description"] = ISSUES["blocked_outcome_block_touch_no"].description
        flags.append(out)

    # 3) serve_type present but not round 1
    m3 = nonempty(serve_type) & (round_num != 1)
    if m3.any():
        out = df.loc[m3].copy()
        out["issue_code"] = ISSUES["serve_type_not_round1"].code
        out["issue_description"] = ISSUES["serve_type_not_round1"].description
        flags.append(out)

    if not flags:
        flagged = pd.DataFrame(
            columns=["issue_code", "issue_description", "source_line"] + list(df.columns)
        )
    else:
        flagged = pd.concat(flags, ignore_index=True)

    # Put the key columns up front, keep everything else for debugging in spreadsheets.
    front_cols = [
        "issue_code",
        "issue_description",
        "source_line",
        "rally",
        "round",
        "team",
        "serve_type",
        "hit_type",
        "num_blockers",
        "block_touch",
        "win_reason",
        "lose_reason",
        "winning_team",
    ]
    front_cols = [c for c in front_cols if c in flagged.columns]
    ordered_cols = front_cols + [c for c in flagged.columns if c not in front_cols]
    flagged = flagged.loc[:, ordered_cols]

    flagged.to_csv(FLAGGED_OUT_PATH, index=False)

    summary = (
        flagged.groupby(["issue_code", "issue_description"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "issue_code"], ascending=[False, True])
    )
    summary.to_csv(SUMMARY_OUT_PATH, index=False)

    print(f"Scanned: {DATA_PATH}")
    print(f"Flagged rows: {len(flagged)}")
    print(f"Wrote: {FLAGGED_OUT_PATH}")
    print(f"Wrote: {SUMMARY_OUT_PATH}")
    if not summary.empty:
        print("\nTop issues:")
        for _, r in summary.head(10).iterrows():
            print(f"- {r['issue_code']}: {int(r['count'])}")


if __name__ == "__main__":
    main()

