import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Jump:
    line_number: int
    rally: str
    prev_round: int
    curr_round: int


def _parse_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def count_nonconsecutive_round_jumps(csv_path: Path, *, max_examples: int) -> tuple[int, list[Jump]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        if "round" not in reader.fieldnames:
            raise ValueError(f"Missing required column 'round'. Found: {reader.fieldnames}")
        if "rally" not in reader.fieldnames:
            raise ValueError(f"Missing required column 'rally'. Found: {reader.fieldnames}")

        count = 0
        examples: list[Jump] = []

        prev_rally: Optional[str] = None
        prev_round: Optional[int] = None

        # DictReader counts data rows only; we want actual file line numbers.
        # Line 1 is header, so first data row is line 2.
        for idx, row in enumerate(reader, start=2):
            rally = (row.get("rally") or "").strip()
            curr_round = _parse_int(row.get("round") or "")

            if rally == "" or curr_round is None:
                prev_rally = rally if rally != "" else prev_rally
                prev_round = None
                continue

            if prev_rally != rally:
                prev_rally = rally
                prev_round = curr_round
                continue

            if prev_round is not None and curr_round != prev_round + 1:
                count += 1
                if len(examples) < max_examples:
                    examples.append(
                        Jump(
                            line_number=idx,
                            rally=rally,
                            prev_round=prev_round,
                            curr_round=curr_round,
                        )
                    )

            prev_round = curr_round

        return count, examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count rows where the 'round' column jumps non-consecutively within the same rally "
            "(e.g. 1->3 instead of 1->2)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Raw Data") / "dataset_full.csv",
        help="Path to dataset_full.csv (default: Raw Data/dataset_full.csv)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="How many example jumps to print (default: 20)",
    )
    args = parser.parse_args()

    count, examples = count_nonconsecutive_round_jumps(args.csv, max_examples=max(0, args.max_examples))

    print(f"Non-consecutive 'round' jumps (within same rally): {count}")
    if examples:
        print()
        print("Examples:")
        for j in examples:
            print(f"- line {j.line_number}: rally={j.rally} round {j.prev_round} -> {j.curr_round}")


if __name__ == "__main__":
    main()

