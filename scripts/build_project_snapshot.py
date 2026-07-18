from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.project_snapshot import write_project_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the integrated project dashboard snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dashboard/data/project_snapshot.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = ROOT
    output = args.output if args.output.is_absolute() else repo_root / args.output
    snapshot = write_project_snapshot(repo_root, output)
    print(
        f"Wrote {output} with {len(snapshot['readiness']['items'])} readiness checks "
        f"and {len(snapshot['model_comparison'])} model rows."
    )


if __name__ == "__main__":
    main()
