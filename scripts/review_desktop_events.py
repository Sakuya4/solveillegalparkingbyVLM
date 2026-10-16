import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))

from illegal_parking.incident_review_service import review_saved_events


def main():
    parser = argparse.ArgumentParser(description="Review saved event evidence without rerunning detection or changing the source report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--event-index", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(review_saved_events(ROOT, args.report, args.event_index, args.output_dir), flush=True)


if __name__ == "__main__":
    main()
