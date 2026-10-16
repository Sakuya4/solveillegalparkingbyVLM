import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.observation_review import summarize_observation_reviews


def main():
    parser = argparse.ArgumentParser(description="Evaluate hash-bound reviewed normal observations under one fixed inference contract.")
    parser.add_argument("--review-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(name) for name in sorted(glob.glob(args.review_glob))]
    if not paths:
        parser.error("No observation review records matched")
    result = summarize_observation_reviews(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("normal_camera_hours", "normal_alert_episodes", "alerts_per_normal_camera_hour", "poisson_one_sided_upper_95_per_hour")}, indent=2))


if __name__ == "__main__":
    main()
