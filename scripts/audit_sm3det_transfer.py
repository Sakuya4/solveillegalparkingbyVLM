from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.sm3det_audit import audit_sm3det_release


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the official SM3Det release for roadside-CCTV transfer."
    )
    parser.add_argument("--repository", default="outputs/build/sm3det-official")
    parser.add_argument(
        "--output", default="docs/assets/sm3det_transfer_audit.json"
    )
    args = parser.parse_args()

    repository = _resolve(args.repository)
    if not (repository / "README.md").is_file():
        raise SystemExit(
            "SM3Det release not found. Clone https://github.com/zcablii/SM3Det "
            f"to {repository} or pass --repository."
        )
    report = audit_sm3det_release(repository)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
