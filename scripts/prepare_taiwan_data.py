from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.data_sources import build_inventory, download_source, load_dataset_sources


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Download Taiwan traffic open-data sources for the main pipeline.")
    parser.add_argument("--manifest", default="data/sources/taiwan_transport_sources.json")
    parser.add_argument("--output-dir", default="data/raw/taiwan")
    parser.add_argument("--inventory", default="data/raw/taiwan/inventory.json")
    parser.add_argument("--only", nargs="*", help="Optional source IDs to download.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir
    inventory_path = ROOT / args.inventory

    sources = load_dataset_sources(manifest_path)
    if args.only:
        requested = set(args.only)
        sources = [source for source in sources if source.id in requested]
    else:
        sources = [source for source in sources if source.enabled_by_default]

    records = [download_source(source, output_dir, overwrite=args.overwrite) for source in sources]
    inventory = build_inventory(sources, output_dir)
    inventory["download_results"] = records

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
