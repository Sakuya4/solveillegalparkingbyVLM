from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSource:
    id: str
    title: str
    role: str
    source_page: str
    license: str
    download_url: str | None
    target_filename: str | None
    format: str
    encoding: str | None = None
    enabled_by_default: bool = True
    notes: str | None = None


def load_dataset_sources(path: str | Path) -> list[DatasetSource]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Dataset source manifest must be a list.")

    sources: list[DatasetSource] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset source row {index} must be an object.")
        sources.append(DatasetSource(**row))
    return sources


def download_source(source: DatasetSource, output_dir: str | Path, overwrite: bool = False) -> dict[str, Any]:
    if not source.download_url or not source.target_filename:
        return _source_record(source, status="metadata_only")

    output_path = Path(output_dir) / source.target_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return _source_record(source, status="exists", path=str(output_path), bytes=output_path.stat().st_size)

    request = urllib.request.Request(
        source.download_url,
        headers={
            "User-Agent": "solveillegalparkingbyVLM data-prep/0.1",
            "Accept": "text/csv,application/json,application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, output_path.open("wb") as file:
        shutil.copyfileobj(response, file)

    record = _source_record(source, status="downloaded", path=str(output_path), bytes=output_path.stat().st_size)
    if source.format.lower() == "csv":
        record.update(summarize_csv_file(output_path, preferred_encoding=source.encoding))
    elif source.format.lower() == "zip":
        record.update(summarize_zip_file(output_path))
    return record


def summarize_csv_file(path: str | Path, preferred_encoding: str | None = None) -> dict[str, Any]:
    csv_path = Path(path)
    encoding = _detect_encoding(csv_path, preferred_encoding)
    rows = 0
    columns: list[str] = []

    with csv_path.open("r", encoding=encoding, errors="replace", newline="") as file:
        reader = csv.reader(file)
        try:
            columns = next(reader)
        except StopIteration:
            columns = []
        for _row in reader:
            rows += 1

    return {
        "rows": rows,
        "columns": columns,
        "encoding": encoding,
    }


def summarize_zip_file(path: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        entries = [_normalize_zip_name(name) for name in archive.namelist() if not name.endswith("/")]
    return {
        "entry_count": len(entries),
        "entries": entries,
    }


def build_inventory(sources: list[DatasetSource], output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    inventory_sources: list[dict[str, Any]] = []
    for source in sources:
        if not source.download_url or not source.target_filename:
            inventory_sources.append(_source_record(source, status="metadata_only"))
            continue

        local_path = output_path / source.target_filename
        if not local_path.exists():
            inventory_sources.append(_source_record(source, status="missing", path=str(local_path)))
            continue

        record = _source_record(
            source,
            status="downloaded",
            path=str(local_path),
            bytes=local_path.stat().st_size,
        )
        if source.format.lower() == "csv":
            record.update(summarize_csv_file(local_path, preferred_encoding=source.encoding))
        elif source.format.lower() == "zip":
            record.update(summarize_zip_file(local_path))
        inventory_sources.append(record)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_data_dir": str(output_path),
        "sources": inventory_sources,
    }


def _detect_encoding(path: Path, preferred_encoding: str | None = None) -> str:
    sample = path.read_bytes()[:4096]
    if preferred_encoding and _can_decode_sample(sample, preferred_encoding):
        return preferred_encoding

    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        if _can_decode_sample(sample, encoding):
            return encoding
    return "utf-8-sig"


def _can_decode_sample(sample: bytes, encoding: str) -> bool:
    for trim in (0, 1, 2, 3, 4):
        candidate = sample[:-trim] if trim else sample
        try:
            candidate.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def _normalize_zip_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("big5")
    except UnicodeError:
        return name


def _source_record(source: DatasetSource, status: str, **extra: Any) -> dict[str, Any]:
    record = asdict(source)
    record["status"] = status
    record.update(extra)
    return record
