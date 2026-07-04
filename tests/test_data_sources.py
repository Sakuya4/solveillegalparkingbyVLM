import json

from illegal_parking.data_sources import (
    DatasetSource,
    build_inventory,
    load_dataset_sources,
    summarize_csv_file,
    summarize_zip_file,
)


def test_load_dataset_sources_validates_manifest(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "demo",
                    "title": "Demo dataset",
                    "role": "violation_records",
                    "source_page": "https://example.test/dataset",
                    "license": "demo",
                    "download_url": "https://example.test/demo.csv",
                    "target_filename": "demo.csv",
                    "format": "csv",
                }
            ]
        ),
        encoding="utf-8",
    )

    sources = load_dataset_sources(manifest)

    assert sources == [
        DatasetSource(
            id="demo",
            title="Demo dataset",
            role="violation_records",
            source_page="https://example.test/dataset",
            license="demo",
            download_url="https://example.test/demo.csv",
            target_filename="demo.csv",
            format="csv",
            encoding=None,
            enabled_by_default=True,
            notes=None,
        )
    ]


def test_summarize_csv_file_counts_rows_and_columns(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("year,month,Road\n2024,01,A road\n2024,02,B road\n", encoding="utf-8")

    summary = summarize_csv_file(csv_path)

    assert summary["rows"] == 2
    assert summary["columns"] == ["year", "month", "Road"]
    assert summary["encoding"] == "utf-8-sig"


def test_summarize_csv_file_handles_big5_csv(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_bytes("西元年,道路\n2024,測試路\n".encode("big5"))

    summary = summarize_csv_file(csv_path)

    assert summary["rows"] == 1
    assert summary["columns"] == ["西元年", "道路"]
    assert summary["encoding"] == "big5"


def test_build_inventory_marks_metadata_only_sources(tmp_path):
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text("id,value\n1,a\n", encoding="utf-8")
    sources = [
        DatasetSource(
            id="demo",
            title="Demo dataset",
            role="violation_records",
            source_page="https://example.test/dataset",
            license="demo",
            download_url="https://example.test/demo.csv",
            target_filename="demo.csv",
            format="csv",
        ),
        DatasetSource(
            id="fisheye8k",
            title="FishEye8K",
            role="vision_training",
            source_page="https://example.test/fisheye8k",
            license="research",
            download_url=None,
            target_filename=None,
            format="external",
        ),
    ]

    inventory = build_inventory(sources, tmp_path)

    assert inventory["sources"][0]["rows"] == 1
    assert inventory["sources"][0]["status"] == "downloaded"
    assert inventory["sources"][1]["status"] == "metadata_only"


def test_summarize_zip_file_lists_entries(tmp_path):
    import zipfile

    zip_path = tmp_path / "records.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("records.csv", "id,value\n1,a\n")

    summary = summarize_zip_file(zip_path)

    assert summary["entries"] == ["records.csv"]
    assert summary["entry_count"] == 1
