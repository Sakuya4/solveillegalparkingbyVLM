from pathlib import Path

from illegal_parking.sm3det_audit import audit_sm3det_release


def test_audit_sm3det_release_reads_official_configuration(tmp_path: Path) -> None:
    repository = tmp_path / "SM3Det"
    config_dir = repository / "configs" / "SM3Det"
    config_dir.mkdir(parents=True)
    (repository / "README.md").write_text(
        """
        <td>SM3Det</td><td>487G</td><td>178M</td><td>Overall</td>
        <td>80.68</td><td>50.20</td><td>51.31</td>
        """,
        encoding="utf-8",
    )
    (config_dir / "SM3Det_convnext_t.py").write_text(
        """
num_classes = 26
source_ratio = [2, 1, 1]
model = dict(num_experts=8, top_k=3)
        """,
        encoding="utf-8",
    )

    report = audit_sm3det_release(repository)

    assert report["source_ratio"] == [2, 1, 1]
    assert report["mixture_of_experts"] == {"num_experts": 8, "top_k": 3}
    assert report["official_release_metrics"]["flops_g"] == 487
    assert report["official_release_metrics"]["parameters_m"] == 178
    assert report["cctv_transfer_assessment"]["direct_cctv_metric_available"] is False
