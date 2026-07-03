import json
from pathlib import Path

from hardware_sim.python_golden.dwell_fsm_model import DwellStep, first_candidate_cycle


ROOT = Path(__file__).resolve().parents[1]


def test_dwell_fsm_vectors_match_python_golden_model():
    vectors_path = ROOT / "hardware_sim" / "test_vectors" / "dwell_fsm_vectors.json"
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))

    for vector in vectors:
        sequence = [DwellStep(**step) for step in vector["sequence"]]
        actual = first_candidate_cycle(sequence, vector["threshold_cycles"])

        assert actual == vector["expected_candidate_cycle"], vector["name"]


def test_hardware_sim_sources_are_present():
    assert (ROOT / "hardware_sim" / "rtl" / "dwell_fsm.v").exists()
    assert (ROOT / "hardware_sim" / "rtl" / "bbox_overlap_counter.v").exists()
    assert (ROOT / "hardware_sim" / "systemc" / "event_pipeline_sim.cpp").exists()
