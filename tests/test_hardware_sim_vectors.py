import json
from pathlib import Path

from hardware_sim.python_golden.dwell_fsm_model import DwellStep, first_candidate_cycle
from hardware_sim.python_golden.overlap_counter_model import OverlapPixel, count_overlap
from hardware_sim.python_golden.motion_trigger_model import MotionStep, trigger_cycles


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
    assert (ROOT / "hardware_sim" / "rtl" / "motion_trigger.v").exists()
    assert (ROOT / "hardware_sim" / "systemc" / "event_pipeline_sim.cpp").exists()
    assert (ROOT / "hardware_sim" / "systemc" / "npu_queue_sim.cpp").exists()
    assert (ROOT / "hardware_sim" / "testbench" / "dwell_fsm_tb.v").exists()
    assert (ROOT / "hardware_sim" / "testbench" / "bbox_overlap_counter_tb.v").exists()
    assert (ROOT / "hardware_sim" / "testbench" / "motion_trigger_tb.v").exists()


def test_overlap_counter_vectors_match_python_golden_model():
    vectors_path = ROOT / "hardware_sim" / "test_vectors" / "overlap_counter_vectors.json"
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))

    for vector in vectors:
        pixels = [OverlapPixel(**pixel) for pixel in vector["pixels"]]
        actual = count_overlap(pixels, vector["min_red_pixels"])

        assert actual.band_pixels == vector["expected"]["band_pixels"], vector["name"]
        assert actual.red_pixels == vector["expected"]["red_pixels"], vector["name"]
        assert actual.overlap_hit == vector["expected"]["overlap_hit"], vector["name"]


def test_motion_trigger_vectors_match_python_golden_model():
    vectors_path = ROOT / "hardware_sim" / "test_vectors" / "motion_trigger_vectors.json"
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))

    for vector in vectors:
        sequence = [MotionStep(**step) for step in vector["sequence"]]
        actual = trigger_cycles(
            sequence,
            min_roi_motion=vector["min_roi_motion"],
            roi_margin=vector["roi_margin"],
        )

        assert actual == vector["expected_trigger_cycles"], vector["name"]
