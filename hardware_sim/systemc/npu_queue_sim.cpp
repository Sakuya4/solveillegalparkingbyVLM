#include <systemc>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace sc_core;

struct QueueProfile {
    unsigned camera_count = 4;
    double camera_fps = 15.0;
    double duration_sec = 60.0;
    double detector_latency_ms = 29.0;
    double temporal_latency_ms = 3.0;
    unsigned npu_queue_capacity = 8;
    unsigned candidate_stride = 100;
    std::string candidate_flags_path;
    std::vector<bool> candidate_flags;
    double review_latency_ms = 300.0;
    unsigned review_queue_capacity = 8;
};

SC_MODULE(CameraFarm) {
    sc_fifo_out<unsigned> frames;
    QueueProfile profile;
    unsigned generated = 0;
    unsigned dropped = 0;

    SC_HAS_PROCESS(CameraFarm);

    CameraFarm(sc_module_name name, QueueProfile input_profile)
        : sc_module(name), profile(input_profile) {
        SC_THREAD(run);
    }

    void run() {
        const sc_time frame_period(1000.0 / profile.camera_fps, SC_MS);
        unsigned frame_id = 0;
        while (sc_time_stamp() < sc_time(profile.duration_sec, SC_SEC)) {
            for (unsigned camera = 0; camera < profile.camera_count; ++camera) {
                ++generated;
                if (!frames->nb_write(frame_id++)) {
                    ++dropped;
                }
            }
            wait(frame_period);
        }
    }
};

SC_MODULE(NpuWorker) {
    sc_fifo_in<unsigned> frames;
    sc_fifo_out<unsigned> candidates;
    QueueProfile profile;
    unsigned processed = 0;
    unsigned candidate_count = 0;
    unsigned dropped_candidates = 0;
    bool busy = false;

    SC_HAS_PROCESS(NpuWorker);

    NpuWorker(sc_module_name name, QueueProfile input_profile)
        : sc_module(name), profile(input_profile) {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            unsigned frame_id = frames->read();
            busy = true;
            wait(sc_time(profile.detector_latency_ms + profile.temporal_latency_ms, SC_MS));
            ++processed;
            busy = false;
            const bool is_candidate = !profile.candidate_flags.empty()
                ? frame_id < profile.candidate_flags.size() && profile.candidate_flags[frame_id]
                : profile.candidate_stride > 0 && processed % profile.candidate_stride == 0;
            if (is_candidate) {
                ++candidate_count;
                if (!candidates->nb_write(frame_id)) {
                    ++dropped_candidates;
                }
            }
        }
    }
};

SC_MODULE(ReviewWorker) {
    sc_fifo_in<unsigned> candidates;
    QueueProfile profile;
    unsigned completed = 0;
    bool busy = false;

    SC_HAS_PROCESS(ReviewWorker);

    ReviewWorker(sc_module_name name, QueueProfile input_profile)
        : sc_module(name), profile(input_profile) {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            candidates->read();
            busy = true;
            wait(sc_time(profile.review_latency_ms, SC_MS));
            ++completed;
            busy = false;
        }
    }
};

static QueueProfile parse_args(int argc, char* argv[]) {
    if ((argc - 1) % 2 != 0) {
        std::cerr << "Every option requires a value" << std::endl;
        std::exit(2);
    }
    QueueProfile profile;
    for (int index = 1; index + 1 < argc; index += 2) {
        std::string option = argv[index];
        std::string value = argv[index + 1];
        if (option == "--camera-count") profile.camera_count = std::stoul(value);
        else if (option == "--camera-fps") profile.camera_fps = std::stod(value);
        else if (option == "--duration-sec") profile.duration_sec = std::stod(value);
        else if (option == "--detector-latency-ms") profile.detector_latency_ms = std::stod(value);
        else if (option == "--temporal-latency-ms") profile.temporal_latency_ms = std::stod(value);
        else if (option == "--npu-queue-capacity") profile.npu_queue_capacity = std::stoul(value);
        else if (option == "--candidate-stride") profile.candidate_stride = std::stoul(value);
        else if (option == "--candidate-flags") profile.candidate_flags_path = value;
        else if (option == "--review-latency-ms") profile.review_latency_ms = std::stod(value);
        else if (option == "--review-queue-capacity") profile.review_queue_capacity = std::stoul(value);
        else {
            std::cerr << "Unknown option: " << option << std::endl;
            std::exit(2);
        }
    }
    if (profile.camera_count == 0 || profile.camera_fps <= 0.0 ||
        profile.duration_sec <= 0.0 || profile.detector_latency_ms < 0.0 ||
        profile.temporal_latency_ms < 0.0 || profile.review_latency_ms < 0.0 ||
        profile.npu_queue_capacity == 0 || profile.review_queue_capacity == 0) {
        std::cerr << "Counts, rates, duration, and FIFO capacities must be positive; "
                     "latencies must be non-negative"
                  << std::endl;
        std::exit(2);
    }
    if (!profile.candidate_flags_path.empty()) {
        std::ifstream input(profile.candidate_flags_path);
        if (!input) {
            std::cerr << "Cannot open candidate flags: " << profile.candidate_flags_path << std::endl;
            std::exit(2);
        }
        std::string value;
        while (input >> value) {
            if (value == "0") profile.candidate_flags.push_back(false);
            else if (value == "1") profile.candidate_flags.push_back(true);
            else {
                std::cerr << "Candidate flags must contain only 0 or 1" << std::endl;
                std::exit(2);
            }
        }
        if (profile.candidate_flags.empty()) {
            std::cerr << "Candidate flags file is empty" << std::endl;
            std::exit(2);
        }
        const auto expected_count = static_cast<std::size_t>(profile.camera_count) *
            static_cast<std::size_t>(std::lround(profile.camera_fps * profile.duration_sec));
        if (profile.candidate_flags.size() != expected_count) {
            std::cerr << "Candidate flag count must match generated frames: "
                      << profile.candidate_flags.size() << " != " << expected_count << std::endl;
            std::exit(2);
        }
    }
    return profile;
}

int sc_main(int argc, char* argv[]) {
    QueueProfile profile = parse_args(argc, argv);
    sc_fifo<unsigned> frame_queue("frame_queue", profile.npu_queue_capacity);
    sc_fifo<unsigned> review_queue("review_queue", profile.review_queue_capacity);

    CameraFarm camera_farm("camera_farm", profile);
    NpuWorker npu("npu", profile);
    ReviewWorker reviewer("reviewer", profile);
    camera_farm.frames(frame_queue);
    npu.frames(frame_queue);
    npu.candidates(review_queue);
    reviewer.candidates(review_queue);

    sc_start(sc_time(profile.duration_sec, SC_SEC));

    const unsigned pending_frames = frame_queue.num_available() + (npu.busy ? 1U : 0U);
    const unsigned pending_reviews = review_queue.num_available() + (reviewer.busy ? 1U : 0U);

    std::cout << "generated_frames=" << camera_farm.generated << '\n'
              << "candidate_trace_enabled=" << (!profile.candidate_flags.empty() ? 1 : 0) << '\n'
              << "processed_frames=" << npu.processed << '\n'
              << "pending_frames=" << pending_frames << '\n'
              << "dropped_frames=" << camera_farm.dropped << '\n'
              << "generated_candidates=" << npu.candidate_count << '\n'
              << "completed_reviews=" << reviewer.completed << '\n'
              << "pending_reviews=" << pending_reviews << '\n'
              << "dropped_candidates=" << npu.dropped_candidates << std::endl;
    return 0;
}
