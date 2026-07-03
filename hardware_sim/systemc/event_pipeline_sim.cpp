#include <systemc>
#include <iostream>

using namespace sc_core;
using namespace sc_dt;

struct EventPipelineSim : sc_module {
    sc_in<bool> clk;
    sc_in<bool> rst_n;
    sc_in<bool> track_valid;
    sc_in<bool> restricted_zone;
    sc_in<bool> stationary;
    sc_out<bool> candidate_event;

    unsigned dwell_count = 0;
    const unsigned threshold_cycles = 4;

    SC_CTOR(EventPipelineSim) {
        SC_METHOD(tick);
        sensitive << clk.pos();
        dont_initialize();
    }

    void tick() {
        if (!rst_n.read()) {
            dwell_count = 0;
            candidate_event.write(false);
            return;
        }

        bool condition = track_valid.read() && restricted_zone.read() && stationary.read();
        if (condition) {
            dwell_count += 1;
        } else {
            dwell_count = 0;
        }

        candidate_event.write(dwell_count >= threshold_cycles);
    }
};

int sc_main(int argc, char* argv[]) {
    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> rst_n;
    sc_signal<bool> track_valid;
    sc_signal<bool> restricted_zone;
    sc_signal<bool> stationary;
    sc_signal<bool> candidate_event;

    EventPipelineSim dut("event_pipeline");
    dut.clk(clk);
    dut.rst_n(rst_n);
    dut.track_valid(track_valid);
    dut.restricted_zone(restricted_zone);
    dut.stationary(stationary);
    dut.candidate_event(candidate_event);

    rst_n = false;
    track_valid = false;
    restricted_zone = false;
    stationary = false;
    sc_start(10, SC_NS);

    rst_n = true;
    track_valid = true;
    restricted_zone = true;
    stationary = true;

    for (int i = 0; i < 5; ++i) {
        sc_start(10, SC_NS);
        std::cout << "cycle=" << i
                  << " candidate=" << candidate_event.read()
                  << std::endl;
    }

    return 0;
}
