`timescale 1ns/1ps

module dwell_fsm_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg track_valid = 1'b0;
    reg restricted_zone = 1'b0;
    reg stationary = 1'b0;
    reg ack_event = 1'b0;
    wire candidate_event;
    wire [1:0] state;
    wire [3:0] dwell_count;

    dwell_fsm #(
        .DWELL_THRESHOLD_CYCLES(4),
        .COUNT_WIDTH(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .track_valid(track_valid),
        .restricted_zone(restricted_zone),
        .stationary(stationary),
        .ack_event(ack_event),
        .candidate_event(candidate_event),
        .state(state),
        .dwell_count(dwell_count)
    );

    always #5 clk = ~clk;

    initial begin
        #12 rst_n = 1'b1;
        track_valid = 1'b1;
        restricted_zone = 1'b1;
        stationary = 1'b1;

        repeat (4) @(posedge clk);
        #1;
        if (candidate_event !== 1'b1) begin
            $display("FAIL: candidate_event did not assert at threshold");
            $finish;
        end

        ack_event = 1'b1;
        @(posedge clk);
        #1;
        ack_event = 1'b0;
        if (state !== 2'd3) begin
            $display("FAIL: FSM did not move to EVENT_SENT after ack");
            $finish;
        end

        stationary = 1'b0;
        @(posedge clk);
        #1;
        if (state !== 2'd0) begin
            $display("FAIL: FSM did not reset after condition cleared");
            $finish;
        end

        $display("PASS: dwell_fsm_tb");
        $finish;
    end
endmodule
