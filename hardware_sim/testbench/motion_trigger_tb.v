`timescale 1ns/1ps

module motion_trigger_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg frame_valid = 0;
    reg [15:0] global_motion = 0;
    reg [15:0] roi_motion = 0;
    reg [15:0] min_roi_motion = 200;
    reg [15:0] roi_margin = 80;
    reg ack_event = 0;
    wire incident_trigger;
    wire [1:0] state;

    motion_trigger dut (
        .clk(clk),
        .rst_n(rst_n),
        .frame_valid(frame_valid),
        .global_motion(global_motion),
        .roi_motion(roi_motion),
        .min_roi_motion(min_roi_motion),
        .roi_margin(roi_margin),
        .ack_event(ack_event),
        .incident_trigger(incident_trigger),
        .state(state)
    );

    always #5 clk = ~clk;

    task apply_step;
        input [15:0] global_value;
        input [15:0] roi_value;
        input ack_value;
        input expected_trigger;
        begin
            @(negedge clk);
            frame_valid = 1;
            global_motion = global_value;
            roi_motion = roi_value;
            ack_event = ack_value;
            @(posedge clk);
            #1;
            if (incident_trigger !== expected_trigger) begin
                $display("FAIL global=%0d roi=%0d expected=%0d actual=%0d", global_value, roi_value, expected_trigger, incident_trigger);
                $fatal(1);
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1;

        apply_step(100, 120, 0, 0);
        apply_step(100, 300, 0, 1);
        apply_step(90, 320, 0, 0);
        apply_step(90, 310, 1, 0);
        apply_step(80, 100, 0, 0);
        apply_step(100, 300, 0, 1);

        $display("PASS motion_trigger_tb");
        $finish;
    end
endmodule
