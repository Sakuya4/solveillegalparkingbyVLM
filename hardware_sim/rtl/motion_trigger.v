`timescale 1ns/1ps

module motion_trigger #(
    parameter integer SCORE_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire frame_valid,
    input  wire [SCORE_WIDTH-1:0] global_motion,
    input  wire [SCORE_WIDTH-1:0] roi_motion,
    input  wire [SCORE_WIDTH-1:0] min_roi_motion,
    input  wire [SCORE_WIDTH-1:0] roi_margin,
    input  wire ack_event,
    output reg  incident_trigger,
    output reg  [1:0] state
);

    localparam [1:0] ARMED   = 2'd0;
    localparam [1:0] LATCHED = 2'd1;
    localparam [1:0] REARM   = 2'd2;

    wire [SCORE_WIDTH:0] global_with_margin = {1'b0, global_motion} + {1'b0, roi_margin};
    wire spike_condition = frame_valid
        && (roi_motion >= min_roi_motion)
        && ({1'b0, roi_motion} >= global_with_margin);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            incident_trigger <= 1'b0;
            state <= ARMED;
        end else begin
            incident_trigger <= 1'b0;
            case (state)
                ARMED: begin
                    if (spike_condition) begin
                        incident_trigger <= 1'b1;
                        state <= LATCHED;
                    end
                end
                LATCHED: begin
                    if (ack_event) begin
                        state <= REARM;
                    end
                end
                REARM: begin
                    if (!spike_condition) begin
                        state <= ARMED;
                    end
                end
                default: state <= ARMED;
            endcase
        end
    end

endmodule
