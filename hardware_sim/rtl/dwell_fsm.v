`timescale 1ns/1ps

module dwell_fsm #(
    parameter integer DWELL_THRESHOLD_CYCLES = 30,
    parameter integer COUNT_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire track_valid,
    input  wire restricted_zone,
    input  wire stationary,
    input  wire ack_event,
    output reg  candidate_event,
    output reg  [1:0] state,
    output reg  [COUNT_WIDTH-1:0] dwell_count
);

    localparam [1:0] IDLE       = 2'd0;
    localparam [1:0] OBSERVING  = 2'd1;
    localparam [1:0] CANDIDATE  = 2'd2;
    localparam [1:0] EVENT_SENT = 2'd3;

    wire violation_condition = track_valid && restricted_zone && stationary;
    wire threshold_met = dwell_count >= (DWELL_THRESHOLD_CYCLES - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            dwell_count <= {COUNT_WIDTH{1'b0}};
            candidate_event <= 1'b0;
        end else begin
            candidate_event <= 1'b0;

            case (state)
                IDLE: begin
                    dwell_count <= {COUNT_WIDTH{1'b0}};
                    if (violation_condition) begin
                        state <= OBSERVING;
                        dwell_count <= {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                OBSERVING: begin
                    if (!violation_condition) begin
                        state <= IDLE;
                        dwell_count <= {COUNT_WIDTH{1'b0}};
                    end else if (threshold_met) begin
                        state <= CANDIDATE;
                        candidate_event <= 1'b1;
                    end else begin
                        dwell_count <= dwell_count + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                    end
                end

                CANDIDATE: begin
                    candidate_event <= 1'b1;
                    if (ack_event) begin
                        state <= EVENT_SENT;
                    end
                end

                EVENT_SENT: begin
                    if (!violation_condition) begin
                        state <= IDLE;
                        dwell_count <= {COUNT_WIDTH{1'b0}};
                    end
                end

                default: begin
                    state <= IDLE;
                    dwell_count <= {COUNT_WIDTH{1'b0}};
                end
            endcase
        end
    end

endmodule
