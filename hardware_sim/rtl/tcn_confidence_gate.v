`timescale 1ns/1ps

module tcn_confidence_gate (
    input wire clk,
    input wire rst_n,
    input wire decision_valid,
    input wire context_reset,
    input wire [15:0] probability,
    input wire [15:0] threshold,
    output reg review_flag,
    output reg episode_pulse,
    output reg [1:0] positive_run
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            review_flag <= 1'b0;
            episode_pulse <= 1'b0;
            positive_run <= 2'd0;
        end else begin
            review_flag <= 1'b0;
            episode_pulse <= 1'b0;
            if (context_reset) begin
                positive_run <= 2'd0;
            end else if (decision_valid) begin
                if (probability < threshold) begin
                    positive_run <= 2'd0;
                end else if (positive_run == 2'd0) begin
                    positive_run <= 2'd1;
                end else begin
                    review_flag <= 1'b1;
                    episode_pulse <= (positive_run == 2'd1);
                    positive_run <= 2'd2;
                end
            end
        end
    end
endmodule
