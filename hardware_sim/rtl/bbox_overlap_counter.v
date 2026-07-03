`timescale 1ns/1ps

module bbox_overlap_counter #(
    parameter integer COUNT_WIDTH = 16,
    parameter integer RATIO_WIDTH = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire frame_start,
    input  wire pixel_valid,
    input  wire in_bbox_band,
    input  wire is_red_pixel,
    input  wire frame_done,
    input  wire [COUNT_WIDTH-1:0] min_red_pixels,
    output reg  [COUNT_WIDTH-1:0] band_pixels,
    output reg  [COUNT_WIDTH-1:0] red_pixels,
    output reg  overlap_hit
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            band_pixels <= {COUNT_WIDTH{1'b0}};
            red_pixels <= {COUNT_WIDTH{1'b0}};
            overlap_hit <= 1'b0;
        end else begin
            if (frame_start) begin
                band_pixels <= {COUNT_WIDTH{1'b0}};
                red_pixels <= {COUNT_WIDTH{1'b0}};
                overlap_hit <= 1'b0;
            end

            if (pixel_valid && in_bbox_band) begin
                band_pixels <= band_pixels + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                if (is_red_pixel) begin
                    red_pixels <= red_pixels + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                end
            end

            if (frame_done) begin
                overlap_hit <= red_pixels >= min_red_pixels;
            end
        end
    end

endmodule
