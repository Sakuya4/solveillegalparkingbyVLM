`timescale 1ns/1ps

module bbox_overlap_counter_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg frame_start = 1'b0;
    reg pixel_valid = 1'b0;
    reg in_bbox_band = 1'b0;
    reg is_red_pixel = 1'b0;
    reg frame_done = 1'b0;
    reg [7:0] min_red_pixels = 8'd3;
    wire [7:0] band_pixels;
    wire [7:0] red_pixels;
    wire overlap_hit;

    bbox_overlap_counter #(
        .COUNT_WIDTH(8),
        .RATIO_WIDTH(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .pixel_valid(pixel_valid),
        .in_bbox_band(in_bbox_band),
        .is_red_pixel(is_red_pixel),
        .frame_done(frame_done),
        .min_red_pixels(min_red_pixels),
        .band_pixels(band_pixels),
        .red_pixels(red_pixels),
        .overlap_hit(overlap_hit)
    );

    always #5 clk = ~clk;

    task push_pixel;
        input band;
        input red;
        begin
            in_bbox_band = band;
            is_red_pixel = red;
            pixel_valid = 1'b1;
            @(posedge clk);
            #1;
            pixel_valid = 1'b0;
        end
    endtask

    initial begin
        #12 rst_n = 1'b1;
        frame_start = 1'b1;
        @(posedge clk);
        #1 frame_start = 1'b0;

        push_pixel(1'b1, 1'b1);
        push_pixel(1'b1, 1'b0);
        push_pixel(1'b1, 1'b1);
        push_pixel(1'b0, 1'b1);
        push_pixel(1'b1, 1'b1);

        frame_done = 1'b1;
        @(posedge clk);
        #1 frame_done = 1'b0;

        if (band_pixels !== 8'd4 || red_pixels !== 8'd3 || overlap_hit !== 1'b1) begin
            $display("FAIL: overlap counter mismatch");
            $finish;
        end

        $display("PASS: bbox_overlap_counter_tb");
        $finish;
    end
endmodule
