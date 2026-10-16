`timescale 1ns/1ps

module tcn_confidence_gate_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg decision_valid = 0;
    reg context_reset = 0;
    reg [15:0] probability = 0;
    reg [15:0] threshold = 0;
    wire review_flag;
    wire episode_pulse;
    wire [1:0] positive_run;
    integer source;
    integer fields;
    integer line_number = 0;
    integer valid_value, reset_value, probability_value, threshold_value;
    integer expected_review, expected_pulse, expected_run;
    reg [8191:0] vector_path;

    tcn_confidence_gate dut (
        .clk(clk), .rst_n(rst_n), .decision_valid(decision_valid),
        .context_reset(context_reset), .probability(probability), .threshold(threshold),
        .review_flag(review_flag), .episode_pulse(episode_pulse), .positive_run(positive_run)
    );
    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("vectors=%s", vector_path))
            $fatal(1, "A vector file is required");
        source = $fopen(vector_path, "r");
        if (!source) $fatal(1, "Cannot open vector file");
        repeat (2) @(negedge clk);
        rst_n = 1;
        while (!$feof(source)) begin
            fields = $fscanf(source, "%d %d %d %d %d %d %d\n", valid_value, reset_value,
                             probability_value, threshold_value, expected_review, expected_pulse, expected_run);
            if (fields != 7) $fatal(1, "Invalid vector at line %0d", line_number + 1);
            @(negedge clk);
            decision_valid = valid_value;
            context_reset = reset_value;
            probability = probability_value;
            threshold = threshold_value;
            @(posedge clk);
            #1;
            line_number = line_number + 1;
            if (review_flag !== expected_review[0] || episode_pulse !== expected_pulse[0] || positive_run !== expected_run[1:0])
                $fatal(1, "Mismatch at line %0d: review=%0d pulse=%0d run=%0d", line_number, review_flag, episode_pulse, positive_run);
        end
        if (line_number == 0) $fatal(1, "Empty verification stream");
        $fclose(source);
        $display("PASS tcn_confidence_gate_tb tokens=%0d", line_number);
        $finish;
    end
endmodule
