from illegal_parking.hotspots import (
    EnforcementCostConfig,
    ViolationRecord,
    classify_hotspots,
    estimate_intervention_impact,
    load_taoyuan_violation_records,
    summarize_road_hotspots,
)


def test_summarize_road_hotspots_counts_and_ranks_roads():
    records = [
        ViolationRecord(city="桃園市", area="中壢區", road="A路", law="56", fact="違停", hour=8),
        ViolationRecord(city="桃園市", area="中壢區", road="A路", law="56", fact="違停", hour=9),
        ViolationRecord(city="桃園市", area="桃園區", road="B路", law="56", fact="違停", hour=8),
    ]

    hotspots = summarize_road_hotspots(records, top_n=2)

    assert [hotspot.road for hotspot in hotspots] == ["A路", "B路"]
    assert hotspots[0].violation_count == 2
    assert hotspots[0].share_of_total == 2 / 3
    assert hotspots[0].peak_hour == 8


def test_classify_hotspots_uses_rank_percentiles():
    records = [
        ViolationRecord(city="桃園市", area="A", road="R1", law="56", fact="違停", hour=8),
        ViolationRecord(city="桃園市", area="A", road="R1", law="56", fact="違停", hour=9),
        ViolationRecord(city="桃園市", area="A", road="R2", law="56", fact="違停", hour=8),
        ViolationRecord(city="桃園市", area="A", road="R3", law="56", fact="違停", hour=8),
        ViolationRecord(city="桃園市", area="A", road="R4", law="56", fact="違停", hour=8),
    ]

    hotspots = classify_hotspots(summarize_road_hotspots(records, top_n=4), high_ratio=0.25, medium_ratio=0.5)

    assert [hotspot.heat_level for hotspot in hotspots] == ["high", "medium", "medium", "low"]


def test_estimate_intervention_impact_reduces_high_heat_violations_and_costs():
    records = [
        ViolationRecord(city="桃園市", area="A", road="R1", law="56", fact="違停", hour=8),
        ViolationRecord(city="桃園市", area="A", road="R1", law="56", fact="違停", hour=9),
        ViolationRecord(city="桃園市", area="A", road="R2", law="56", fact="違停", hour=8),
    ]
    hotspots = classify_hotspots(summarize_road_hotspots(records, top_n=2), high_ratio=0.5, medium_ratio=0.5)

    impact = estimate_intervention_impact(
        hotspots,
        high_heat_reduction_rate=0.5,
        cost_config=EnforcementCostConfig(minutes_per_manual_case=10, hourly_labor_cost=600, system_monthly_cost=100),
    )

    assert impact.baseline_violations == 3
    assert impact.projected_violations == 2
    assert impact.reduced_violations == 1
    assert impact.manual_cost_before == 300
    assert impact.manual_cost_after == 200
    assert impact.net_savings == 0


def test_load_taoyuan_violation_records_parses_hhmm_time(tmp_path):
    csv_path = tmp_path / "taoyuan.csv"
    csv_path.write_text(
        "year,month,time,law,fact,CountyCode,AreaName,AreaCode,Road,latitude,longitude\n"
        "113,1,0724,56,違停,68000,大園區,68000060,大觀路,25.0,121.0\n",
        encoding="utf-8",
    )

    records = load_taoyuan_violation_records(csv_path, encoding="utf-8")

    assert records[0].hour == 7
