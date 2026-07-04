from illegal_parking.accidents import (
    AccidentRecord,
    AccidentRiskImpactConfig,
    estimate_accident_risk_impact,
    load_a1_a2_accident_records,
    parse_casualties,
    summarize_accident_hotspots,
)


def test_parse_casualties_extracts_deaths_and_injuries():
    assert parse_casualties("死亡1;受傷3") == (1, 3)
    assert parse_casualties("死亡0;受傷0") == (0, 0)
    assert parse_casualties("") == (0, 0)


def test_summarize_accident_hotspots_deduplicates_and_scores_risk():
    records = [
        AccidentRecord(
            accident_id="a1",
            category="A1",
            date="20240101",
            time="080000",
            city="桃園市",
            area="八德區",
            location="桃園市八德區介壽路口",
            hour=8,
            cause="違反號誌",
            longitude=121.30151,
            latitude=24.97471,
            deaths=1,
            injuries=1,
        ),
        AccidentRecord(
            accident_id="a2",
            category="A2",
            date="20240102",
            time="090000",
            city="桃園市",
            area="八德區",
            location="桃園市八德區介壽路口",
            hour=9,
            cause="未注意車前狀況",
            longitude=121.30154,
            latitude=24.97474,
            deaths=0,
            injuries=1,
        ),
        AccidentRecord(
            accident_id="b1",
            category="A2",
            date="20240103",
            time="220000",
            city="臺中市",
            area="北屯區",
            location="臺中市北屯區經貿東路",
            hour=22,
            cause="未保持距離",
            longitude=120.65903,
            latitude=24.18709,
            deaths=0,
            injuries=2,
        ),
    ]

    hotspots = summarize_accident_hotspots(records, top_n=2, grid_precision=3, a1_weight=5, a2_weight=1)

    assert hotspots[0].city == "桃園市"
    assert hotspots[0].grid_id == "24.975,121.302"
    assert hotspots[0].a1_count == 1
    assert hotspots[0].a2_count == 1
    assert hotspots[0].risk_score == 6
    assert hotspots[0].peak_hour == 8
    assert hotspots[1].city == "臺中市"


def test_estimate_accident_risk_impact_reduces_high_risk_hotspots():
    records = [
        AccidentRecord(
            accident_id="a1",
            category="A1",
            date="20240101",
            time="080000",
            city="桃園市",
            area="八德區",
            location="桃園市八德區介壽路口",
            hour=8,
            cause="違反號誌",
            longitude=121.30151,
            latitude=24.97471,
            deaths=1,
            injuries=0,
        ),
        AccidentRecord(
            accident_id="b1",
            category="A2",
            date="20240102",
            time="090000",
            city="臺中市",
            area="北屯區",
            location="臺中市北屯區經貿東路",
            hour=9,
            cause="未保持距離",
            longitude=120.65903,
            latitude=24.18709,
            deaths=0,
            injuries=1,
        ),
    ]
    hotspots = summarize_accident_hotspots(records, top_n=2, grid_precision=3, a1_weight=5, a2_weight=1)

    impact = estimate_accident_risk_impact(
        hotspots,
        config=AccidentRiskImpactConfig(high_risk_reduction_rate=0.2),
    )

    assert impact.baseline_risk_score == 6
    assert impact.projected_risk_score == 5
    assert impact.reduced_risk_score == 1
    assert impact.baseline_cases == 2
    assert impact.projected_cases == 1.8


def test_load_a1_a2_accident_records_deduplicates_party_rows(tmp_path):
    from zipfile import ZipFile

    zip_path = tmp_path / "accidents.zip"
    rows = (
        "發生年度,發生月份,發生日期,發生時間,事故類別名稱,處理單位名稱警局層,發生地點,"
        "死亡受傷人數,肇因研判子類別名稱-主要,經度,緯度\n"
        "2024,1,20240101,080000,A1,桃園市政府警察局,桃園市八德區介壽路口,"
        "死亡1;受傷1,違反號誌,121.30151,24.97471\n"
        "2024,1,20240101,080000,A1,桃園市政府警察局,桃園市八德區介壽路口,"
        "死亡1;受傷1,違反號誌,121.30151,24.97471\n"
        "2024,1,20240102,090000,A2,桃園市政府警察局,桃園市八德區介壽路口,"
        "死亡0;受傷1,未注意車前狀況,121.30154,24.97474\n"
    )
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("manifest.csv", "")
        archive.writestr("113年度A1交通事故資料.csv", rows)

    records = load_a1_a2_accident_records(zip_path)

    assert len(records) == 2
    assert records[0].city == "桃園市"
    assert records[0].area == "八德區"
    assert records[0].hour == 8
