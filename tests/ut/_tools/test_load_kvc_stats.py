from tools.load_kvc_stats import parse_lines, summarize


def test_parse_exists_timing_for_all_backends():
    records = parse_lines(
        [
            "Yuanrong exists took 1.250 ms, keys=2",
            "MooncakeBackend exists took 2.500 ms, keys=3",
            "Memcache exists took 3.750 ms, keys=4",
        ]
    )

    assert records == {
        "Yuanrong/exists": [{"elapsed_ms": 1.25}],
        "Mooncake/exists": [{"elapsed_ms": 2.5}],
        "Memcache/exists": [{"elapsed_ms": 3.75}],
    }


def test_exists_summary_has_latency_without_transfer_throughput():
    summary = summarize([{"elapsed_ms": 1.0}, {"elapsed_ms": 3.0}])

    assert summary["count"] == 2
    assert summary["avg_ms"] == 2.0
    assert summary["total_mb"] == 0.0
    assert summary["throughput_mb_s"] == 0.0
