"""The benchmark's percentile maths (p50/p95 figures in the README come from it)."""


def test_percentiles(bench_module):
    values = [1, 2, 3, 4, 5]
    assert bench_module.percentile(values, 0.5) == 3
    assert bench_module.percentile(values, 0.95) == 4.8
    assert bench_module.percentile([], 0.5) != bench_module.percentile([], 0.5)  # NaN for no data
