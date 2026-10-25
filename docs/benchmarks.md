# Benchmarks

FrameLaw is built for throughput on Polars data, but benchmark numbers are
only useful with the build profile, hardware, and workload attached. This page
documents the scripts and the methodology; re-run them on your own machine
before quoting numbers.

## Methodology

- **Release builds only.** Editable installs compile Rust in debug mode.
  Run `make build-release` before benchmarking.
- **Best of N.** Each script times several validation rounds and reports the
  best wall-clock run to reduce scheduler noise.
- **In-process data.** Frames are constructed once, outside the timed loop, so
  the measurement covers validation rather than I/O.
- **Single machine.** Numbers below are indicative, not a guarantee. Memory
  bandwidth and core count dominate wide-frame workloads.

## Reproduce

```bash
make build-release

uv run --with-editable ./framelaw-py python scripts/benchmark.py
uv run --with-editable ./framelaw-py python scripts/benchmark_cross_column.py
uv run --with-editable ./framelaw-py python scripts/benchmark_temporal.py
uv run --with-editable ./framelaw-py python scripts/benchmark_nested.py
```

## Indicative results

Measured on an Apple M5 (arm64), macOS 26.6, Python 3.14, with the extension
built in release mode (`make build-release`). Treat these as
order-of-magnitude figures.

| Workload | Rows | Checks | Throughput |
|---|---:|---|---:|
| Mixed column checks | 2,000,000 | `unique`, `gt`, `lt`, `matches`, `one_of` | ~29.0M rows/s |
| Cross-column checks | 1,000,000 | `ordered`, `sum_columns`, `implies`, `determines`, `within` | ~9.5M rows/s |
| Inferred temporal bounds | 1,000,000 | inferred `gt`/`lt` on `Date` | ~201M rows/s |
| Temporal gap (`within`) | 1,000,000 | `within(..., timedelta(days=60))` | ~100M rows/s |
| Temporal sampling | 1,000,000 | generate + validate | ~30M rows/s |

The mixed workload is the one quoted in the README. The nested benchmark is
dominated by generation cost rather than validation; run it locally for
nested-specific numbers.

## What affects performance

- **Column count and check count.** Checks run in parallel across columns;
  very wide frames scale with available cores.
- **String checks.** Regex and membership checks touch every non-null value
  and cost more than numeric bounds.
- **Dataset-wide checks.** `unique`, `unique_together`, and `determines`
  maintain state across rows; they are intentionally heavier than per-row
  comparisons.
- **Chunked validation.** `chunk_rows` bounds peak memory but adds per-batch
  overhead and weakens dataset-wide checks; see
  [Lazy validation](lazy-validation.md#chunked-validation).
- **Build profile.** A debug build can be an order of magnitude slower. Never
  quote a debug-build number.

## Comparing against another tool

If you publish a comparison, pin the versions, use the same input frame, and
measure end-to-end (including any conversion the other tool needs). FrameLaw
validates native Polars memory; a tool that converts to Python objects or
another frame format pays a cost that a pure validation loop does not.
