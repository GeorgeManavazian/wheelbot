"""Data loading for the bench, absorbed from the scratchpad harnesses.

Every A/B script in `scratchpad/` grew its own loader, and they disagreed: some
still called `options.data.chain_path()`, which points at
`data/options/{ticker}_eod.parquet` -- a layout with ZERO files left, replaced
by the month-partitioned `data/options/chains/{TICKER}/*.parquet` store. A
harness whose loader is subtly different from the last one's is a harness whose
result is not comparable to the last one's, which is most of why the old grid
outputs cannot be stacked against each other.

One loader. Every arm sees the same bytes.
"""
