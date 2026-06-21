#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper for CI to load stock prices
echo "Starting CI ingest..."
python -V
python scripts/load_stock_prices.py --universe-csv data/universe.csv --truncate
echo "CI ingest finished."
