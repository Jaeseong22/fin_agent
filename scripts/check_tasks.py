#!/usr/bin/env python3
from types import SimpleNamespace
from datetime import date
import sys
from pathlib import Path

# ensure repo root on sys.path so `import graph.utils` works
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "graph"))

import graph.utils as gu

# pick a recent date from DB range
sample_date = '2026-06-19'

t1 = SimpleNamespace(date=sample_date, market='KOSPI', clauses=None)
print('check_task1 (single date, KOSPI):', gu.check_task1(t1))

# Task2: clauses required; test with a dummy clause for a date we have
t2 = SimpleNamespace(date=sample_date, market='KOSDAQ', clauses=[{'type':'price_range'}])
print('check_task2 (single date, KOSDAQ, clauses):', gu.check_task2(t2))

# Task3: period range and signal_type
period_start = '2026-06-01'
period_end = '2026-06-19'
t3 = SimpleNamespace(period_start=period_start, period_end=period_end, market=['KOSPI','KOSDAQ'], signal_type={'type':'rsi'})
print('check_task3 (range with rsi signal):', gu.check_task3(t3))

# Additional: check table existence and sample latest market_caps
print('table_exists stock_prices:', gu.table_exists('stock_prices'))
print('table_exists market_cap_stage:', gu.table_exists('market_cap_stage'))
