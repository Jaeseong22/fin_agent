from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote_plus

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert


load_dotenv()


@dataclass(frozen=True)
class UniverseItem:
    code: str
    official_name: str
    market: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load 6 months of Korean stock prices from yfinance into MySQL.")
    parser.add_argument("--universe-csv", type=Path, required=True, help="CSV with 종목코드,종목명,시장구분 columns")
    parser.add_argument("--truncate", action="store_true", help="Delete existing rows before loading")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols to ingest")
    parser.add_argument("--skip-market-cap", action="store_true", help="Skip market_cap_stage and market_cap update")
    return parser.parse_args()


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = __import__("os").getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing env: {name}")
    return value  # type: ignore[return-value]


def make_engine():
    user = quote_plus(_env("MYSQL_USER", required=True))
    password = quote_plus(_env("MYSQL_PASSWORD", required=True))
    host = _env("MYSQL_HOST", "localhost")
    port = _env("MYSQL_PORT", "3306")
    database = _env("MYSQL_DATABASE", required=True)
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    return create_engine(url, future=True, pool_pre_ping=True)


def ensure_tables(engine) -> None:
    stock_prices_ddl = """
    CREATE TABLE IF NOT EXISTS stock_prices (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        official_name VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adj_close DOUBLE,
        volume BIGINT UNSIGNED,
        market_cap BIGINT,
        UNIQUE KEY uq_stock_date (name, trade_date),
        INDEX idx_trade_date (trade_date),
        INDEX idx_name (name),
        INDEX idx_official_name (official_name),
        INDEX idx_market_cap (market_cap)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    stage_ddl = """
    CREATE TABLE IF NOT EXISTS market_cap_stage (
        name VARCHAR(100) NOT NULL,
        trade_date DATE NOT NULL,
        market_cap BIGINT,
        UNIQUE KEY uq_stage (name, trade_date),
        INDEX idx_stage_name_date (name, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    with engine.begin() as conn:
        conn.execute(text(stock_prices_ddl))
        conn.execute(text(stage_ddl))


def load_universe(csv_path: Path) -> list[UniverseItem]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe CSV not found: {csv_path}")

    items: list[UniverseItem] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("종목코드") or row.get("code") or "").strip()
            name = (row.get("종목명") or row.get("official_name") or row.get("name") or "").strip()
            market = (row.get("시장구분") or row.get("market") or "").strip()
            if not code or not name:
                continue
            items.append(UniverseItem(code=code, official_name=name, market=market))
    return items


def to_yfinance_symbol(item: UniverseItem) -> str:
    code = item.code.strip()
    if code.endswith(".KS") or code.endswith(".KQ"):
        return code
    market = item.market.upper()
    if market == "KOSDAQ":
        return f"{code}.KQ"
    return f"{code}.KS"


def history_to_rows(symbol: str, item: UniverseItem) -> tuple[list[dict], Optional[int]]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="6mo", interval="1d", auto_adjust=False, actions=False)
    if history.empty:
        return [], None

    shares_outstanding = None
    try:
        info = ticker.fast_info
        shares_outstanding = info.get("shares") or info.get("shares_outstanding")
    except Exception:
        shares_outstanding = None

    if shares_outstanding is None:
        try:
            info = ticker.info
            shares_outstanding = info.get("sharesOutstanding") or info.get("shares_outstanding")
        except Exception:
            shares_outstanding = None

    rows: list[dict] = []
    for index, row in history.iterrows():
        trade_date = pd.Timestamp(index).date()
        close_value = row.get("Close")
        market_cap = None
        if shares_outstanding and pd.notna(close_value):
            market_cap = int(float(close_value) * float(shares_outstanding))

        rows.append(
            {
                "name": f"{item.code}_{item.official_name}",
                "official_name": item.official_name,
                "trade_date": trade_date,
                "open": None if pd.isna(row.get("Open")) else float(row.get("Open")),
                "high": None if pd.isna(row.get("High")) else float(row.get("High")),
                "low": None if pd.isna(row.get("Low")) else float(row.get("Low")),
                "close": None if pd.isna(row.get("Close")) else float(row.get("Close")),
                "adj_close": None if pd.isna(row.get("Adj Close")) else float(row.get("Adj Close")),
                "volume": None if pd.isna(row.get("Volume")) else int(float(row.get("Volume"))),
                "market_cap": market_cap,
            }
        )

    return rows, shares_outstanding


def upsert_rows(engine, rows: list[dict]) -> None:
    if not rows:
        return
    table_name = "stock_prices"
    stmt = mysql_insert(__import__("sqlalchemy").Table(table_name, __import__("sqlalchemy").MetaData(), autoload_with=engine))
    stmt = stmt.on_duplicate_key_update(
        official_name=stmt.inserted.official_name,
        open=stmt.inserted.open,
        high=stmt.inserted.high,
        low=stmt.inserted.low,
        close=stmt.inserted.close,
        adj_close=stmt.inserted.adj_close,
        volume=stmt.inserted.volume,
        market_cap=stmt.inserted.market_cap,
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def upsert_market_cap_stage(engine, rows: list[dict]) -> None:
    if not rows:
        return
    table_name = "market_cap_stage"
    stmt = mysql_insert(__import__("sqlalchemy").Table(table_name, __import__("sqlalchemy").MetaData(), autoload_with=engine))
    stmt = stmt.on_duplicate_key_update(market_cap=stmt.inserted.market_cap)
    payload = [{"name": row["name"], "trade_date": row["trade_date"], "market_cap": row["market_cap"]} for row in rows]
    with engine.begin() as conn:
        conn.execute(stmt, payload)


def update_market_cap(engine) -> None:
    sql = text(
        """
        UPDATE stock_prices s
        JOIN market_cap_stage m
          ON s.name = m.name
         AND s.trade_date = m.trade_date
        SET s.market_cap = m.market_cap
        """
    )
    with engine.begin() as conn:
        conn.execute(sql)


def truncate_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM market_cap_stage"))
        conn.execute(text("DELETE FROM stock_prices"))


def main() -> int:
    args = parse_args()
    engine = make_engine()
    ensure_tables(engine)

    if args.truncate:
        truncate_tables(engine)

    universe = load_universe(args.universe_csv)
    if args.limit is not None:
        universe = universe[: args.limit]

    if not universe:
        print("No universe rows found.", file=sys.stderr)
        return 1

    loaded = 0
    stage_rows: list[dict] = []

    for item in universe:
        symbol = to_yfinance_symbol(item)
        try:
            rows, shares_outstanding = history_to_rows(symbol, item)
        except Exception as exc:
            print(f"[skip] {symbol}: {exc}")
            continue

        if not rows:
            print(f"[skip] {symbol}: no history")
            continue

        upsert_rows(engine, rows)
        if not args.skip_market_cap:
            stage_rows.extend(
                {
                    "name": row["name"],
                    "trade_date": row["trade_date"],
                    "market_cap": row["market_cap"],
                }
                for row in rows
                if row["market_cap"] is not None
            )

        loaded += 1
        print(f"[ok] {symbol}: {len(rows)} rows")

    if not args.skip_market_cap and stage_rows:
        upsert_market_cap_stage(engine, stage_rows)
        update_market_cap(engine)

    print(f"done: loaded {loaded} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())