#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import timedelta, date

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "graph"))

from graph.schema import Task3
import graph.utils as gu
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()

def make_engine():
    user = os.getenv('MYSQL_USER')
    pw = quote_plus(os.getenv('MYSQL_PASSWORD') or '')
    host = os.getenv('MYSQL_HOST','localhost')
    port = os.getenv('MYSQL_PORT','3306')
    db = os.getenv('MYSQL_DATABASE')
    return create_engine(f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4", future=True)

def get_latest_date(engine):
    with engine.connect() as conn:
        r = conn.execute(text('SELECT MAX(trade_date) FROM stock_prices')).scalar()
    return r

def period_return(engine, company_keyword, ds, de):
    like = f"%{company_keyword}%"
    sql = text("SELECT trade_date, close FROM stock_prices WHERE name LIKE :like AND trade_date BETWEEN :ds AND :de ORDER BY trade_date ASC")
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, {"like": like, "ds": ds, "de": de})]
    if not rows:
        return None
    first = rows[0]['close']
    last = rows[-1]['close']
    if first is None or last is None or first == 0:
        return None
    return (last / first - 1.0) * 100.0

def example_compare_3m(engine):
    latest = get_latest_date(engine)
    if latest is None:
        print('No data')
        return
    end = latest
    start = end - timedelta(days=90)
    a = '삼성전자'
    b = 'LG전자'
    ra = period_return(engine, a, start, end)
    rb = period_return(engine, b, start, end)
    print('=== 삼성전자 vs LG전자 (최근 3개월) ===')
    if ra is None:
        print('삼성전자 데이터 없음')
    else:
        print(f'삼성전자 수익률: {ra:.2f}% ({start}~{end})')
    if rb is None:
        print('LG전자 데이터 없음')
    else:
        print(f'LG전자 수익률: {rb:.2f}% ({start}~{end})')
    if ra is not None and rb is not None:
        if ra > rb:
            print('결론: 삼성전자가 우위')
        elif rb > ra:
            print('결론: LG전자가 우위')
        else:
            print('결론: 두 종목 수익률 동일')

def example_rsi_70(engine):
    latest = get_latest_date(engine)
    if latest is None:
        print('No data')
        return
    end = latest
    start = end - timedelta(days=90)
    task = Task3(company_name=None, market=None, period_start=start.isoformat(), period_end=end.isoformat(), signal_type=[{"type":"rsi","threshold":70,"condition":"overbought"}], mode="list")
    res = gu.run_task3_query(task, engine)
    print('=== RSI ≥ 70 종목 (최근 3개월) ===')
    print(gu._r_task3(res))

def main():
    engine = make_engine()
    example_compare_3m(engine)
    print()
    example_rsi_70(engine)

if __name__ == '__main__':
    main()
