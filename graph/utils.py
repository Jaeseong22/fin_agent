from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional, Tuple, Dict, List, Union

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# ============================================================
# 환경 변수 & 엔진
# ============================================================

def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing env: {name}")
    return val  # type: ignore[return-value]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


USER = _env("MYSQL_USER", required=True)
PASS = quote_plus(_env("MYSQL_PASSWORD", required=True))
HOST = _env("MYSQL_HOST", "localhost")
PORT = _int_env("MYSQL_PORT", 3306)
DB   = _env("MYSQL_DATABASE", required=True)

ENGINE: Engine = create_engine(
    f"mysql+pymysql://{USER}:{PASS}@{HOST}:{PORT}/{DB}",
    future=True,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# ============================================================
# 공통 유틸
# ============================================================

def parse_tool_json(result: Any) -> Optional[dict]:
    """LangChain tool_calls 결과에서 첫 번째 함수 인자의 JSON을 파싱. 실패하면 None."""
    try:
        tool_calls = getattr(result, "additional_kwargs", {}).get("tool_calls", [])
        if not tool_calls:
            return None
        args = tool_calls[0]["function"]["arguments"]
        return json.loads(args)
    except Exception:
        return None


def _to_date(s: Optional[Union[str, date]]) -> Optional[date]:
    """YYYY-MM-DD 문자열/ date / None / 빈문자열 모두 안전 처리."""
    if s is None:
        return None
    if isinstance(s, date):
        return s
    s = s.strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _date_range_if_single(ps: Optional[str], pe: Optional[str]) -> Tuple[Optional[date], Optional[date]]:
    """
    period_start/period_end 중 하나만 있으면 단일일자 범위로 보정.
    둘 다 없으면 (None, None).
    """
    ds = _to_date(ps)
    de = _to_date(pe)
    if ds and not de:
        de = ds
    if de and not ds:
        ds = de
    return ds, de


def _normalize_market(market: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """
    market이 None / 'KOSPI' / 'KOSDAQ' / ['KOSPI', 'KOSDAQ'] 등으로 들어와도
    일관되게 ['KOSPI', ...] 형태로 반환. 유효 값만 유지. 결과가 빈 리스트면 None.
    """
    if market is None:
        return None
    valid = {"KOSPI", "KOSDAQ"}
    if isinstance(market, str):
        return [market] if market in valid else None
    if isinstance(market, list):
        vals = [m for m in market if isinstance(m, str) and m in valid]
        return vals or None
    return None


def _market_filter(market: Optional[Union[str, List[str]]]) -> Tuple[str, Dict[str, Any]]:
    """
    name 컬럼이 'KOSPI_...'처럼 시장 프리픽스라 가정하고 LIKE 필터 생성.
    단일/다중 시장 지원.
    """
    markets = _normalize_market(market)
    if not markets:
        return "", {}

    # 동적 OR 생성: (name LIKE :m0 OR name LIKE :m1 ...)
    conds = []
    params: Dict[str, Any] = {}
    for i, m in enumerate(markets):
        key = f"m{i}"
        conds.append(f"name LIKE :{key}")
        params[key] = f"{m}%"
    sql = " AND (" + " OR ".join(conds) + ") "
    return sql, params


def _run_exists_query(sql: str, params: Dict[str, Any]) -> bool:
    """SELECT 1 ... LIMIT 1 형태의 존재성 확인 쿼리."""
    with ENGINE.connect() as conn:
        r = conn.execute(text(sql), params).first()
        return r is not None


def _exists_price_on(target: date, market: Optional[Union[str, List[str]]] = None) -> bool:
    """특정 거래일 데이터 존재 여부."""
    mf_sql, mf_params = _market_filter(market)
    sql = f"""
        SELECT 1
        FROM stock_prices
        WHERE trade_date = :d
        {mf_sql}
        LIMIT 1
    """
    params = {"d": target, **mf_params}
    return _run_exists_query(sql, params)


def _exists_price_between(d1: date, d2: date, market: Optional[Union[str, List[str]]] = None) -> bool:
    """거래일 범위 데이터 존재 여부."""
    mf_sql, mf_params = _market_filter(market)
    sql = f"""
        SELECT 1
        FROM stock_prices
        WHERE trade_date BETWEEN :d1 AND :d2
        {mf_sql}
        LIMIT 1
    """
    params = {"d1": d1, "d2": d2, **mf_params}
    return _run_exists_query(sql, params)


def _count_rows_between(d1: date, d2: date, market: Optional[Union[str, List[str]]] = None) -> int:
    """범위 내 전체 행 수(느슨한 커버리지 판단용)."""
    mf_sql, mf_params = _market_filter(market)
    sql = f"""
        SELECT COUNT(*) AS cnt
        FROM stock_prices
        WHERE trade_date BETWEEN :s AND :e
        {mf_sql}
    """
    params = {"s": d1, "e": d2, **mf_params}
    with ENGINE.connect() as conn:
        return int(conn.execute(text(sql), params).scalar() or 0)


def _has_window_coverage(end_date: date, window_days: int, market: Optional[Union[str, List[str]]] = None) -> bool:
    """
    지표 계산용 커버리지 느슨 확인.
    버퍼 포함하여 end_date-2*window_days ~ end_date 사이에 데이터가 최소 존재하면 OK.
    """
    start_date = end_date - timedelta(days=window_days * 2)
    return _count_rows_between(start_date, end_date, market) > 0


def table_exists(table_name: str) -> bool:
    """DB에 특정 테이블 존재 여부."""
    sql = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :db AND table_name = :t
        LIMIT 1
    """
    with ENGINE.connect() as conn:
        r = conn.execute(text(sql), {"db": DB, "t": table_name}).first()
        return r is not None

# ============================================================
# Task 체크 로직
# ============================================================

def _clause_key(cl: Any) -> Optional[str]:
    """clause가 문자열/딕셔너리 어느 형태여도 키(type/name)를 추출."""
    if isinstance(cl, dict):
        return cl.get("type") or cl.get("name")
    if isinstance(cl, str):
        return cl
    return None


# Task3 신호별 요구 윈도우(느슨 기준)
# - moving_average_diff: period의 2~3배 정도 여유
# - rsi: 30일 정도
# - volume_signal: days가 있으면 2*days, 없으면 40
# - cross_events: 60
# - bollinger_band: 40
def _task3_required_window(signals: Optional[Union[dict, List[dict]]]) -> int:
    if signals is None:
        return 0
    if isinstance(signals, dict):
        signals = [signals]

    max_need = 0
    for sig in signals:
        t = (sig or {}).get("type")
        if t == "moving_average_diff":
            period = int((sig or {}).get("period") or 20)
            need = max(40, period * 2)
        elif t == "rsi":
            need = 30
        elif t == "volume_signal":
            days = int((sig or {}).get("days") or 20)
            need = max(40, days * 2)
        elif t == "cross_events":
            need = 60
        elif t == "bollinger_band":
            need = 40
        else:
            need = 0
        max_need = max(max_need, need)
    return max_need

def check_task1(task: Any) -> bool:
    """
    Task1: 특정 날짜 단순 조회/요약
    요구사항: 제공된 날짜 중 최소 1일 이상 데이터 존재.
    """
    raw_date = getattr(task, "date", None)

    dates: List[date] = []
    if isinstance(raw_date, list):
        dates = [d for d in (_to_date(x) for x in raw_date) if d]
    else:
        d = _to_date(raw_date)
        dates = [d] if d else []

    if not dates:
        return False

    market = getattr(task, "market", None)
    return any(_exists_price_on(d, market) for d in dates)


def check_task2(task: Any) -> bool:
    """
    Task2: 조건 검색 (pct/vol_pct/vol_abs/price_range).
    - clauses는 반드시 최소 1개 있어야 함(None/빈 리스트 불가).
    - date 유효 + 해당 일자에 데이터 존재.
    """
    clauses = getattr(task, "clauses", None)
    if not clauses:
        return False

    ds = _to_date(getattr(task, "date", None))
    if not ds:
        return False

    market = getattr(task, "market", None)
    return _exists_price_on(ds, market)


def check_task3(task: Any) -> bool:
    """
    Task3: 구간/패턴/신호 기반 탐색
    - 기간 내 가격 데이터 존재
    - signal_type이 있으면 필요한 윈도우 커버리지 충족
    """
    ds, de = _date_range_if_single(
        getattr(task, "period_start", None),
        getattr(task, "period_end", None),
    )
    if not ds or not de:
        return False

    market = getattr(task, "market", None)
    if not _exists_price_between(ds, de, market):
        return False

    signals = getattr(task, "signal_type", None)
    need_window = _task3_required_window(signals)
    if need_window > 0 and not _has_window_coverage(de, need_window, market):
        return False

    return True