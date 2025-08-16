# db_checks.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional, Tuple, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()
# ============================================================
# 환경 변수 & 엔진
# ============================================================

def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    환경변수를 읽어온다. required=True인데 값이 없으면 RuntimeError.
    """
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing env: {name}")
    return val  # type: ignore[return-value]


def _int_env(name: str, default: int) -> int:
    """
    정수형 환경변수. 없거나 정수 변환 실패 시 default 사용.
    """
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
    """
    LangChain tool_calls 결과에서 첫 번째 함수 인자의 JSON을 파싱.
    실패하면 None.
    """
    try:
        tool_calls = getattr(result, "additional_kwargs", {}).get("tool_calls", [])
        if not tool_calls:
            return None
        args = tool_calls[0]["function"]["arguments"]
        return json.loads(args)
    except Exception:
        return None


def _to_date(s: Optional[str]) -> Optional[date]:
    """
    ISO 포맷(YYYY-MM-DD) 문자열을 date로. None/빈문자열/실패 시 None.
    """
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _date_range_if_single(ps: Optional[str], pe: Optional[str]) -> Tuple[Optional[date], Optional[date]]:
    """
    period_start/period_end 중 하나만 있으면 단일일자 범위로 보정.
    """
    ds = _to_date(ps)
    de = _to_date(pe)
    if ds and not de:
        de = ds
    if de and not ds:
        ds = de
    return ds, de


def _market_filter(market: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    마켓 필터용 SQL 조각과 바인딩 파라미터를 반환한다.
    예) name이 'KOSPI_005930_삼성전자' 형태라면 LIKE 프리픽스로 필터링.
    실제 스키마에 맞게 변경해서 쓰면 된다.
    """
    if not market:
        return "", {}
    # 예시: KOSPI / KOSDAQ 등으로 name prefix 필터
    return " AND name LIKE :market_prefix ", {"market_prefix": f"{market}%"}


def _run_exists_query(sql: str, params: Dict[str, Any]) -> bool:
    """
    SELECT 1 ... LIMIT 1 형태의 존재성 확인 쿼리를 실행하고 결과 유무를 반환.
    """
    with ENGINE.connect() as conn:
        r = conn.execute(text(sql), params).first()
        return r is not None


def _exists_price_on(target: date, market: Optional[str] = None) -> bool:
    """
    특정 거래일 데이터가 존재하는지 확인.
    """
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


def _exists_price_between(d1: date, d2: date, market: Optional[str] = None) -> bool:
    """
    거래일 범위에 데이터가 존재하는지 확인.
    """
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


def _has_window_coverage(end_date: date, window_days: int, market: Optional[str] = None) -> bool:
    """
    이동평균/RSI 등 지표 계산을 위해 end_date 포함 과거 window_days 만큼의
    데이터가 '시장 내 최소 한 종목' 기준으로 충분히 존재하는지 느슨하게 확인.
    (정밀 체크가 필요하면 티커별 존재성 검사로 확장)
    """
    # 영업일 공휴일 보정 여유(2배)로 느슨하게 범위 확대
    start_date = end_date - timedelta(days=window_days * 2)

    mf_sql, mf_params = _market_filter(market)
    sql = f"""
        SELECT COUNT(*) AS cnt
        FROM stock_prices
        WHERE trade_date BETWEEN :s AND :e
        {mf_sql}
    """
    params = {"s": start_date, "e": end_date, **mf_params}

    with ENGINE.connect() as conn:
        cnt = conn.execute(text(sql), params).scalar() or 0
    return cnt > 0


def table_exists(table_name: str) -> bool:
    """
    DB에 특정 테이블 존재 여부 확인.
    """
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
# (여기서는 Pydantic 모델 대신 duck-typing: task.date 등 속성을 가진 객체 가정)
# ============================================================

def _clause_key(cl: Any) -> Optional[str]:
    """
    clause가 문자열/딕셔너리 어느 형태여도 키를 추출.
    """
    if isinstance(cl, dict):
        return cl.get("type") or cl.get("name")
    if isinstance(cl, str):
        return cl
    return None


REQUIRED_WINDOWS: dict[str, int] = {
    "SMA_20": 40,
    "RSI_14": 30,
    "VOL_SPIKE_20": 40,
    # 필요 시 여기에 계속 추가
}


def check_task1(task: Any) -> tuple[bool, Optional[str], dict]:
    """
    Task1: 특정 날짜 단순 조회/요약
    - 요구: 해당 date에 가격 데이터 존재
    """
    ds = _to_date(getattr(task, "date", None))
    if not ds:
        return False, "날짜(date)가 비어 있습니다.", {"need": "date"}

    if not _exists_price_on(ds, getattr(task, "market", None)):
        return False, f"{ds} 기준 가격 데이터가 없습니다.", {"missing": "price", "date": getattr(task, "date", None)}

    return True, None, {"date": getattr(task, "date", None)}


def check_task2(task: Any) -> tuple[bool, Optional[str], dict]:
    """
    Task2: 조건 검색 (이동평균/RSI/거래량 급증 등)
    - 요구: 조건별 최소 윈도우 커버리지 확보
    """
    ds = _to_date(getattr(task, "date", None))
    if not ds:
        return False, "날짜(date)가 비어 있습니다.", {"need": "date"}

    min_window = 0
    for cl in getattr(task, "clauses", []) or []:
        key = _clause_key(cl)
        need = REQUIRED_WINDOWS.get(key or "", 30)
        min_window = max(min_window, need)

    # 지표가 한 개도 없더라도 기본 20일 정도는 확보되도록
    min_req = max(min_window, 20)

    if not _has_window_coverage(ds, min_req, getattr(task, "market", None)):
        return False, f"{ds} 기준 지표 계산에 필요한 과거 데이터가 부족합니다.", {
            "missing": "coverage",
            "min_window_days": min_req,
        }

    return True, None, {"date": getattr(task, "date", None)}


def check_task3(task: Any) -> tuple[bool, Optional[str], dict]:
    """
    Task3: 구간/패턴/신호 기반 탐색
    - 요구: 기간 내 가격 데이터 존재 (+ 필요 시 신호 테이블/지표 사전 계산 확인)
    - 예: "2025-02-17에 거래량이 20일 평균 대비 300% 이상 급증한 종목"
      -> 단일일자 범위, VOL_SPIKE_20 필요 등
    """
    ds, de = _date_range_if_single(
        getattr(task, "period_start", None),
        getattr(task, "period_end", None),
    )
    if not ds or not de:
        return False, "기간(period_start/period_end)이 비었습니다.", {"need": "period"}

    if not _exists_price_between(ds, de, getattr(task, "market", None)):
        return False, f"{ds}~{de} 구간의 가격 데이터가 없습니다.", {"missing": "price_range"}

    # 신호테이블 의존 시 아래 사용
    # if "VOL_SPIKE_20" in (getattr(task, "signal_type", []) or []):
    #     if not table_exists("stock_indicators"):
    #         return False, "지표 테이블(stock_indicators)이 없습니다.", {"missing": "table:stock_indicators"}

    return True, None, {
        "period_start": getattr(task, "period_start", ds.isoformat()),
        "period_end": getattr(task, "period_end", de.isoformat()),
    }