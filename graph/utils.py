from __future__ import annotations

import csv
import re
import json
import os
import difflib 
from datetime import date, timedelta
from typing import Any, Optional, Tuple, Dict, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from schema import State
from pathlib import Path

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# ============================================================
# 환경 변수 & 엔진
# ============================================================

COMPANY_CSV = Path("/Users/jaeseong/Documents/finance_agent/graph/company_name.csv")
TERMS_CSV   = Path("/Users/jaeseong/Documents/finance_agent/graph/stock_terms.csv")

_COMPANY_ALIASES: List[Dict[str, str]] = []
_TERM_RULES: List[Dict[str, str]] = []

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

def _market_filter_names_sql(alias: str, market: Optional[Union[str, List[str]]]) -> Tuple[str, Dict[str, Any]]:
    markets = _normalize_market(market)
    if not markets:
        return "", {}

    aliases = _load_company_aliases()
    # 해당 시장의 공식명+코드로 stock_prices.name 문자열 생성
    # ex) "005930_삼성전자"
    full_names = sorted({f"{a['ticker']}_{a['official']}"
                         for a in aliases if a.get("market") in markets})

    if not full_names:
        return "", {}

    params = {}
    keys = []
    for i, nm in enumerate(full_names):
        k = f"nm{i}"
        keys.append(f":{k}")
        params[k] = nm

    # ✅ stock_prices.{alias}.name 기준으로 필터링
    cond = f" AND {alias}.name IN ({', '.join(keys)}) "
    return cond, params

def _market_filter_names_sql_plain(market):
    markets = _normalize_market(market)
    if not markets:
        return "", {}

    aliases = _load_company_aliases()
    full_names = sorted({f"{a['ticker']}_{a['official']}"
                         for a in aliases if a.get("market") in markets})

    if not full_names:
        return "", {}

    params = {}
    keys = []
    for i, nm in enumerate(full_names):
        k = f"nm{i}"
        keys.append(f":{k}")
        params[k] = nm

    # ✅ stock_prices.name 기준으로 필터링
    cond = f" AND name IN ({', '.join(keys)}) "
    return cond, params

def _run_exists_query(sql: str, params: Dict[str, Any]) -> bool:
    """SELECT 1 ... LIMIT 1 형태의 존재성 확인 쿼리."""
    with ENGINE.connect() as conn:
        r = conn.execute(text(sql), params).first()
        return r is not None


def _exists_price_on(target: date, market: Optional[Union[str, List[str]]] = None) -> bool:
    """특정 거래일 데이터 존재 여부."""
    mf_sql, mf_params = _market_filter_names_sql_plain(market)
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
    mf_sql, mf_params = _market_filter_names_sql_plain(market)
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
    mf_sql, mf_params = _market_filter_names_sql_plain(market)
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
    - 단, 지수 비교 절이 섞여 있으면 시장필터는 무시(지수는 종목 CSV 기반 필터링 대상 아님)
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

    clauses = getattr(task, "clauses", None) or []
    # 지수 비교 절이 하나라도 있으면 시장필터 미적용
    has_index_clause = any(
        isinstance(c, dict) and c.get("type") == "market_index_comparison"
        for c in clauses
    )

    market = None if has_index_clause else getattr(task, "market", None)
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

def _name_like_sql(company_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not company_name:
        return "", {}
    return " AND name LIKE :nm ", {"nm": f"%{company_name}%"}

def _between_sql() -> str:
    return " trade_date BETWEEN :ds AND :de "

def _order_limit_sql(mode: str) -> str:
    # 결과 정렬/리밋 (원하면 커스텀 가능)
    # list/both에서만 적용. count는 무관.
    return " ORDER BY trade_date ASC, name ASC LIMIT 5000 " if mode in ("list", "both") else ""

def _mode_payload(mode: str, rows: List[dict]) -> Dict[str, Any]:
    if mode == "count":
        return {"count": len(rows)}
    elif mode == "list":
        return {"rows": rows}
    else:
        return {"count": len(rows), "rows": rows}


def _first_valid_date(raw_date: Optional[Union[str, List[str]]]) -> Optional[date]:
    if raw_date is None:
        return None
    if isinstance(raw_date, list):
        for s in raw_date:
            d = _to_date(s)
            if d:
                return d
        return None
    return _to_date(raw_date)

def _prev_trade_date(engine: Engine, d: date) -> Optional[date]:
    q = "SELECT MAX(trade_date) FROM stock_prices WHERE trade_date < :d"
    with engine.connect() as conn:
        prev = conn.execute(text(q), {"d": d}).scalar()
    return prev

def _pct_change_expr(alias_today="t", alias_prev="p") -> str:
    # ((오늘종가 - 전일종가) / 전일종가) * 100
    return f"(( {alias_today}.close - {alias_prev}.close ) / NULLIF({alias_prev}.close,0) * 100.0)"

def _like_company(company_name: str) -> Tuple[str, Dict[str, Any]]:
    # name 컬럼에 '회사명'이 포함되도록 검색 (예: '%삼성전자%')
    return " AND name LIKE :nm ", {"nm": f"%{company_name}%"}

def _index_names_from_indices(indices: List[str]) -> List[str]:
    mapping = {
        "KOSPI": "INDEX_KOSPI",
        "KOSDAQ": "INDEX_KOSDAQ",
    }
    out = []
    for idx in indices or []:
        name = mapping.get(idx)
        if name:
            out.append(name)
    return out

# ─────────────────────────────────────────────────────────────
# Clause 핸들러
# ─────────────────────────────────────────────────────────────

def _task1_metric_rank(engine: Engine, d: date, clause: dict, market=None) -> Dict[str, Any]:
    metric = clause["metric"]           # "등락률" | "거래량" | "시가총액" | "종가"
    rank_type = clause["rank_type"]     # "top" | "bottom"
    top_n = int(clause.get("top_n") or 10)

    order = "DESC" if rank_type == "top" else "ASC"

    if metric == "거래량":
        mk_sql, mk_params = _market_filter_names_sql_plain(market)
        q = f"""
            SELECT name, volume AS value
            FROM stock_prices
            WHERE trade_date = :d
            {mk_sql}
            ORDER BY value {order}
            LIMIT :n
        """
        params = {"d": d, "n": top_n, **mk_params}

    elif metric == "종가":
        mk_sql, mk_params = _market_filter_names_sql_plain(market)
        q = f"""
            SELECT name, close AS value
            FROM stock_prices
            WHERE trade_date = :d
            {mk_sql}
            ORDER BY value {order}
            LIMIT :n
        """
        params = {"d": d, "n": top_n, **mk_params}

    elif metric == "시가총액":
        mk_sql, mk_params = _market_filter_names_sql_plain(market)
        q = f"""
            SELECT name, market_cap AS value
            FROM stock_prices
            WHERE trade_date = :d
            {mk_sql}
              AND market_cap IS NOT NULL
            ORDER BY value {order}
            LIMIT :n
        """
        params = {"d": d, "n": top_n, **mk_params}

    elif metric == "등락률":
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return {"status": "error", "type": "metric_rank", "reason": "전일 거래일을 찾을 수 없습니다."}
        mk_sql, mk_params = _market_filter_names_sql("t", market)
        q = f"""
            SELECT t.name, {_pct_change_expr('t','p')} AS value
            FROM stock_prices t
            JOIN stock_prices p
              ON p.name = t.name AND p.trade_date = :prev_d
            WHERE t.trade_date = :d
            {mk_sql}
            ORDER BY value {order}
            LIMIT :n
        """
        params = {"d": d, "prev_d": prev_d, "n": top_n, **mk_params}

    else:
        return {"status": "unsupported", "type": "metric_rank", "reason": f"지원하지 않는 metric: {metric}"}

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(q), params)]
    return {"status": "ok", "type": "metric_rank", "metric": metric, "rank_type": rank_type,
            "date": d.isoformat(), "rows": rows}

def _task1_market_comparison(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    metric = clause["metric"]
    comparison = clause["comparison"]
    op = ">" if comparison == "above_avg" else "<"

    avg_val = None  # ← 추가

    if metric == "거래량":
        q_avg = "SELECT AVG(volume) FROM stock_prices WHERE trade_date = :d"
        with engine.connect() as conn:
            avg_val = conn.execute(text(q_avg), {"d": d}).scalar()
        if avg_val is None:
            return {"status": "error", "type": "market_comparison", "reason": "시장 평균 거래량 계산 실패"}

        q = f"""
            SELECT name, volume AS value
            FROM stock_prices
            WHERE trade_date = :d AND volume {op} :avgv
            ORDER BY value DESC
            LIMIT 200
        """
        params = {"d": d, "avgv": avg_val}

    elif metric == "등락률":
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return {"status": "error", "type": "market_comparison", "reason": "전일 거래일을 찾을 수 없습니다."}
        q_avg = f"""
            SELECT AVG({_pct_change_expr('t','p')})
            FROM stock_prices t
            JOIN stock_prices p
              ON p.name = t.name AND p.trade_date = :prev_d
            WHERE t.trade_date = :d
        """
        with engine.connect() as conn:
            avg_val = conn.execute(text(q_avg), {"d": d, "prev_d": prev_d}).scalar()
        if avg_val is None:
            return {"status": "error", "type": "market_comparison", "reason": "시장 평균 등락률 계산 실패"}

        q = f"""
            SELECT t.name, {_pct_change_expr('t','p')} AS value
            FROM stock_prices t
            JOIN stock_prices p
              ON p.name = t.name AND p.trade_date = :prev_d
            WHERE t.trade_date = :d AND {_pct_change_expr('t','p')} {op} :avgv
            ORDER BY value DESC
            LIMIT 200
        """
        params = {"d": d, "prev_d": prev_d, "avgv": avg_val}

    else:
        return {"status": "unsupported", "type": "market_comparison", "reason": f"지원하지 않는 metric: {metric}"}

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(q), params)]
    return {
        "status": "ok",
        "type": "market_comparison",
        "metric": metric,
        "comparison": comparison,
        "date": d.isoformat(),
        "market_avg": float(avg_val),
        "rows": rows
    }

def _task1_stock_comparison(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    metric = clause["metric"]  # "등락률" | "시가총액" | "거래량" | "종가"
    a = clause["stock_a"]
    b = clause["stock_b"]

    nm_sql_a = " AND t.name LIKE :nm_a "
    nm_sql_b = " AND t.name LIKE :nm_b "
    params = {"d": d, "nm_a": f"%{a}%", "nm_b": f"%{b}%", "exact_a": a, "exact_b": b}

    # 정확 일치 우선 정렬 절 (두 분기에서 공통으로 사용)
    exact_order_a = "CASE WHEN t.name = :exact_a THEN 0 ELSE 1 END"
    exact_order_b = "CASE WHEN t.name = :exact_b THEN 0 ELSE 1 END"

    if metric == "거래량":
        col = "volume"
        null_guard = ""  # volume은 대부분 NOT NULL일 테니 생략
    elif metric == "종가":
        col = "close"
        null_guard = ""
    elif metric == "시가총액":
        col = "market_cap"
        null_guard = " AND t.market_cap IS NOT NULL "
    else:
        col = None
        null_guard = ""

    if col:
        q = f"""
            SELECT * FROM (
                SELECT 'A' AS which, t.name, t.{col} AS value
                FROM stock_prices t
                WHERE t.trade_date = :d {nm_sql_a} {null_guard}
                ORDER BY {exact_order_a} ASC, t.{col} DESC
                LIMIT 1
            ) A
            UNION ALL
            SELECT * FROM (
                SELECT 'B' AS which, t.name, t.{col} AS value
                FROM stock_prices t
                WHERE t.trade_date = :d {nm_sql_b} {null_guard}
                ORDER BY {exact_order_b} ASC, t.{col} DESC
                LIMIT 1
            ) B
        """
    elif metric == "등락률":
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return {
                "status": "error",
                "type": "stock_comparison",
                "reason": "전일 거래일을 찾을 수 없습니다."
            }
        params["prev_d"] = prev_d
        pct_expr = _pct_change_expr("t", "p")
        q = f"""
            SELECT * FROM (
                SELECT 'A' AS which, t.name, {pct_expr} AS value
                FROM stock_prices t
                JOIN stock_prices p
                  ON p.name = t.name AND p.trade_date = :prev_d
                WHERE t.trade_date = :d {nm_sql_a}
                ORDER BY {exact_order_a} ASC, value DESC
                LIMIT 1
            ) A
            UNION ALL
            SELECT * FROM (
                SELECT 'B' AS which, t.name, {pct_expr} AS value
                FROM stock_prices t
                JOIN stock_prices p
                  ON p.name = t.name AND p.trade_date = :prev_d
                WHERE t.trade_date = :d {nm_sql_b}
                ORDER BY {exact_order_b} ASC, value DESC
                LIMIT 1
            ) B
        """
    else:
        return {
            "status": "unsupported",
            "type": "stock_comparison",
            "reason": f"지원하지 않는 metric: {metric}"
        }

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(q), params)]

    # 승자 계산 (값이 큰 쪽이 우위)
    a_val = next((r["value"] for r in rows if r.get("which") == "A"), None)
    b_val = next((r["value"] for r in rows if r.get("which") == "B"), None)
    winner = None
    if a_val is not None and b_val is not None:
        if a_val > b_val:
            winner = "A"
        elif b_val > a_val:
            winner = "B"
        else:
            winner = "tie"

    return {
        "status": "ok",
        "type": "stock_comparison",
        "metric": metric,
        "date": d.isoformat(),
        "stock_a": a,
        "stock_b": b,
        "rows": rows,
        "winner": winner
    }

def _task1_simple_lookup(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    company = clause["company_name"]
    metric  = clause["metric"]

    nm_sql, nm_param = _like_company(company)

    if metric == "시가총액":
        q = f"""
            SELECT name, market_cap AS value
            FROM stock_prices
            WHERE trade_date = :d {nm_sql}
              AND market_cap IS NOT NULL
            ORDER BY value DESC
            LIMIT 5
        """
        params = {"d": d, **nm_param}

    elif metric == "등락률":
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return {"status": "error", "type": "simple_lookup", "reason": "전일 거래일을 찾을 수 없습니다."}
        q = f"""
            SELECT t.name, {_pct_change_expr('t','p')} AS value
            FROM stock_prices t
            JOIN stock_prices p ON p.name = t.name AND p.trade_date = :prev_d
            WHERE t.trade_date = :d {nm_sql}
            ORDER BY value DESC
            LIMIT 5
        """
        params = {"d": d, "prev_d": prev_d, **nm_param}

    else:
        col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
        if metric not in col_map:
            return {"status": "unsupported", "type": "simple_lookup", "reason": f"지원하지 않는 metric: {metric}"}
        col = col_map[metric]
        q = f"""
            SELECT name, {col} AS value
            FROM stock_prices
            WHERE trade_date = :d {nm_sql}
            ORDER BY value DESC
            LIMIT 5
        """
        params = {"d": d, **nm_param}

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(q), params)]
    return {"status": "ok", "type": "simple_lookup", "metric": metric, "company_name": company,
            "date": d.isoformat(), "rows": rows}

def _task1_stock_rank(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    company = clause["company_name"]
    metric  = clause["metric"]
    nm_sql, nm_param = _like_company(company)

    if metric == "시가총액":
        q = f"""
            WITH T AS (
                SELECT name, market_cap AS val,
                       DENSE_RANK() OVER (ORDER BY market_cap DESC) AS rnk
                FROM stock_prices
                WHERE trade_date = :d AND market_cap IS NOT NULL
            )
            SELECT name, val AS value, rnk
            FROM T
            WHERE 1=1 {nm_sql}
            LIMIT 5
        """
        params = {"d": d, **nm_param}

    elif metric == "거래량":
        q = f"""
            WITH T AS (
                SELECT name, volume AS val,
                       DENSE_RANK() OVER (ORDER BY volume DESC) AS rnk
                FROM stock_prices
                WHERE trade_date = :d
            )
            SELECT name, val AS value, rnk
            FROM T
            WHERE 1=1 {nm_sql}
            LIMIT 5
        """
        params = {"d": d, **nm_param}

    elif metric == "등락률":
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return {"status": "error", "type": "stock_rank", "reason": "전일 거래일을 찾을 수 없습니다."}
        q = f"""
            WITH T AS (
                SELECT t.name, {_pct_change_expr('t','p')} AS val,
                       DENSE_RANK() OVER (ORDER BY {_pct_change_expr('t','p')} DESC) AS rnk
                FROM stock_prices t
                JOIN stock_prices p ON p.name = t.name AND p.trade_date = :prev_d
                WHERE t.trade_date = :d
            )
            SELECT name, val AS value, rnk
            FROM T
            WHERE 1=1 {nm_sql}
            LIMIT 5
        """
        params = {"d": d, "prev_d": prev_d, **nm_param}

    else:
        return {"status": "unsupported", "type": "stock_rank", "reason": f"지원하지 않는 metric: {metric}"}

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(q), params)]
    return {"status": "ok", "type": "stock_rank", "metric": metric, "company_name": company,
            "date": d.isoformat(), "rows": rows}

def _task1_stock_to_market_ratio(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    company = clause["company_name"]
    metric  = clause["metric"]  # "거래량" | "시가총액"

    if metric != "거래량":
        return {"status": "unsupported", "type": "stock_to_market_ratio", "reason": "거래량만 지원합니다."}

    nm_sql, nm_param = _like_company(company)

    q = f"""
        SELECT
          (SELECT SUM(volume) FROM stock_prices WHERE trade_date = :d) AS market_volume,
          (SELECT SUM(volume) FROM stock_prices WHERE trade_date = :d {nm_sql}) AS stock_volume
    """
    with engine.connect() as conn:
        row = conn.execute(text(q), {"d": d, **nm_param}).mappings().first()

    if not row or row["market_volume"] in (None, 0) or row["stock_volume"] is None:
        return {"status": "error", "type": "stock_to_market_ratio", "reason": "비율 계산 실패"}

    ratio_pct = float(row["stock_volume"]) / float(row["market_volume"]) * 100.0
    return {"status": "ok", "type": "stock_to_market_ratio", "metric": metric,
            "company_name": company, "date": d.isoformat(),
            "market_volume": int(row["market_volume"]), "stock_volume": int(row["stock_volume"]),
            "ratio_pct": ratio_pct}

def _task1_market_index_comparison(engine: Engine, d: date, clause: dict) -> Dict[str, Any]:
    """
    시장 지수 비교:
      - indices: ["KOSPI", "KOSDAQ"] 중 일부
      - metric: "지수" | "전일 대비 상승률"
    DB에는 name: INDEX_KOSPI / INDEX_KOSDAQ 로 들어가 있으므로 이를 사용.
    """
    indices: List[str] = clause.get("indices", []) or []
    metric: str = clause["metric"]  # "지수" | "전일 대비 상승률"

    names = _index_names_from_indices(indices)
    if not names:
        return {"status": "error", "type": "market_index_comparison", "reason": "유효한 지수명이 없습니다.(indices)"}

    if metric == "지수":
        # 해당 일자 close 값 그대로
        sql = text("""
            SELECT name, close AS value
            FROM stock_prices
            WHERE trade_date = :d AND name IN :names
        """).bindparams(bindparam("names", expanding=True))

        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(sql, {"d": d, "names": names})]

        return {
            "status": "ok",
            "type": "market_index_comparison",
            "metric": metric,
            "date": d.isoformat(),
            "rows": rows,
        }

    elif metric == "전일 대비 상승률":
        # 각 지수별 전일 거래일을 구해 pct change 계산
        # MySQL 8 CTE 사용
        sql = text(f"""
            WITH prev AS (
              SELECT name, MAX(trade_date) AS prev_d
              FROM stock_prices
              WHERE trade_date < :d AND name IN :names
              GROUP BY name
            )
            SELECT t.name,
                   ((t.close - p.close) / NULLIF(p.close, 0) * 100.0) AS value
            FROM stock_prices t
            JOIN prev v ON v.name = t.name
            JOIN stock_prices p ON p.name = t.name AND p.trade_date = v.prev_d
            WHERE t.trade_date = :d AND t.name IN :names
        """).bindparams(bindparam("names", expanding=True))

        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(sql, {"d": d, "names": names})]

        return {
            "status": "ok",
            "type": "market_index_comparison",
            "metric": metric,
            "date": d.isoformat(),
            "rows": rows,
        }

    else:
        return {"status": "unsupported", "type": "market_index_comparison", "reason": f"지원하지 않는 metric: {metric}"}

# ─────────────────────────────────────────────────────────────
# Dispatcher: run_task1_query
# ─────────────────────────────────────────────────────────────
def run_task1_query(task_obj, engine: Engine = ENGINE) -> Dict[str, Any]:
    d = _first_valid_date(getattr(task_obj, "date", None))
    if not d:
        return {"status": "error", "reason": "유효한 날짜가 없습니다.", "stage": "precheck"}

    clauses = getattr(task_obj, "clauses", None) or []
    if not isinstance(clauses, list) or not clauses:
        return {"status": "error", "reason": "Task1.clauses가 비어 있습니다.", "stage": "precheck"}

    clause = clauses[0]
    ctype = clause.get("type")
    market = getattr(task_obj, "market", None)  # ⬅️ 추가

    if ctype == "metric_rank":
        return _task1_metric_rank(engine, d, clause, market=market)
    elif ctype == "market_comparison":
        # (원하면 market_comparison에도 시장 필터 적용 가능하지만, '시장 평균' 의미가 달라질 수 있어 기본은 전체시장)
        return _task1_market_comparison(engine, d, clause)
    elif ctype == "stock_comparison":
        return _task1_stock_comparison(engine, d, clause)
    elif ctype == "simple_lookup":
        return _task1_simple_lookup(engine, d, clause)
    elif ctype == "market_index_comparison":
        return _task1_market_index_comparison(engine, d, clause)
    elif ctype == "stock_rank":
        return _task1_stock_rank(engine, d, clause)
    elif ctype == "stock_to_market_ratio":
        return _task1_stock_to_market_ratio(engine, d, clause)
    else:
        return {"status": "error", "reason": f"알 수 없는 clause type: {ctype}"}

def _validate_task2_inputs(task_obj, engine: Engine) -> Tuple[Optional[date], Optional[List[dict]], Optional[List[str]], bool, Optional[date], Optional[str]]:
    d = _to_date(getattr(task_obj, "date", None))
    if not d:
        return None, None, None, False, None, "유효한 date가 없습니다."

    clauses = getattr(task_obj, "clauses", None)
    if not isinstance(clauses, list) or len(clauses) == 0:
        return None, None, None, False, None, "clauses가 비어 있습니다."

    market = _normalize_market(getattr(task_obj, "market", None))

    # pct / vol_pct 중 하나라도 포함되면 전일 필요
    need_prev = any(isinstance(c, dict) and c.get("type") in ("pct", "vol_pct") for c in clauses)
    prev_d = None
    if need_prev:
        prev_d = _prev_trade_date(engine, d)
        if not prev_d:
            return None, None, None, False, None, "전일 거래일을 찾을 수 없습니다."

    return d, clauses, market, need_prev, prev_d, None


# 2) SQL 파트 빌더
def _build_select(need_prev: bool) -> str:
    base_cols = [
        "t.name",
        "t.trade_date",
        "t.open",
        "t.high",
        "t.low",
        "t.close",
        "t.volume",
    ]
    if need_prev:
        base_cols += [
            "((t.close - p.close) / NULLIF(p.close,0) * 100.0) AS pct_change",
            "((t.volume - p.volume) / NULLIF(p.volume,0) * 100.0) AS vol_pct_change",
        ]
    return ", ".join(base_cols)

def _build_from_join(need_prev: bool) -> str:
    if not need_prev:
        return "FROM stock_prices t"
    return "FROM stock_prices t JOIN stock_prices p ON p.name = t.name AND p.trade_date = :prev_d"

def _build_base_where(d: date, market: Optional[List[str]]) -> Tuple[List[str], Dict[str, Any]]:
    where = ["t.trade_date = :d"]
    params: Dict[str, Any] = {"d": d}
    mk_sql, mk_params = _market_filter_names_sql("t", market)
    if mk_sql:
        where.append(mk_sql)
        params.update(mk_params)
    return where, params


# 3) Clause 핸들러들: 각 핸들러는 (snippet, params) 반환
def _handle_pct(cl: dict, idx: int) -> Tuple[str, Dict[str, Any]]:
    op = cl.get("op"); val = cl.get("value_pct")
    if op not in (">=", "<=", ">", "<") or not isinstance(val, (int, float)):
        raise ValueError(f"pct clause 형식 오류: {cl}")
    key = f"pct_{idx}"
    return f"( (t.close - p.close) / NULLIF(p.close,0) * 100.0 ) {op} :{key}", {key: float(val)}

def _handle_vol_pct(cl: dict, idx: int) -> Tuple[str, Dict[str, Any]]:
    op = cl.get("op"); val = cl.get("value_pct")
    if op not in (">=", "<=", ">", "<") or not isinstance(val, (int, float)):
        raise ValueError(f"vol_pct clause 형식 오류: {cl}")
    key = f"volpct_{idx}"
    return f"( (t.volume - p.volume) / NULLIF(p.volume,0) * 100.0 ) {op} :{key}", {key: float(val)}

def _handle_vol_abs(cl: dict, idx: int) -> Tuple[str, Dict[str, Any]]:
    op = cl.get("op"); shares = cl.get("shares")
    if op not in (">=", "<=", ">", "<") or not isinstance(shares, int):
        raise ValueError(f"vol_abs clause 형식 오류: {cl}")
    key = f"volabs_{idx}"
    return f"t.volume {op} :{key}", {key: int(shares)}

def _handle_price_range(cl: dict, idx: int) -> Tuple[str, Dict[str, Any]]:
    low = cl.get("low"); high = cl.get("high")
    parts, params = [], {}
    if low is not None:
        key = f"pr_low_{idx}"
        parts.append(f"t.close >= :{key}")
        params[key] = float(low)
    if high is not None:
        key = f"pr_high_{idx}"
        parts.append(f"t.close <= :{key}")
        params[key] = float(high)
    return " AND ".join(parts) if parts else "1=1", params


# 4) 레지스트리
CLAUSE_HANDLERS = {
    "pct": _handle_pct,
    "vol_pct": _handle_vol_pct,
    "vol_abs": _handle_vol_abs,
    "price_range": _handle_price_range,
}


# 5) 메인 실행
def run_task2_query(task_obj, engine: Engine = ENGINE) -> Dict[str, Any]:
    d, clauses, market, need_prev, prev_d, err = _validate_task2_inputs(task_obj, engine)
    if err:
        return {"status": "error", "reason": err}
    assert d is not None and clauses is not None  # mypy용

    select_sql = _build_select(need_prev)
    from_join  = _build_from_join(need_prev)
    where_parts, params = _build_base_where(d, market)

    if need_prev:
        params["prev_d"] = prev_d

    # Clause 적용
    for idx, cl in enumerate(clauses):
        if not isinstance(cl, dict):
            return {"status": "error", "reason": f"clause가 dict가 아님: {cl}"}
        ctype = cl.get("type")
        handler = CLAUSE_HANDLERS.get(ctype)
        if handler is None:
            return {"status": "error", "reason": f"알 수 없는 clause type: {ctype}"}
        try:
            snippet, p = handler(cl, idx)
        except ValueError as e:
            return {"status": "error", "reason": str(e)}
        if snippet and snippet != "1=1":
            where_parts.append(snippet)
        params.update(p)

    sql = f"""
        SELECT {select_sql}
        {from_join}
        WHERE {' AND '.join(where_parts)}
        ORDER BY t.close DESC
        LIMIT 500
    """

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(sql), params)]

    return {
        "status": "ok",
        "date": d.isoformat(),
        "need_prev": need_prev,
        "rows": rows,
        "applied_clauses": clauses,
        "market": market,
        "sql_debug": {"sql": sql, "params": params},  # 필요 없으면 제거
    }

def _task3_volume_signal(engine: Engine, company_name: Optional[str], market, ds: date, de: date,
                         days: Optional[int], change_percent: Optional[float], mode: str) -> Dict[str, Any]:
    n = int(days or 20)
    nm_sql, nm_param = _name_like_sql(company_name)
    mk_sql, mk_param = _market_filter_names_sql_plain(market)

    # 윈도우 평균용 버퍼 범위: 과거 데이터를 충분히 포함
    # (여유 있게 3N일 또는 최소 60일 권장)
    buf_days = max(n * 3, 60)
    ds_buf = ds - timedelta(days=buf_days)

    base = f"""
        WITH X AS (
          SELECT
            name, trade_date, volume,
            AVG(volume) OVER (
              PARTITION BY name
              ORDER BY trade_date
              ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING
            ) AS avg_prev_vol
          FROM stock_prices
          WHERE trade_date BETWEEN :ds_buf AND :de
          {mk_sql}
        )
        SELECT
          name, trade_date, volume,
          ((volume / NULLIF(avg_prev_vol,0)) - 1.0) * 100.0 AS vol_spike_pct
        FROM X
        WHERE avg_prev_vol IS NOT NULL
          AND trade_date BETWEEN :ds AND :de
          {nm_sql}
    """

    cond = ""
    if change_percent is not None:
        cond = " AND ((volume / NULLIF(avg_prev_vol,0)) - 1.0) * 100.0 >= :thr "

    # 정렬을 스파이크 내림차순으로 (리스트/보스 모드에만 의미)
    order_limit = " ORDER BY vol_spike_pct DESC, name ASC LIMIT 5000 " if mode in ("list", "both") else ""

    sql = text(base + cond + order_limit)
    params = {
        "ds_buf": ds_buf,
        "ds": ds,
        "de": de,
        **mk_param, **nm_param
    }
    if change_percent is not None:
        params["thr"] = float(change_percent)

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]

    payload = _mode_payload(mode, rows)
    return {
        "status": "ok",
        "type": "volume_signal",
        "period_start": ds.isoformat(),
        "period_end": de.isoformat(),
        "days": n,
        "change_percent": change_percent,
        **payload
    }

# ─────────────────────────────────────────────────────────────
# 시그널: 이동평균 괴리 (moving_average_diff)
# period N, direction above|below
# pct_diff = (close - SMA_N) / SMA_N * 100
# above: pct_diff >= diff_percentage
# below: pct_diff <= -diff_percentage
# SMA는 포함 윈도우(오늘 포함)로 계산
# ─────────────────────────────────────────────────────────────
def _task3_moving_average_diff(
    engine: Engine,
    company_name: Optional[str],
    market,
    ds: date,
    de: date,
    period: int,
    diff_percentage: float,
    direction: Optional[str] = None,   # ✅ Optional로
    mode: str = "list"
) -> Dict[str, Any]:
    n = int(period or 20)
    direction = (direction or "above")  # ✅ 기본값 여기서 보정

    nm_sql, nm_param = _name_like_sql(company_name)
    mk_sql, mk_param = _market_filter_names_sql_plain(market)

    buf_start = ds - timedelta(days=max(40, n * 3))

    base = f"""
        WITH X AS (
          SELECT
            name, trade_date, close,
            AVG(close) OVER (
              PARTITION BY name
              ORDER BY trade_date
              ROWS BETWEEN {n-1} PRECEDING AND CURRENT ROW
            ) AS sma
          FROM stock_prices
          WHERE trade_date BETWEEN :buf_start AND :de
            {mk_sql}
        )
        SELECT
          name, trade_date, close, sma,
          ((close - sma) / NULLIF(sma,0)) * 100.0 AS pct_diff
        FROM X
        WHERE sma IS NOT NULL
          AND trade_date BETWEEN :ds AND :de
          {nm_sql}
    """

    if direction == "above":
        cond = " AND ((close - sma) / NULLIF(sma,0)) * 100.0 >= :thr "
        thr = float(diff_percentage)
        order_clause = " ORDER BY pct_diff DESC, trade_date ASC, name ASC "
    elif direction == "below":
        cond = " AND ((close - sma) / NULLIF(sma,0)) * 100.0 <= :thr "
        thr = -abs(float(diff_percentage))
        order_clause = " ORDER BY pct_diff ASC, trade_date ASC, name ASC "
    else:
        cond = " AND ABS(((close - sma) / NULLIF(sma,0)) * 100.0) >= :thr "
        thr = abs(float(diff_percentage))
        # 양방향이면 절댓값 기준 내림차순
        order_clause = " ORDER BY ABS(pct_diff) DESC, trade_date ASC, name ASC "

    # ✅ 이 함수에서는 정렬을 우리가 넣었으므로 LIMIT만 별도 처리
    limit_clause = " LIMIT 5000 " if mode in ("list", "both") else ""

    sql = text(base + cond + order_clause + limit_clause)
    params = {"buf_start": buf_start, "ds": ds, "de": de, "thr": thr, **mk_param, **nm_param}

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]

    payload = _mode_payload(mode, rows)
    return {
        "status": "ok",
        "type": "moving_average_diff",
        "period_start": ds.isoformat(),
        "period_end": de.isoformat(),
        "period": n,
        "diff_percentage": diff_percentage,
        "direction": direction,
        **payload
    }

# ─────────────────────────────────────────────────────────────
# 시그널: 크로스 이벤트 (cross_events)
# 기본 파라미터(관례): short=50, long=200
# golden: 전일 short<=long 이고 오늘 short>long
# death : 전일 short>=long 이고 오늘 short<long
# ─────────────────────────────────────────────────────────────
def _task3_cross_events(
    engine: Engine,
    company_name: Optional[str],
    market,
    ds: date,
    de: date,
    cross_types: List[str],
    mode: str,
    short_win: int = 5,
    long_win: int = 20,
) -> Dict[str, Any]:
    nm_sql, nm_param = _name_like_sql(company_name)
    mk_sql, mk_param = _market_filter_names_sql_plain(market)

    # 장기선/라그 계산용 버퍼
    buf_start = ds - timedelta(days=max(40, long_win * 3))

    # cross_types 정규화
    cset = set((cross_types or []))
    want_golden = ("golden" in cset) or (not cset)  # 비었으면 둘 다 허용
    want_death  = ("death"  in cset) or (not cset)

    # 조건 블록을 cross_types에 맞춰 구성
    conds = []
    # 공통: lag 존재
    base_ok = "(lag_ma_s IS NOT NULL AND lag_ma_l IS NOT NULL)"
    if want_golden:
        conds.append(f"({base_ok} AND lag_ma_s <= lag_ma_l AND ma_s >  ma_l)")
    if want_death:
        conds.append(f"({base_ok} AND lag_ma_s >= lag_ma_l AND ma_s <  ma_l)")
    cross_cond = " OR ".join(conds) if conds else "0"  # 안전장치

    base = f"""
        WITH MA AS (
          SELECT
            name, trade_date,
            AVG(close) OVER (
              PARTITION BY name ORDER BY trade_date
              ROWS BETWEEN {short_win-1} PRECEDING AND CURRENT ROW
            ) AS ma_s,
            AVG(close) OVER (
              PARTITION BY name ORDER BY trade_date
              ROWS BETWEEN {long_win-1} PRECEDING AND CURRENT ROW
            ) AS ma_l
          FROM stock_prices
          WHERE trade_date BETWEEN :buf_start AND :de
            {mk_sql}
        ),
        X AS (
          SELECT
            name, trade_date, ma_s, ma_l,
            LAG(ma_s) OVER (PARTITION BY name ORDER BY trade_date) AS lag_ma_s,
            LAG(ma_l) OVER (PARTITION BY name ORDER BY trade_date) AS lag_ma_l
          FROM MA
        )
        SELECT
          name,
          trade_date,
          CASE
            WHEN {base_ok} AND lag_ma_s <= lag_ma_l AND ma_s >  ma_l THEN 'golden'
            WHEN {base_ok} AND lag_ma_s >= lag_ma_l AND ma_s <  ma_l THEN 'death'
            ELSE NULL
          END AS cross_type
        FROM X
        WHERE ma_s IS NOT NULL
          AND ma_l IS NOT NULL
          AND trade_date BETWEEN :ds AND :de
          {nm_sql}
          AND ( {cross_cond} )
    """

    sql = text(base + _order_limit_sql(mode))
    params = {
        "buf_start": buf_start,
        "ds": ds,
        "de": de,
        **mk_param,
        **nm_param,
    }

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]

    payload = _mode_payload(mode, rows)
    return {
        "status": "ok",
        "type": "cross_events",
        "period_start": ds.isoformat(),
        "period_end": de.isoformat(),
        "cross_types": cross_types,
        "short_win": short_win,
        "long_win": long_win,
        **payload
    }

# ─────────────────────────────────────────────────────────────
# 시그널: RSI (간단 SMA 버전 14)
# RSI = 100 - 100/(1+RS), RS = avg(gain,14)/avg(loss,14)
# gain = GREATEST(close - LAG(close), 0), loss = GREATEST(LAG(close)-close, 0)
# condition: overbought(>=threshold) / oversold(<=100-threshold)
# ─────────────────────────────────────────────────────────────
def _task3_rsi(
    engine: Engine,
    company_name: Optional[str],
    market,
    ds: date,
    de: date,
    threshold: float,
    condition: str,
    mode: str,
    period: int = 14,
) -> Dict[str, Any]:
    nm_sql, nm_param = _name_like_sql(company_name)
    mk_sql, mk_param = _market_filter_names_sql_plain(market)

    n = int(period or 14)
    ds_buf = ds - timedelta(days=max(60, n * 3))

    base_sql = f"""
        WITH RECURSIVE
        BASE AS (
          SELECT
            name, trade_date, close,
            LAG(close) OVER (PARTITION BY name ORDER BY trade_date) AS prev_close
          FROM stock_prices
          WHERE trade_date BETWEEN :ds_buf AND :de
          {mk_sql}
        ),
        D AS (
          SELECT
            name,
            trade_date,
            close,
            GREATEST(close - prev_close, 0) AS gain,
            GREATEST(prev_close - close, 0) AS loss,
            ROW_NUMBER() OVER (PARTITION BY name ORDER BY trade_date) AS rn
          FROM BASE
          WHERE prev_close IS NOT NULL
        ),
        SEED AS (
          SELECT
            name, trade_date, close, gain, loss, rn,
            CASE WHEN rn = :n
              THEN (SUM(gain) OVER (
                      PARTITION BY name ORDER BY trade_date
                      ROWS BETWEEN :n_minus1 PRECEDING AND CURRENT ROW
                    ) / :n)
            END AS avg_gain,
            CASE WHEN rn = :n
              THEN (SUM(loss) OVER (
                      PARTITION BY name ORDER BY trade_date
                      ROWS BETWEEN :n_minus1 PRECEDING AND CURRENT ROW
                    ) / :n)
            END AS avg_loss
          FROM D
        ),
        REC AS (
          -- 씨앗: rn = n
          SELECT
            name, trade_date, close, rn, gain, loss, avg_gain, avg_loss
          FROM SEED
          WHERE rn = :n

          UNION ALL

          -- Wilder 누적: 다음 행부터 끝까지
          SELECT
            d.name,
            d.trade_date,
            d.close,
            d.rn,
            d.gain,
            d.loss,
            (r.avg_gain * (:n - 1) + d.gain) / :n AS avg_gain,
            (r.avg_loss * (:n - 1) + d.loss) / :n AS avg_loss
          FROM REC r
          JOIN D d
            ON d.name = r.name
           AND d.rn   = r.rn + 1
        ),
        RSI AS (
          SELECT
            name,
            trade_date,
            close,
            CASE
              WHEN avg_gain IS NULL OR avg_loss IS NULL THEN NULL
              WHEN avg_gain = 0 AND avg_loss = 0 THEN 50.0
              WHEN avg_loss = 0 THEN 100.0
              WHEN avg_gain = 0 THEN 0.0
              ELSE 100.0 - (100.0 / (1.0 + (avg_gain / NULLIF(avg_loss,0))))
            END AS rsi
          FROM REC
        )
        SELECT name, trade_date, close, rsi
        FROM RSI
        WHERE trade_date BETWEEN :ds AND :de
        {nm_sql}
    """

    # 조건/정렬
    if condition == "overbought":
        base_sql += " AND rsi IS NOT NULL AND rsi >= :thr "
        order_sql = " ORDER BY rsi DESC, trade_date DESC, name ASC "
    else:
        base_sql += " AND rsi IS NOT NULL AND rsi <= :thr "
        order_sql = " ORDER BY rsi ASC, trade_date DESC, name ASC "

    if mode in ("list", "both"):
        order_sql += " LIMIT 5000 "

    sql = text(base_sql + order_sql)
    params = {
        "ds_buf": ds_buf,
        "ds": ds,
        "de": de,
        "thr": float(threshold),
        "n": n,
        "n_minus1": n - 1,
        **mk_param,
        **nm_param,
    }

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]

    payload = _mode_payload(mode, rows)
    return {
        "status": "ok",
        "type": "rsi",
        "period_start": ds.isoformat(),
        "period_end": de.isoformat(),
        "threshold": threshold,
        "condition": condition,
        "period": n,
        **payload
    }

# ─────────────────────────────────────────────────────────────
# 시그널: 볼린저 밴드 터치 (기본 20일, 표준편차 k=2)
# band: upper|lower, touch: True → upper: close>=upper, lower: close<=lower
# ─────────────────────────────────────────────────────────────
def _task3_bollinger(
    engine: Engine,
    company_name: Optional[str],
    market,
    ds: date,
    de: date,
    band: str,
    touch: bool,
    mode: str,
    period: int = 20,
    k: float = 2.0,
    prefer_adj: bool = True,
) -> Dict[str, Any]:
    nm_sql, nm_param = _name_like_sql(company_name)
    mk_sql, mk_param = _market_filter_names_sql_plain(market)

    # 윈도우용 버퍼(과거 봉 확보)
    buf_start = ds - timedelta(days=max(40, period * 3))

    # adj_close 선호, 없으면 close
    price_col = "adj_close" if prefer_adj else "close"
    with engine.connect() as conn:
        cols = {r[0] for r in conn.execute(text("SHOW COLUMNS FROM stock_prices")).fetchall()}
    if price_col not in cols:
        price_col = "close"

    base = f"""
        WITH B AS (
          SELECT
            name,
            trade_date,
            {price_col} AS close,
            AVG({price_col}) OVER (
              PARTITION BY name ORDER BY trade_date
              ROWS BETWEEN {period-1} PRECEDING AND CURRENT ROW
            ) AS ma,
            STDDEV_SAMP({price_col}) OVER (
              PARTITION BY name ORDER BY trade_date
              ROWS BETWEEN {period-1} PRECEDING AND CURRENT ROW
            ) AS sd
          FROM stock_prices
          WHERE trade_date BETWEEN :buf_start AND :de
          {mk_sql}
        )
        SELECT
          name,
          trade_date,
          close,
          ma,
          sd,
          (ma + :k * sd) AS upper_band,
          (ma - :k * sd) AS lower_band,
          -- 밴드와의 괴리(정렬용)
          CASE
            WHEN :band = 'upper' THEN close - (ma + :k * sd)
            ELSE (ma - :k * sd) - close
          END AS band_distance
        FROM B
        WHERE ma IS NOT NULL
          AND sd IS NOT NULL
          AND trade_date BETWEEN :ds AND :de
          {nm_sql}
    """

    # 터치 조건
    cond = ""
    if touch:
        if band == "upper":
            cond = " AND close >= (ma + :k * sd) "
        else:
            cond = " AND close <= (ma - :k * sd) "

    # 정렬/리밋(여기서는 커스텀 정렬 사용: 괴리 큰 순)
    order_sql = ""
    if mode in ("list", "both"):
        # 상단 터치면 band_distance 큰 순(양수 클수록 더 위), 하단 터치면 band_distance 큰 순(양수 클수록 더 아래)
        # touch가 False여도 괴리 큰 순으로 보여주면 직관적
        order_sql = " ORDER BY band_distance DESC, trade_date DESC, name ASC LIMIT 5000 "

    sql = text(base + cond + order_sql)
    params = {
        "buf_start": buf_start,
        "ds": ds,
        "de": de,
        "k": float(k),
        "band": band,               # CASE에서 사용
        **mk_param,
        **nm_param,
    }

    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql, params)]

    payload = _mode_payload(mode, rows)
    return {
        "status": "ok",
        "type": "bollinger_band",
        "period_start": ds.isoformat(),
        "period_end": de.isoformat(),
        "band": band,
        "touch": touch,
        "period": period,
        "k": k,
        "price_col": price_col,
        **payload
    }

# ─────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────
def run_task3_query(task_obj, engine: Engine = ENGINE) -> Dict[str, Any]:
    # 기간 보정
    ds, de = _date_range_if_single(getattr(task_obj, "period_start", None),
                                   getattr(task_obj, "period_end", None))
    if not ds or not de:
        return {"status": "error", "reason": "기간(period_start/period_end)이 유효하지 않습니다.", "stage": "precheck"}

    company_name: Optional[str] = getattr(task_obj, "company_name", None)
    market = getattr(task_obj, "market", None)
    mode: str = getattr(task_obj, "mode", "list") or "list"
    signals = getattr(task_obj, "signal_type", None)

    # signal_type: 단일 or 배열 모두 허용 → 리스트로 정규화
    sig_list: List[dict]
    if signals is None:
        sig_list = []
    elif isinstance(signals, dict):
        sig_list = [signals]
    else:
        sig_list = list(signals)

    if not sig_list:
        return {"status": "error", "reason": "signal_type이 없습니다.", "stage": "precheck"}

    # 현재는 첫 시그널만 처리(원하면 loop로 교집합/합집합 로직 확장)
    sig = sig_list[0]
    stype = sig.get("type")

    if stype == "volume_signal":
        return _task3_volume_signal(
            engine, company_name, market, ds, de,
            days=sig.get("days"), change_percent=sig.get("change_percent"),
            mode=mode
        )

    elif stype == "cross_events":
        return _task3_cross_events(
            engine, company_name, market, ds, de,
            cross_types=sig.get("cross_types", []), mode=mode
        )

    elif stype == "rsi":
        return _task3_rsi(
            engine, company_name, market, ds, de,
            threshold=float(sig["threshold"]), condition=sig["condition"], mode=mode
        )

    elif stype == "bollinger_band":
        return _task3_bollinger(
            engine, company_name, market, ds, de,
            band=sig["band"], touch=bool(sig["touch"]), mode=mode
        )
    elif stype == "moving_average_diff":
        return _task3_moving_average_diff(
            engine, company_name, market, ds, de,
            period=int(sig.get("period") or 20),
            diff_percentage=float(sig.get("diff_percentage") or 0),
            direction=(sig.get("direction") or "above"),
            mode=mode
        )

    else:
        return {"status": "error", "reason": f"알 수 없는 signal_type: {stype}"}
    
def _find_last_ai_question(messages: list[BaseMessage], question_text: Optional[str]) -> int:
    """
    state['human_question']와 동일한 내용을 가진 마지막 AIMessage의 인덱스를 찾는다.
    못 찾으면 -1 반환.
    """
    if not question_text:
        return -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "content", None) == question_text:
            return i
    return -1

def _find_user_query_before(messages: list[BaseMessage], idx: int) -> Optional[str]:
    """
    idx(보통 질문 AIMessage)의 이전 구간에서 가장 가까운 HumanMessage의 content를 반환.
    없으면 None.
    """
    j = idx - 1 if idx >= 0 else len(messages) - 1
    for k in range(j, -1, -1):
        if isinstance(messages[k], HumanMessage):
            return messages[k].content
    return None

def rewrite_query_with_human_feedback(state: State) -> Dict[str, Any]:
    """
    사람의 추가 입력(clarification)을 기존 사용자 질의와 병합해
    파이프라인을 재시작할 수 있도록 messages를 재작성한다.

    반환값: state.update에 바로 넣을 수 있는 dict
      - messages: 재작성된 메시지 시퀀스
      - ask_human: False 로 reset
      - human_question: None 로 reset
      - question: [] 로 reset
    """
    messages: list[BaseMessage] = list(state["messages"])
    last_msg: BaseMessage = messages[-1]

    if not isinstance(last_msg, HumanMessage):
        # 사람이 아직 답하지 않은 경우: 아무것도 하지 않음(보호적 처리)
        return {"ask_human": True}

    clarification = last_msg.content
    # 질문이던 AIMessage 위치 찾기
    ai_q_idx = _find_last_ai_question(messages, state.get("human_question"))
    # 직전 사용자 원질의(질문 전에 있던) 추출
    original_user_query = _find_user_query_before(messages, ai_q_idx)

    # 병합된 프롬프트(다음 분류/파싱이 보기에 깔끔한 형태)
    fused = []
    if original_user_query:
        fused.append(f"[원래 질문]\n{original_user_query}")
    else:
        # 원본을 못 찾으면 대체로 직전 HumanMessage가 원본일 가능성 → 생략 가능
        pass
    fused.append(f"[추가 조건]\n{clarification}")
    fused_content = "\n\n".join(fused)

    # messages 재구성:
    # - 기존 히스토리는 유지하되, 마지막 clarification만 남기고
    # - 새로 합쳐진 HumanMessage 하나를 추가(중복 방지)
    #   (원하면 히스토리를 간결히 하도록 잘라낼 수도 있음)
    new_messages = messages[:-1] + [HumanMessage(content=fused_content)]

    return {
        "messages": new_messages,
        "ask_human": False,
        "human_question": None,
        "question": [],   # UI용 질문 표시 필드 초기화
    }

def _load_company_aliases() -> List[Dict[str, str]]:
    """
    입력 CSV: 종목코드,종목명,시장구분
    출력 레코드: {alias, official, ticker, market} (alias를 여러 개로 explode)
    """
    global _COMPANY_ALIASES
    if _COMPANY_ALIASES:
        return _COMPANY_ALIASES
    if not COMPANY_CSV.exists():
        return []

    rows = []
    with COMPANY_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code   = (r.get("종목코드") or "").strip()
            name   = (r.get("종목명") or "").strip()
            market = (r.get("시장구분") or "").strip()  # KOSPI/KOSDAQ

            if not code or not name:
                continue

            # 대표명
            aliases = {name, code}

            # 공백/특수문자 제거 변형
            compact = re.sub(r"\s+", "", name)
            aliases.add(compact)

            # 우/1우/2우B 등의 괄호/공백 변형(간단 처리)
            simple = re.sub(r"[\(\) ]+", "", name)
            aliases.add(simple)

            # 자주 쓰는 별칭 seed 추가(원하면 확장)
            builtin = {
                "삼전": "삼성전자", "하닉": "SK하이닉스", "엘지화학": "LG화학", "엘지전자": "LG전자"
            }
            if name in builtin.values():
                for k, v in builtin.items():
                    if v == name:
                        aliases.add(k)

            for al in aliases:
                rows.append({"alias": al, "official": name, "ticker": code, "market": market})

    _COMPANY_ALIASES = rows
    return rows

def _load_term_rules() -> List[Dict[str, str]]:
    """
    입력: type,canonical_name,alias,description
    출력: {slang, canonical, rewrite}
    - alias는 ';'로 분리하여 여러 줄로 확장
    - rewrite는 description이 있으면 그걸, 없으면 canonical_name
    """
    global _TERM_RULES
    if _TERM_RULES:
        return _TERM_RULES
    if not TERMS_CSV.exists():
        return []

    out = []
    with TERMS_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            canonical = (r.get("canonical_name") or "").strip()
            desc      = (r.get("description") or "").strip()
            alias_raw = (r.get("alias") or "").strip()
            if not alias_raw:
                continue
            rewrite = desc or canonical or ""
            for slang in [a.strip() for a in alias_raw.split(";") if a.strip()]:
                out.append({"slang": slang, "canonical": canonical, "rewrite": rewrite})
    _TERM_RULES = out
    return out
# ─────────────────────────────────────────────────────────────
# 회사명 퍼지 매칭

def fuzzy_match_company_name(name: str, limit: int = 5) -> List[Tuple[str, float, Dict[str, str]]]:
    """
    입력 문자열 name에 대해 company_name.csv의 alias를 기준으로 퍼지 후보를 반환.
    return: [(official, score, meta_dict), ...]  (score: 0~100)
    """
    aliases = _load_company_aliases()
    if not aliases:
        return []

    alias_list = [r["alias"] for r in aliases]

    if _HAS_RAPIDFUZZ:
        matches = rf_process.extract(
            name, alias_list, scorer=rf_fuzz.WRatio, limit=limit
        )
        # matches: [(matched_alias, score, index), ...]
        out: List[Tuple[str, float, Dict[str, str]]] = []
        for alias_text, score, idx in matches:
            meta = aliases[idx]
            out.append((meta["official"], float(score), meta))
        return out
    else:
        # difflib 폴백
        close = difflib.get_close_matches(name, alias_list, n=limit, cutoff=0.0)
        out: List[Tuple[str, float, Dict[str, str]]] = []
        for alias_text in close:
            idx = alias_list.index(alias_text)
            meta = aliases[idx]
            # 대략적 점수(간이) : 길이 차감
            score = 100.0 * (1.0 - (abs(len(alias_text) - len(name)) / max(1, len(alias_text))))
            out.append((meta["official"], float(score), meta))
        return out

def normalize_company_in_text(text: str, score_threshold: float = 70.0) -> Tuple[str, Optional[str], Optional[str]]:
    """
    문장 내 회사명(애칭/오타)을 퍼지 매칭해 공식명으로 치환.
    return: (replaced_text, official_name_or_None, market_or_None)
    """
    aliases = _load_company_aliases()
    if not aliases:
        return text, None, None

    tokens = re.findall(r"[가-힣A-Za-z0-9_]+", text)
    replaced = text
    best_official: Optional[str] = None
    best_market: Optional[str] = None

    # 긴 토큰부터 시도 (부분 치환 방지)
    for tok in sorted(set(tokens), key=len, reverse=True):
        cands = fuzzy_match_company_name(tok, limit=1)
        if not cands:
            continue
        official, score, meta = cands[0]
        if score >= score_threshold:
            # 한글/영문/숫자 경계 안전 치환
            pattern = rf"(?<![가-힣A-Za-z0-9_]){re.escape(tok)}(?![가-힣A-Za-z0-9_])"
            replaced = re.sub(pattern, official, replaced)
            if not best_official:
                best_official = official
                best_market = meta.get("market") or None

    return replaced, best_official, best_market

# ─────────────────────────────────────────────────────────────
# 용어 정규화 (적삼병 등)

def normalize_terms_in_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    stock_terms.csv 기반으로 문장 내 용어를 표준 라벨/치환문으로 정규화.
    return: (치환된문장, [{slang, canonical, rewrite_used}, ...])
    """
    rules = _load_term_rules()
    if not rules:
        return text, []

    applied: List[Dict[str, str]] = []
    replaced = text
    for r in rules:
        slang = r["slang"]
        canonical = r["canonical"]
        rewrite = r["rewrite"] or canonical
        # 부분 일치 허용(단, 과치환 방지 위해 긴 것부터)
        if slang and slang in replaced:
            replaced = replaced.replace(slang, rewrite)
            applied.append({"slang": slang, "canonical": canonical, "rewrite_used": rewrite})
    return replaced, applied

# ─────────────────────────────────────────────────────────────
# 모호성 해소 → 메시지/필드 업데이트

def apply_ambiguity_resolution(raw_text: str) -> Dict[str, Any]:
    """
    1) 회사명 정규화
    2) 용어 정규화
    3) 정규화한 문장/필드를 반환 (state.update에 그대로 사용 가능)
    """
    after_company, official, market = normalize_company_in_text(raw_text)
    after_terms, term_hits = normalize_terms_in_text(after_company)

    update: Dict[str, Any] = {
        "normalized_query": after_terms,
        "term_hits": term_hits,  # 디버그/트레이스용
    }
    clarified_fields: Dict[str, Any] = {}
    if official:
        clarified_fields["company_name"] = official
    if market:
        # 이미 사용자가 market 지정했을 수 있으니 '힌트'로
        clarified_fields.setdefault("market_hint", market)

    if clarified_fields:
        update["clarified_fields"] = clarified_fields

    return update

def _pick_best_result(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 우선순위 키 후보
    for k in ["task3_result", "task2_result", "task1_result", "result", "last_result", "final_result"]:
        v = state.get(k)
        if isinstance(v, dict) and ("status" in v or "rows" in v):
            return v
    # 그래도 없으면 state의 dict 값 뒤에서 앞으로 스캔
    for _, v in list(state.items())[::-1]:
        if isinstance(v, dict) and ("status" in v or "rows" in v):
            return v
    return None

def _fmt_num(x: Any) -> str:
    try:
        if isinstance(x, int):
            return f"{x:,}"
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

# ===== Task1 렌더러 =====
def _r_task1_metric_rank(r: Dict[str, Any]) -> str:
    metric, rank_type, dt = r.get("metric"), r.get("rank_type"), r.get("date")
    rows = r.get("rows", [])[:10]
    head = f"{dt} 기준 {metric} {'상위' if rank_type=='top' else '하위'} 종목:"
    body = [f"  {i}. {row.get('name')} — {_fmt_num(row.get('value'))}" for i, row in enumerate(rows, 1)]
    return "\n".join([head] + body)

def _r_task1_market_comparison(r: Dict[str, Any]) -> str:
    metric, cmpk, dt = r.get("metric"), r.get("comparison"), r.get("date")
    cmp_txt = "시장 평균 이상" if cmpk == "above_avg" else "시장 평균 이하"
    avgv = r.get("market_avg")
    rows = r.get("rows", [])[:10]
    head = f"{dt} {metric} {cmp_txt} 종목 (시장 평균: {_fmt_num(avgv)}):"
    body = [f"  {i}. {row.get('name')} — {_fmt_num(row.get('value'))}" for i, row in enumerate(rows, 1)]
    return "\n".join([head] + body)

def _r_task1_stock_comparison(r: Dict[str, Any]) -> str:
    metric, dt = r.get("metric"), r.get("date")
    a, b = r.get("stock_a"), r.get("stock_b")
    rows = r.get("rows", [])
    a_val = next((x.get("value") for x in rows if x.get("which")=="A"), None)
    b_val = next((x.get("value") for x in rows if x.get("which")=="B"), None)
    winner = r.get("winner")
    wtxt = {"A": a, "B": b, "tie": "동일"}.get(winner, None)
    lines = [f"{dt} {metric} 비교: {a} vs {b}",
             f"{a}: {_fmt_num(a_val)} / {b}: {_fmt_num(b_val)}"]
    if wtxt: lines.append(f"→ 우위: {wtxt}")
    return "\n".join(lines)

def _r_task1_simple_lookup(r: Dict[str, Any]) -> str:
    metric, comp, dt = r.get("metric"), r.get("company_name"), r.get("date")
    rows = r.get("rows", [])
    if not rows:
        return f"{dt} {comp}의 {metric} 데이터를 찾지 못했습니다."
    head = f"{dt} {comp} {metric}:"
    body = [f"  - {row.get('name')}: {_fmt_num(row.get('value'))}" for row in rows]
    return "\n".join([head] + body)

def _r_task1_stock_rank(r: Dict[str, Any]) -> str:
    metric, comp, dt = r.get("metric"), r.get("company_name"), r.get("date")
    rows = r.get("rows", [])
    if not rows:
        return f"{dt} {comp}의 {metric} 순위를 찾지 못했습니다."
    head = f"{dt} {comp} {metric} 순위:"
    body = [f"  - {row.get('name')}: {_fmt_num(row.get('value'))} (순위 {row.get('rnk')})" for row in rows]
    return "\n".join([head] + body)

def _r_task1_stock_to_market_ratio(r: Dict[str, Any]) -> str:
    comp, dt = r.get("company_name"), r.get("date")
    mv, sv, pct = r.get("market_volume"), r.get("stock_volume"), r.get("ratio_pct")
    return (f"{dt} 기준 {comp} 거래량 시장점유\n"
            f"  - 종목 거래량: {_fmt_num(sv)} / 시장 전체: {_fmt_num(mv)}\n"
            f"  - 비중: {_fmt_num(pct)}%")

def _r_task1_market_index_comparison(r: Dict[str, Any]) -> str:
    metric, dt = r.get("metric"), r.get("date")
    rows = r.get("rows", [])
    head = f"{dt} 지수 비교({metric}):"
    body = [f"  - {row.get('name')}: {_fmt_num(row.get('value'))}" for row in rows]
    return "\n".join([head] + body)

# ===== Task2 렌더러 =====
def _describe_task2_clauses(clauses: List[Dict[str, Any]]) -> str:
    parts = []
    for c in clauses or []:
        t = c.get("type")
        if t == "pct":
            parts.append(f"등락률 {c.get('op')} {c.get('value_pct')}%")
        elif t == "vol_pct":
            parts.append(f"거래량 변화율 {c.get('op')} {c.get('value_pct')}%")
        elif t == "vol_abs":
            parts.append(f"거래량 {c.get('op')} {c.get('shares'):,}주")
        elif t == "price_range":
            lo, hi = c.get("low"), c.get("high")
            if lo is not None and hi is not None:
                parts.append(f"종가 {lo}~{hi}")
            elif lo is not None:
                parts.append(f"종가 ≥ {lo}")
            elif hi is not None:
                parts.append(f"종가 ≤ {hi}")
    return ", ".join(parts)

def _r_task2(r: Dict[str, Any]) -> str:
    dt, need_prev = r.get("date"), r.get("need_prev")
    clauses, rows = r.get("applied_clauses", []), r.get("rows", [])
    desc = _describe_task2_clauses(clauses)
    head = f"{dt} 조건검색 결과 ({desc})"
    if need_prev: head += " — 전일 데이터 활용"
    n = len(rows)
    lines = [f"{head}: 총 {n}개"]
    for row in rows[:15]:
        base = f"  - {row.get('name')} | 종가 {_fmt_num(row.get('close'))}, 거래량 {_fmt_num(row.get('volume'))}"
        if row.get("pct_change") is not None:
            base += f", 등락률 {_fmt_num(row.get('pct_change'))}%"
        if row.get("vol_pct_change") is not None:
            base += f", 거래량 변화율 {_fmt_num(row.get('vol_pct_change'))}%"
        lines.append(base)
    if n > 15:
        lines.append(f"  … 외 {n-15}개")
    return "\n".join(lines)

# ===== Task3 렌더러 =====
def _r_task3_volume_signal(r: Dict[str, Any]) -> str:
    ds, de, days, thr = r.get("period_start"), r.get("period_end"), r.get("days"), r.get("change_percent")
    rows = r.get("rows", [])
    head = f"{ds}~{de} 거래량 스파이크(최근 {days}일 평균 대비"
    head += f" ≥ {thr}%):" if thr is not None else "):"
    lines = [f"{head} 총 {len(rows)}건"]
    for row in rows[:15]:
        lines.append(f"  - {row.get('trade_date')} {row.get('name')} | 거래량 {_fmt_num(row.get('volume'))}, 변화율 {_fmt_num(row.get('vol_spike_pct'))}%")
    return "\n".join(lines)

def _r_task3_moving_average_diff(r: Dict[str, Any]) -> str:
    ds, de, period, diffp, direction = r.get("period_start"), r.get("period_end"), r.get("period"), r.get("diff_percentage"), r.get("direction")
    rows = r.get("rows", [])
    dir_txt = "상단 이탈" if direction == "above" else "하단 이탈"
    head = f"{ds}~{de} {period}일 SMA 대비 괴리 {diffp}% {dir_txt}: 총 {len(rows)}건"
    body = [f"  - {row.get('trade_date')} {row.get('name')} | 종가 {_fmt_num(row.get('close'))}, 괴리 {_fmt_num(row.get('pct_diff'))}%"
            for row in rows[:15]]
    return "\n".join([head] + body)

def _r_task3_cross_events(r: Dict[str, Any]) -> str:
    ds, de, cts = r.get("period_start"), r.get("period_end"), r.get("cross_types", [])
    rows = r.get("rows", [])
    head = f"{ds}~{de} 이동평균 크로스 이벤트({', '.join(cts) or '전체'}): 총 {len(rows)}건"
    body = [f"  - {row.get('trade_date')} {row.get('name')} | {row.get('cross_type')}" for row in rows[:15]]
    return "\n".join([head] + body)

def _r_task3_rsi(r: Dict[str, Any]) -> str:
    ds, de, thr, cond = r.get("period_start"), r.get("period_end"), r.get("threshold"), r.get("condition")
    cond_txt = "과매수" if cond == "overbought" else "과매도"
    rows = r.get("rows", [])
    head = f"{ds}~{de} RSI {thr} 기준 {cond_txt}: 총 {len(rows)}건"
    body = [f"  - {row.get('trade_date')} {row.get('name')} | RSI {_fmt_num(row.get('rsi'))}" for row in rows[:15]]
    return "\n".join([head] + body)

def _r_task3_bollinger(r: Dict[str, Any]) -> str:
    ds, de, band, touch = r.get("period_start"), r.get("period_end"), r.get("band"), r.get("touch")
    rows = r.get("rows", [])
    touch_txt = "터치" if touch else "밴드 계산"
    head = f"{ds}~{de} 볼린저밴드 {band} {touch_txt}: 총 {len(rows)}건"
    body = [f"  - {row.get('trade_date')} {row.get('name')} | 종가 {_fmt_num(row.get('close'))}" for row in rows[:15]]
    return "\n".join([head] + body)

def _r_task3(r: Dict[str, Any]) -> str:
    t = r.get("type")
    if t == "volume_signal":     return _r_task3_volume_signal(r)
    if t == "moving_average_diff": return _r_task3_moving_average_diff(r)
    if t == "cross_events":      return _r_task3_cross_events(r)
    if t == "rsi":               return _r_task3_rsi(r)
    if t == "bollinger_band":    return _r_task3_bollinger(r)
    return "시그널 결과를 렌더링할 수 없습니다."