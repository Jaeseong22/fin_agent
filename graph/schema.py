from typing import (
    Sequence,
    List,
    Union,
    Optional,
    Literal,
    NotRequired,
)
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

# 1. 특정 metric 기준 상위/하위 종목 조회
class MetricRankClause(TypedDict):
    type: Literal["metric_rank"]
    metric: Literal["등락률", "거래량", "시가총액", "종가"]
    rank_type: Literal["top", "bottom"]
    top_n: Optional[int]

# 2. 시장 평균과 비교
class MarketComparisonClause(TypedDict):
    type: Literal["market_comparison"]
    metric: Literal["등락률", "거래량"]
    comparison: Literal["above_avg", "below_avg"]  # 시장 평균보다 높은지 낮은지

# 3. 종목 간 비교 (2개 종목)
class StockComparisonClause(TypedDict):
    type: Literal["stock_comparison"]
    metric: Literal["등락률", "시가총액", "거래량", "종가"]
    stock_a: str
    stock_b: str

class SimpleMetricLookupClause(TypedDict):
    type: Literal["simple_lookup"]
    company_name: str
    metric: Literal["시가", "고가", "저가", "종가", "거래량", "시가총액", "지수", "등락률"]

class StockRankClause(TypedDict):
    type: Literal["stock_rank"]
    company_name: str
    metric: Literal["등락률", "거래량", "시가총액"]

class StockToMarketRatioClause(TypedDict):
    type: Literal["stock_to_market_ratio"]
    company_name: str
    metric: Literal["거래량", "시가총액"]

class MarketIndexComparisonClause(TypedDict):
    type: Literal["market_index_comparison"]
    indices: List[Literal["KOSPI", "KOSDAQ"]]
    metric: Literal["전일 대비 상승률", "지수"]

# 최종 Clause Union
Task1Clause = Annotated[
    Union[MetricRankClause,
          MarketComparisonClause,
          StockComparisonClause,
          SimpleMetricLookupClause,
          MarketIndexComparisonClause,
          StockRankClause,
          StockToMarketRatioClause],
    Field(discriminator="type")
]

class PctClause(TypedDict):
    type: Literal["pct"]
    op: Literal[">=", "<=", ">", "<"]
    value_pct: float

class VolPctClause(TypedDict):
    type: Literal["vol_pct"]
    op: Literal[">=", "<=", ">", "<"]
    value_pct: float

class VolAbsClause(TypedDict):
    type: Literal["vol_abs"]
    op: Literal[">=", "<=", ">", "<"]
    shares: int

class PriceRangeClause(TypedDict):
    type: Literal["price_range"]
    low: Optional[float]
    high: Optional[float]

Clause = Annotated[
    Union[PctClause, VolPctClause, VolAbsClause, PriceRangeClause],
    Field(discriminator="type"),
]

class VolumeSignalClause(TypedDict):
    type: Literal["volume_signal"]
    days: Optional[int]  # 예: 3일 평균
    change_percent: Optional[float]  # 예: 50이면 50%

class MovingAverageClause(TypedDict):
    type: Literal["moving_average_diff"]
    period: int  # 이동평균 기준 (예: 5, 20)
    diff_percentage: float  # 현재가가 n일 이동평균보다 몇 % 높은지/낮은지
    direction: Literal["above", "below"]

class CrossEventsClause(TypedDict):
    type: Literal["cross_events"]   # 이벤트 기반 카운트/리스트를 의도적으로 분리
    cross_types: List[Literal["golden", "death"]]  # ['golden','death'] 같이 다중 지원

class RSIClause(TypedDict):
    type: Literal["rsi"]
    threshold: float  # 예: 70, 30
    condition: Literal["overbought", "oversold"]  # 과매수/과매도

class BollingerClause(TypedDict):
    type: Literal["bollinger_band"]
    band: Literal["upper", "lower"]
    touch: bool

SignalClause = Annotated[
    Union[VolumeSignalClause, MovingAverageClause, RSIClause, BollingerClause, CrossEventsClause],
    Field(discriminator="type"),
]

# ---------- task1 / task2 / task3 ----------
class Task1(BaseModel):
    date: Optional[Union[str, List[str]]] = None
    market: Optional[Union[str, List[str]]] = None
    clauses: Optional[List[Task1Clause]] = Field(default_factory=list)

class Task2(BaseModel):
    date: str
    market: Optional[str] = None
    clauses: Optional[List[Clause]] = Field(default_factory=list)

class Task3(BaseModel):
    company_name: Optional[str] = Field(None, description="특정 종목명 (없으면 전체 검색)")
    market: Optional[str] = Field(None, description="시장 (예: KOSPI, KOSDAQ)")
    period_start: Optional[str] = Field(..., description="조회 시작일 (YYYY-MM-DD)")
    period_end: Optional[str] = Field(..., description="조회 종료일 (YYYY-MM-DD)")
    signal_type: Optional[List[SignalClause]] = Field(default=None, description="기술적 조건 (단일 또는 배열 입력 가능)")
    mode: Optional[Literal["count", "list", "both"]] = Field(default="list", description="조회 모드 (개수, 목록, 둘 다)")
# ---------- 상태 정의 ----------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: BaseModel
    answer: Annotated[List, operator.add]
    task_name: str
    ask_human: NotRequired[bool]
    human_question: NotRequired[Optional[str]]
    question: List[str]