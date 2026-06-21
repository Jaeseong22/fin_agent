from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from schema import Task1, Task2, Task3


class TaskClassification(BaseModel):
    Task: Literal["Task1", "Task2", "Task3", "Task4", "Chatbot"] = Field(
        description="질문의 처리 유형"
    )


SYSTEM_PROMPTS = {
    "task_classifier": """
당신은 한국 주식 질의 라우터입니다.
- Task1: 특정 날짜의 단순 조회, 종목 비교, 순위, 시장 평균 비교
- Task2: 특정 날짜에 수치 조건을 만족하는 종목 검색
- Task3: 기간 내 RSI, 이동평균, 골든/데드크로스, 볼린저밴드, 거래량 신호 분석
- Task4: 금융 질문이지만 의도가 지나치게 불명확함
- Chatbot: 인사나 일반 대화
반드시 가장 적합한 유형 하나를 선택하세요.
""",
    "parsing_task1": """
한국 주식 단순 조회 질문을 Task1 스키마로 변환하세요.
날짜는 YYYY-MM-DD로 작성하고, 명시되지 않은 값은 null로 두세요.
clauses의 type은 metric_rank, market_comparison, stock_comparison,
simple_lookup, stock_rank, stock_to_market_ratio, market_index_comparison 중 하나입니다.
종목의 특정 지표 조회는 simple_lookup을 사용하세요.
""",
    "parsing_task2": """
한국 주식 조건 검색 질문을 Task2 스키마로 변환하세요.
날짜는 YYYY-MM-DD로 작성하세요.
조건 type은 pct(등락률), vol_pct(거래량 변화율), vol_abs(절대 거래량),
price_range(가격 범위) 중 하나입니다.
""",
    "parsing_task3": """
한국 주식 기술적 신호 질문을 Task3 스키마로 변환하세요.
기간은 YYYY-MM-DD로 작성하세요.
signal_type의 type은 volume_signal, moving_average_diff, cross_events,
rsi, bollinger_band 중 하나입니다.
질문에 종목명이 없으면 company_name은 null로 두세요.
""",
}


SCHEMAS = {
    "task_classifier": TaskClassification,
    "parsing_task1": Task1,
    "parsing_task2": Task2,
    "parsing_task3": Task3,
}


def build_prompt(name: str, llm):
    if name not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    schema = SCHEMAS[name]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPTS[name].strip()),
            ("human", "{messages}"),
        ]
    )
    return prompt | llm.bind_tools([schema], tool_choice=schema.__name__)
