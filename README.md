⸻

fin_agent

국내(코스피/코스닥) 전용 자연어 금융 에이전트
사용자 질의를 해석해 yfinance 기반 DB(📅 2023-08-10 ~ 2025-08-08 데이터 커버리지)를 조회하고, 주가/지표/패턴 결과를 자연어로 응답합니다.
LangGraph로 상태 기계, LangSmith로 프롬프트/실행을 추적합니다.

TL;DR
	•	입력: “삼성전자 vs LG전자 최근 3개월 수익률 비교”, “RSI 70 돌파 종목 알려줘”
	•	분류: Task1(단순조회/랭킹) · Task2(조건검색) · Task3(신호/패턴) · Chatbot
	•	실행: Task별 SQL/로직 실행 → 자연어 요약 응답

⸻

✨ Features
	•	자연어 → 작업 분류(Task1/2/3/Chatbot)
LangSmith에 저장된 task_classifier 프롬프트로 라우팅
	•	정밀 파싱
parsing_task1|2|3 프롬프트 → Pydantic 스키마 유효성 검증
	•	DB 질의/후처리
yfinance 기반 ETL로 업데이트된 DB를 ENGINE으로 조회
	•	모호성 처리(Clarification)
필수 필드 미비 시 자동 질문 → 사용자 응답 병합 후 재시도
	•	유형별 응답 템플릿
_r_task1_*, _r_task2, _r_task3로 일관된 결과 포맷
	•	LangSmith 트레이싱
langchain_teddynote.logging.langsmith("fin_agent")

⸻

🗺️ Architecture

flowchart TD
    U[User Message] --> C[task_classifier]
    C -->|Task1| P1[parsing_task1]
    C -->|Task2| P2[parsing_task2]
    C -->|Task3| P3[parsing_task3]
    C -->|Chatbot| CB[chatbot]

    P1 --> DB1[db_check]
    P2 --> DB2[db_check]
    P3 --> DB3[db_check]

    DB1 -->|ok| T1[task1/run_task1_query]
    DB2 -->|ok| T2[task2/run_task2_query]
    DB3 -->|ok| T3[task3/run_task3_query]

    DB1 -->|ambiguous| AH[ambiguity_handler]
    DB2 -->|ambiguous| AH
    DB3 -->|ambiguous| AH
    AH -->|needs input| H[ask_human -> interrupt]
    H --> C

    T1 --> LA[llm_answer]
    T2 --> LA
    T3 --> LA
    CB --> LA
    LA --> END


⸻

📦 Installation

# 1) Python >= 3.10 권장
python -V

# 2) 가상환경
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3) 패키지 설치
pip install -U pip
pip install -r requirements.txt

필수 패키지(예시)
langgraph, langchain-core, langchain-openai, langsmith, pydantic, python-dotenv, sqlalchemy(+드라이버), yfinance 등

⸻

🔐 Configuration

루트에 .env 생성:

OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
# DB 연결 문자열(예: SQLite/MySQL/Postgres)
DATABASE_URL=sqlite:///./stock_data.db

코드 시작부:

from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("fin_agent")

OpenAI Python SDK(>=1.x)는 예전 openai.ChatCompletion 대신 새로운 인터페이스를 권장합니다.

⸻

🧱 Project Structure (예시)

fin_agent/
├─ README.md
├─ requirements.txt
├─ .env
├─ schema.py                  # State, Task1/Task2/Task3 (Pydantic)
├─ utils.py                   # run_task*_query, check_task*, _r_* 포맷터 등
├─ graph.py                   # Core Graph (본 README의 노드와 매핑)
├─ etl/
│  ├─ run_yf_etl.py          # yfinance → DB 적재 스크립트
│  └─ models.sql             # 테이블 스키마(예시)
└─ prompts/
   ├─ task_classifier.json   # LangSmith Prompt(원격 관리 가능)
   ├─ parsing_task1.json
   ├─ parsing_task2.json
   └─ parsing_task3.json


⸻

🧠 Core Graph (노드 동작 요약)

1) 분류기 — task_classifier(state) -> Command[goto]
	•	입력: state["messages"][-1]
	•	동작:
	•	LangSmith의 task_classifier 프롬프트를 pull → .invoke
	•	첫 번째 tool_call의 arguments JSON 파싱 → "Task" 추출
	•	Task1|Task2|Task3|Chatbot|Task4(ask_human) 분기
	•	해당 Task Pydantic 인스턴스 초기화 후 goto
	•	실패 시: chatbot 폴백

2) 챗봇 — chatbot(state) -> llm_answer
	•	ChatOpenAI(model="gpt-4o-mini", temperature=0.2)로 일반 대화 응답
	•	분류 실패/스몰토크 등 폴백 경로

3) 파싱 — query_parsing(state) -> db_check | ask_human
	•	parsing_task{1|2|3} 프롬프트 호출 → 결과를 parse_tool_json
	•	Pydantic 모델(Task1|2|3)에 model_copy(update=payload)로 검증
	•	실패 시: ask_human

4) 준비 점검 — db_check(state) -> task{1|2|3} | ambiguity_handler
	•	필수 필드 체크:
	•	Task1: date
	•	Task2: date & clauses
	•	Task3: period_start|period_end & 신호 단서(term_hits)
	•	미달 시: ambiguity_handler

5) 실행 — task1|task2|task3(state) -> llm_answer
	•	run_task{1|2|3}_query(task_obj, ENGINE) 호출
	•	결과는 state에 taskX_result로 저장

6) 사용자 확인 — ask_human(state) -> task_classifier
	•	AIMessage(content=q)로 부족 항목 질문
	•	interrupt(...)로 그래프 일시정지 → 사용자 입력 대기
	•	재개 시: rewrite_query_with_human_feedback로 원문+추가정보 병합 → 재분류

7) 모호성 처리 — ambiguity_handler(state) -> task{1|2|3} | ask_human
	•	apply_ambiguity_resolution(query_text)로 정규화/용어 히트 추출
	•	최소 요건 충족 시 즉시 실행, 아니면 부족 항목만 묻는 질문 생성

8) 응답 합성 — llm_answer(state) -> END
	•	_pick_best_result(state)로 최신 결과 선택
	•	Task1 유형:
metric_rank, market_comparison, stock_comparison,
simple_lookup, stock_rank, stock_to_market_ratio, market_index_comparison
	•	Task2: _r_task2(result)
	•	Task3: _r_task3(result)
	•	normalized_query가 있으면 헤더에 질의 원문을 표시해 Trace 가독성 확보

⸻

🗃️ Data & DB

ETL 개요
	•	yfinance에서 코스피/코스닥 종목을 수집(종가/거래량/시총/지표 등) → 정규화/계산 → ENGINE이 가리키는 DB에 적재
	•	신호(예: RSI, 볼린저, 골든/데드크로스)는 ETL 선계산 또는 조회 시 계산 중 선택

최소 스키마(예시)
	•	prices(symbol, trade_date, open, high, low, close, volume, ...)
	•	indicators(symbol, trade_date, rsi, macd, bb_up, bb_dn, ...)
	•	signals(symbol, trade_date, signal_type, signal_value, window, ...)

실제 스키마는 운영 ETL에 맞춰 README에 표/컬럼 정의를 추가하는 것을 권장합니다.

⸻

🚀 Quickstart

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from graph import (
    task_classifier, query_parsing, db_check,
    task1, task2, task3, ask_human, ambiguity_handler, llm_answer
)

# 1) 초기 state
state = {
    "messages": [HumanMessage(content="2025-02-11에 KOSDAQ에서 가장 비싼 종목 3개는?")],
}

# 2) 분류
cmd = task_classifier(state)

# 3) 전이 루프 (의사코드)
while True:
    goto = cmd.get("goto")
    state.update(cmd.get("update", {}))

    if goto == "query_parsing":
        cmd = query_parsing(state)
    elif goto == "db_check":
        cmd = db_check(state)
    elif goto == "task1":
        cmd = task1(state)
    elif goto == "task2":
        cmd = task2(state)
    elif goto == "task3":
        cmd = task3(state)
    elif goto == "ask_human":
        # UI/서버에서는 interrupt를 감지해 사용자 입력을 받아 이어붙인 뒤 재개
        cmd = ask_human(state)
    elif goto == "llm_answer":
        cmd = llm_answer(state)
        break
    else:
        break

print(state.get("answer", ["<no answer>"])[0])

실제 앱에서는 **LangGraph StateGraph**로 전이를 선언하고, 웹서버/봇 I/O를 연결하세요.

⸻

🙋 Prompts & Tracing (LangSmith)
	•	Prompts:
task_classifier, parsing_task1, parsing_task2, parsing_task3
→ LangSmith에 원격 등록 후 client.pull_prompt("...")로 버전/이력 관리
	•	Tracing:
.env에 LANGSMITH_API_KEY 설정 후
logging.langsmith("fin_agent") 호출 → 실행/프롬프트/토큰 집계

⸻

🔧 Local ETL (국내시장 예시)

# 장 마감 이후 일괄 업데이트(예시)
python etl/run_yf_etl.py \
  --symbols 005930.KS,000660.KS,035420.KS,035720.KS,091990.KQ \
  --since 2023-08-10 \
  --until 2025-08-08 \
  --db $DATABASE_URL

	•	*.KS = 코스피, *.KQ = 코스닥 (Yahoo Ticker 표기)
	•	RSI/MACD/볼린저밴드 등은 ETL에서 선계산해 indicators/signals에 저장하거나, 조회 시 계산 가능

⸻

⚠️ Ambiguity & Error Handling
	•	Ambiguity: 필수정보 누락 → ambiguity_handler가 부족 항목만 묻는 맞춤 질문 생성
	•	Validation: 파싱 실패/Pydantic 오류 → ask_human로 전환
	•	DB 오류: run_task*_query 예외는 {status:"error", reason}로 래핑 → llm_answer가 사용자 친화 메시지 생성
	•	Fallback: 분류 실패 → chatbot

⸻

🧪 Testing (권장)
	•	Unit:
	•	utils._r_*(응답 포맷터), check_task*(필드검증) → 고정입력/예상출력 테스트
	•	Integration:
	•	in-memory SQLite로 run_task*_query 경로 테스트
	•	소량 샘플 데이터로 ETL → 조회 → 응답까지 E2E 확인
	•	Prompt:
	•	LangSmith Prompt 테스트 런으로 분류/파싱 정확도 점검 및 회귀 테스트

⸻

📐 Design Choices
	•	Core–Satellite 분리: 분류/파싱/질의/응답을 모듈화해 디버깅 용이성 확보
	•	Interrupt 기반 UX: ask → interrupt → merge → 재분류로 모호성 최소화
	•	Typed Renderer: 결과 type 별 포맷터로 재현성/테스트 용이성 강화

⸻

🔌 Integration Tips
	•	API 서버화: FastAPI 등으로 POST /ask → state 초기화·실행 → answer 반환
	•	프런트엔드: interrupt 발생 시 질문 노출 → 사용자 입력을 동일 thread로 전달해 재개
	•	스케줄러: ETL은 cron/Airflow 등으로 정기 실행 (장 마감 이후 권장)

⸻

📄 License

프로젝트 정책에 맞는 라이선스를 명시하세요(예: MIT/Apache-2.0).

⸻

🙌 Acknowledgements

README 구성은 여러 OSS README 스타일을 참고해 정리했습니다. 프롬프트/SDK/예제는 최신 버전을 따라주세요.

⸻

✅ 다음 단계 제안
	•	실제 DB 스키마 표(컬럼/타입/인덱스) 추가
	•	ETL 실행 로그/샘플 스크린샷 첨부
	•	샘플 질의 → 응답 화면(스크린샷)
	•	주요 SQL 뷰/인덱스 설계 요약 (예: 최근 N거래일 윈도우, 랭킹용 인덱스)
