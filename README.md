# Fin_agent란

fin_agent는 국내 주식시장 데이터를 이해하기 어려워하는 사용자를 위해
“질의 → 분석 → 응답”의 전체 과정을 자동화한 금융 분석 LLM 에이전트입니다.
한국 주식시장(코스피·코스닥) 데이터를 기반으로 사용자의 자연어 질의를 자동으로 분석하고,
주가·지표·신호·패턴을 SQL 기반으로 조회하여 자연어로 응답하는 금융 특화 LLM 에이전트입니다.
LangGraph로 상태 전이를 관리하며, LangSmith로 프롬프트·실행을 추적해
안정적인 금융 분석 워크플로우를 제공합니다.

예시 입력 및 출력:
	•	입력: “삼성전자 vs LG전자 최근 3개월 수익률 비교”, “RSI 70 돌파 종목 알려줘”
	•	분류: Task1(단순조회/랭킹) · Task2(조건검색) · Task3(신호/패턴) · Chatbot
	•	실행: Task별 SQL/로직 실행 → 자연어 요약 응답

	• 목적: 국내 종목 기반 ‘질의 → 분류 → SQL 분석 → 자연어 응답’의 전체 흐름을 자동화

⸻

## Features
• 자연어 → Task 분류  
   LangSmith에 저장된 task_classifier 프롬프트 기반 라우팅

• 정밀 파싱  
   parsing_task1/2/3 + Pydantic 스키마 검증

• DB 질의/후처리  
   yfinance 기반 ETL → ENGINE 조회

• 모호성 처리  
   필수 필드 누락 시 Clarification 질문 자동 생성

• 유형별 응답 템플릿  
   _r_task1_*, _r_task2, _r_task3의 일관된 포맷

⸻

## Architecture

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

전체 그래프는 입력 → 분류 → 파싱 → DB 검증 → SQL 실행 → 응답의
안정적인 처리 파이프라인을 LangGraph로 구현한 구조입니다.

⸻

## Installation

### 1) Python >= 3.10 권장
python -V

### 2) 가상환경
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

### 3) 패키지 설치
pip install -U pip
pip install -r requirements.txt

필수 패키지(예시)
langgraph, langchain-core, langchain-openai, langsmith, pydantic, python-dotenv, sqlalchemy(+드라이버), yfinance

⸻

## Configuration

루트에 .env 생성:

OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
# DB 연결 문자열(예: SQLite/MySQL/Postgres)
DATABASE_URL=sqlite:///./stock_data.db

⸻

## Core Graph (노드 동작 요약)

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

## Ambiguity & Error Handling
	•	Ambiguity: 필수정보 누락 → ambiguity_handler가 부족 항목만 묻는 맞춤 질문 생성
	•	Validation: 파싱 실패/Pydantic 오류 → ask_human로 전환
	•	DB 오류: run_task*_query 예외는 {status:"error", reason}로 래핑 → llm_answer가 사용자 친화 메시지 생성
	•	Fallback: 분류 실패 → chatbot
