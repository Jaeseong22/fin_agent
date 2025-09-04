from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("fin_agent")

import os
import json
from typing import Literal, Optional
from schema import State, Task1, Task2, Task3
from utils import (parse_tool_json,
                    check_task1,
                    check_task2, 
                    check_task3, 
                    ENGINE, 
                    run_task1_query, 
                    run_task2_query, 
                    run_task3_query,
                    rewrite_query_with_human_feedback,
                    apply_ambiguity_resolution,
                    _pick_best_result,
                    _r_task1_market_comparison,
                    _r_task1_market_index_comparison,
                    _r_task1_metric_rank,
                    _r_task1_simple_lookup,
                    _r_task1_stock_comparison,
                    _r_task1_stock_rank,
                    _r_task1_stock_to_market_ratio,
                    _r_task2,
                    _r_task3)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.graph import END
from langsmith import Client

# 맨 위 근처에 상수 추가
DEFAULT_Q = "원하시는 조건을 더 구체적으로 알려주세요. 예) 기간, 시장(KOSPI/KOSDAQ), 신호 종류 등"

client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
task_classifier_prompt = client.pull_prompt("task_classifier", include_model=True)

llm = ChatOpenAI(temperature=0.2, 
                 model="gpt-4o-mini",)

def task_classifier(state: State) -> Command[Literal["query_parsing", "ask_human","chatbot"]]:
    messages = state["messages"][-1]
    chain = task_classifier_prompt
    result = chain.invoke({"messages" : messages.content})
    tool_calls = result.additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        # 분류 실패 시 챗봇으로
        return Command(goto="chatbot")

    try:
        arguments_str = tool_calls[0]["function"]["arguments"]
        arguments = json.loads(arguments_str)
        task = arguments.get("Task")  # "Task1"|"Task2"|"Task3"|"Chatbot"
    except Exception as e:
        print("task parse error:", e)
        return Command(goto="chatbot")

    print("Detected task:", task)

    match task:
        case "Task1":
            task_instance = Task1(date="", market=None, clauses=[])
            return Command(goto="query_parsing",
                           update={"task": task_instance, "task_name": task})
        case "Task2":
            task_instance = Task2(date="", market=None, clauses=[])
            return Command(goto="query_parsing",
                           update={"task": task_instance, "task_name": task})
        case "Task3":
            task_instance = Task3(company_name=None, market=None,
                                  period_start="", period_end="",
                                  signal_type=[], mode="list")
            return Command(goto="query_parsing",
                           update={"task": task_instance, "task_name": task})
        case "Task4":
            return Command(
                goto="ask_human",
                update={
                    "ask_human": True,
                    "human_question": "어떤 관점의 금융 질문인지 구체화해 주세요. 예) 날짜/기간, 시장(KOSPI/KOSDAQ), 전수검색 여부, 신호(예: 이동평균/RSI/거래량 스파이크) 등"
                }
            ) 
        case "Chatbot":
            return Command(goto="chatbot")
        
        case _:
            return Command(goto="chatbot")
       
def chatbot(state: State) -> Command[Literal["llm_answer"]]:
    messages = state["messages"][-1]
    result = llm.invoke(messages)
    return Command(goto="llm_answer",
                   update={"answer": [result.content]})

def query_parsing(state: State) -> Command[Literal["db_check", "ask_human"]]:
    messages = state["messages"][-1]
    task = state["task_name"]
    task_obj = state["task"]

    match task:
        case "Task1":
            parsing_prompt = client.pull_prompt("parsing_task1", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="ask_human")
            payload = data.get("Task1", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="ask_human")
            return Command(goto="db_check", update={"task": updated_task})
        
        case "Task2":
            parsing_prompt = client.pull_prompt("parsing_task2", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="ask_human")
            payload = data.get("Task2", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="ask_human")
            return Command(goto="db_check", update={"task": updated_task})
            
        case "Task3":
            parsing_prompt = client.pull_prompt("parsing_task3", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="ask_human")
            payload = data.get("Task3", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="ask_human")
            return Command(goto="db_check", update={"task": updated_task})
        
        case _:
            return Command(goto="ask_human")
        
def db_check(state: State) -> Command[Literal["task1", "task2", "task3", "ambiguity_handler"]]:
    task_class = state["task_name"]
    task_obj   = state["task"]

    match task_class:
        case "Task1":
            if not check_task1(task_obj):
                return Command(goto="ambiguity_handler")
            return Command(goto="task1")

        case "Task2":
            if not check_task2(task_obj):
                return Command(goto="ambiguity_handler")
            return Command(goto="task2")

        case "Task3":
            if not check_task3(task_obj):
                return Command(goto="ambiguity_handler")
            return Command(goto="task3")

        case _:
            return Command(goto="ask_human", update={"ask_human": True})
        
def task1(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task1_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"task1_result": result})

def task2(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task2_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"task2_result": result})

def task3(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task3_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"task3_result": result})


def ask_human(state: State) -> Command[Literal["task_classifier"]]:
    """
    1) 질문을 메시지에 남기고
    2) interrupt()로 실행을 멈춰 사용자 입력을 기다림
    3) 재개 시 사람의 답변을 기존 질의와 병합 → task_classifier로 점프
    """
    q = state.get("human_question")

    # 질문을 대화 흐름에 남겨둔다
    messages = state["messages"] + [AIMessage(content=q)]

    # 🔴 여기서 그래프 실행 '정지' → 동일 thread로 HumanMessage가 도착하면 다음 줄부터 재개
    human_reply: HumanMessage = interrupt(
        value={"reason": "need_more_info", "question": q}
    )

    # 사용자 답변을 messages에 붙인다
    resumed_state = dict(state)
    resumed_state["messages"] = messages + [human_reply]
    resumed_state["ask_human"] = False
    resumed_state["human_question"] = q

    # 기존 병합 유틸 재사용 (원질의 + 추가 입력을 합침)
    merged = rewrite_query_with_human_feedback(resumed_state)

    # 분류 단계로 되돌림
    return Command(goto="task_classifier", update=merged)

def ambiguity_handler(state: State) -> Command[Literal["task1", "task2", "task3", "ask_human"]]:
    last = state["messages"][-1]
    query_text = getattr(last, "content", "")

    upd = apply_ambiguity_resolution(query_text)
    new_state = {"normalized_query": upd.get("normalized_query")}
    if "clarified_fields" in upd:
        prev = state.get("clarified_fields") or {}
        merged = {**prev, **upd["clarified_fields"]}
        new_state["clarified_fields"] = merged
    if "term_hits" in upd:
        new_state["term_hits"] = upd["term_hits"]
    
    task_name = state.get("task_name")
    task_obj  = state.get("task")
    
    def _has(v): return v is not None and v != "" and v != []

    if task_name == "Task3":
        # 최소: 기간 + 신호 표현(정규화된 문장에 패턴/지표 단어가 들어가거나 term_hits 존재)
        has_period = _has(getattr(task_obj, "period_start", None)) or _has(getattr(task_obj, "period_end", None))
        has_signal = bool(new_state.get("term_hits"))
        if has_period and has_signal:
            # 바로 task3 실행
            return Command(goto="task3", update=new_state)
        else:
            # 사람에게 물어볼 질문 제안
            q = []
            if not has_period:
                q.append("기간(예: 2024-01-01~2024-12-31)")
            if not has_signal:
                q.append("신호 종류(예: RSI 과매수/볼린저 상단 터치/골든크로스 등)")
            ask = "분석을 위해 추가 정보가 필요합니다. " + " / ".join(q) + " 를 알려주세요."
            return Command(goto="ask_human", update={**new_state, "human_question": ask, "ask_human": False})

    elif task_name == "Task2":
        has_date = _has(getattr(task_obj, "date", None))
        has_clauses = _has(getattr(task_obj, "clauses", None))
        if has_date and has_clauses:
            return Command(goto="task2", update=new_state)
        else:
            ask = "조건 검색을 위해 날짜와 조건(등락률/거래량 등)을 구체화해 주세요. 예: 2025-06-13, 등락률 ≥ 7%, 거래량 전일 대비 ≥ 300%."
            return Command(goto="ask_human", update={**new_state, "human_question": ask, "ask_human": False})

    elif task_name == "Task1":
        has_date = _has(getattr(task_obj, "date", None))
        if has_date:
            return Command(goto="task1", update=new_state)
        else:
            ask = "조회할 날짜(또는 기간)를 알려주세요. 예: 2025-01-21."
            return Command(goto="ask_human", update={**new_state, "human_question": ask, "ask_human": False})

    # 혹시 모르면 사람에게
    return Command(goto="ask_human", update={**new_state, "human_question": "원하시는 조건을 조금 더 구체적으로 알려주세요.", "ask_human": False})

def llm_answer(state: State) -> Command[END]:
    """
    - state에서 최신 결과(dict)를 고르고
    - 유형별 자연어 요약 생성
    - END로 종료
    """
    result = _pick_best_result(state or {})
    if not isinstance(result, dict):
        # chatbot 경로에서 이미 answer가 들어온 경우는 바로 종료
        if state.get("answer"):
            return Command(goto=END)
        return Command(goto=END, update={"answer": ["결과가 없어 답변을 생성하지 못했습니다."]})

    status = result.get("status", "ok")
    if status != "ok":
        reason = result.get("reason") or "오류가 발생했습니다."
        return Command(goto=END, update={"answer": [f"요청을 처리하지 못했습니다. 사유: {reason}"], "result": result})

    rtype = result.get("type")
    if rtype in {
        "metric_rank", "market_comparison", "stock_comparison",
        "simple_lookup", "stock_rank", "stock_to_market_ratio",
        "market_index_comparison"
    }:
        if rtype == "metric_rank":
            answer = _r_task1_metric_rank(result)
        elif rtype == "market_comparison":
            answer = _r_task1_market_comparison(result)
        elif rtype == "stock_comparison":
            answer = _r_task1_stock_comparison(result)
        elif rtype == "simple_lookup":
            answer = _r_task1_simple_lookup(result)
        elif rtype == "stock_rank":
            answer = _r_task1_stock_rank(result)
        elif rtype == "stock_to_market_ratio":
            answer = _r_task1_stock_to_market_ratio(result)
        elif rtype == "market_index_comparison":
            answer = _r_task1_market_index_comparison(result)
        else:
            answer = "알 수 없는 Task1 결과 유형입니다."
    else:
        if "applied_clauses" in result:       # Task2
            answer = _r_task2(result)
        elif result.get("period_start") and result.get("period_end"):  # Task3
            answer = _r_task3(result)
        else:
            rows = result.get("rows", [])
            answer = f"결과 {len(rows)}건이 조회되었습니다."
            if rows:
                sample = rows[:5]
                bullets = [f"  - {s.get('trade_date','')} {s.get('name','')}" for s in sample]
                answer += "\n예시:\n" + "\n".join(bullets)

    nq = state.get("normalized_query")
    if nq:
        answer = f"질의: {nq}\n\n" + answer

    return Command(goto=END, update={"answer": [answer], "result": result})