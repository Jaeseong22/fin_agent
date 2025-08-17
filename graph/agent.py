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
                    rewrite_query_with_human_feedback)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langsmith import Client

client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
task_classifier_prompt = client.pull_prompt("task_classifier", include_model=True)

llm = ChatOpenAI(temperature=0.2, 
                 model="gpt-4o-mini",)

def task_classifier(state: State) -> Command[Literal["query_parsing", "asking_to_human","chatbot"]]:
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
                goto="asking_to_human",
                update={
                    "ask_human": False,
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

def query_parsing(state: State) -> Command[Literal["db_check", "asking_to_human"]]:
    messages = state["messages"][-1]
    task = state["task_name"]
    task_obj = state["task"]

    match task:
        case "Task1":
            parsing_prompt = client.pull_prompt("parsing_task1", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="asking_to_human")
            payload = data.get("Task1", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="asking_to_human")
            return Command(goto="db_check", update={"task": updated_task})
        
        case "Task2":
            parsing_prompt = client.pull_prompt("parsing_task2", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="asking_to_human")
            payload = data.get("Task2", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="asking_to_human")
            return Command(goto="db_check", update={"task": updated_task})
            
        case "Task3":
            parsing_prompt = client.pull_prompt("parsing_task3", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="asking_to_human")
            payload = data.get("Task3", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="asking_to_human")
            return Command(goto="db_check", update={"task": updated_task})
        
        case _:
            return Command(goto="asking_to_human")
        
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
            return Command(goto="asking_to_human", update={"ask_human": True})
        
def task1(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task1_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"answer": [result]})

def task2(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task2_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"answer": [result]})

def task3(state: State) -> Command[Literal["llm_answer"]]:
    task_obj = state["task"]
    result = run_task3_query(task_obj, ENGINE)
    return Command(goto="llm_answer", update={"answer": [result]})

def asking_to_human(state: State) -> Command[Literal["await_human", "emit_question"]]:
    asked = state.get("ask_human", False)
    if asked:
        return Command(goto="await_human")
    else:
        q = state.get("human_question") or \
        "원하시는 조건을 더 구체적으로 알려주세요. 예) 기간, 시장(kospi, kosdaq), 신호 종류 등"
        return Command(goto="emit_question", update={"human_question": q})
    
def emit_question(state: State) -> None:
    """
    화면에 질문만 출력하고 run을 종료(다음 턴까지 대기).
    구현 방식은 UI에 따라 다르지만, 보통은 state['answer']나 UI 바인딩 필드를 갱신하면 됩니다.
    """
    q = state.get("human_question") or "원하시는 조건을 더 구체적으로 알려주세요. 예) 기간, 시장(kospi, kosdaq), 신호 종류 등"
    state["human_question"] = q
    state["messages"] = state["messages"] + [AIMessage(content=q)]
    state["question"] = [q]
    state["ask_human"] = True

def await_human(state: State) -> Optional[Command[Literal["rewrite_query"]]]:
    """
    다음 턴(사람이 답한 뒤)에 호출된다 가정.
    """
    if isinstance(state["messages"][-1], HumanMessage):
        # 사람 답변 도착 → 쿼리 재작성 단계로
        return Command(goto="rewrite_query", update={"ask_human": False})
    return None

def rewrite_query(state: State) -> Command[Literal["task_classifier"]]:
    """
    사람의 추가 정보를 기존 질의와 병합해 messages를 정제하고,
    파이프라인을 처음 단계(task_classifier)로 되돌린다.
    """
    update = rewrite_query_with_human_feedback(state)
    return Command(goto="task_classifier", update=update)

def ambiguity


init_state = {
    "messages": [HumanMessage(content="삼전이 적삼병일 때를 알려줘")],
    "task_name": "Task3",
    "task": Task3(company_name=None, market=None, period_start="", period_end="",
                  signal_type=[], mode="list"),
    "answer": [],            # 필수 초기화
    "ask_human": False,
    "human_question": None,
    "question": [],          # 질문 기록 필드
}