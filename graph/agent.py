from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("fin_agent")

import os
import json
from typing import Literal
from schema import State, Task1, Task2, Task3
from utils import parse_tool_json, check_task1, check_task2, check_task3, ENGINE
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langsmith import Client

client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
task_classifier_prompt = client.pull_prompt("task_classifier", include_model=True)

llm = ChatOpenAI(temperature=0.2, 
                 model="gpt-4o-mini",)

def task_classifier(state: State) -> Command[Literal["query_parsing"]]:
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
        case "Chatbot":
            return Command(goto="chatbot")
       
def chatbot(state: State) -> Command[Literal["llm_answer"]]:
    messages = state["messages"][-1]
    result = llm.invoke(messages)
    return Command(goto="llm_answer",
                   update={"answer": result.content})

def query_parsing(state: State) -> Command[Literal["db_check", "human"]]:
    messages = state["messages"][-1]
    task = state["task_name"]
    task_obj = state["task"]

    match task:
        case "Task1":
            parsing_prompt = client.pull_prompt("parsing_task1", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="human")
            payload = data.get("Task1", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="human")
            return Command(goto="db_check", update={"task": updated_task})
        
        case "Task2":
            parsing_prompt = client.pull_prompt("parsing_task2", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="human")
            payload = data.get("Task2", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="human")
            return Command(goto="db_check", update={"task": updated_task})
            
        case "Task3":
            parsing_prompt = client.pull_prompt("parsing_task3", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            data = parse_tool_json(result)
            if not data:
                return Command(goto="human")
            payload = data.get("Task3", data)
            try:
                updated_task = task_obj.model_copy(update=payload)
            except Exception as e:
                print("parsing validation error:", e)
                return Command(goto="human")
            return Command(goto="db_check", update={"task": updated_task})
        
def db_check(state: State) -> Command[Literal["task1", "task2", "task3", "human"]]:
    task_class = state["task_name"]   # "Task1" | "Task2" | "Task3"
    task_obj   = state["task"]

    match task_class:
        case "Task1":
            ok, err, info = check_task1(task_obj)
            if not ok:
                return Command(goto="human")
            return Command(goto="task1")

        case "Task2":
            ok, err, info = check_task2(task_obj)
            if not ok:
                return Command(goto="human")
            return Command(goto="task2")

        case "Task3":
            ok, err, info = check_task3(task_obj)
            if not ok:
                return Command(goto="human")
            return Command(goto="task3")

        case _:
            return Command(goto="human")

test = {
    "messages": [HumanMessage(content="2025-02-17에 거래량이 20일 평균 대비 300% 이상 급증한 종목을 알려줘")],
    "task_name": "Task3",
    "task": Task3(company_name=None, market=None, period_start="", period_end="",
                  signal_type=[], mode="list"),
}
out1 = query_parsing(test)
print("query_parsing =>", out1)

if out1.goto == "db_check":
    next_state = {**test, **(out1.update or {})}
    out2 = db_check(next_state)
    print("db_check =>", out2)