from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("fin_agent")

import os
import json
from typing import Literal
from schema import State, Task1, Task2, Task3
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langsmith import Client

client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
task_classifier_prompt = client.pull_prompt("task_classifier", include_model=True)

llm = ChatOpenAI(temperature=0.2, 
                 model_name="gpt-4o-mini",)

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
    match task:
        case "Task1":
            parsing_prompt = client.pull_prompt("parsing_task1", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
        
        case "Task2":
            parsing_prompt = client.pull_prompt("parsing_task2", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            
        case "Task3":
            parsing_prompt = client.pull_prompt("parsing_task3", include_model=True)
            result = parsing_prompt.invoke({"messages" : messages.content})
            

test = {
    "messages": [HumanMessage(content="2025-02-17에 거래량이 20일 평균 대비 300% 이상 급증한 종목을 알려줘")],
}
output=task_classifier(test)
print(output)