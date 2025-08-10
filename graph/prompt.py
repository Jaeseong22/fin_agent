from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("fin_agent")

import asyncio
import json
import os
from langsmith import Client

client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))
prompt = client.pull_prompt("parsing_task2", include_model=True)

with open("/Users/jaeseong/Downloads/conditional_queries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

messages = [{"messages": [item["input_data"]["message"]]} for item in data]

async def run_batch():
    results = await prompt.abatch(messages)
    output_list = []
    for input_item, output_item in zip(messages, results):
        parsed_args = None
        try:
            tool_calls = output_item.additional_kwargs.get("tool_calls", [])
            if tool_calls:
                func_args = tool_calls[0]["function"]["arguments"]
                parsed_args = json.loads(func_args)
        except Exception as e:
            parsed_args = {"error": str(e)}

        formatted = {
            "input": input_item["messages"][0],
            "parsed": parsed_args,
        }
        output_list.append(formatted)

    output_path = "/Users/jaeseong/Desktop/workplace/finance_agent/graph/conditional_queries_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

asyncio.run(run_batch())