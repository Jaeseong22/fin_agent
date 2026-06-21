#!/usr/bin/env python3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "graph"))

from prompt_templates import build_prompt


PROMPT_NAMES = (
    "task_classifier",
    "parsing_task1",
    "parsing_task2",
    "parsing_task3",
)


def main() -> None:
    load_dotenv()
    client = Client()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    for name in PROMPT_NAMES:
        url = client.push_prompt(
            name,
            object=build_prompt(name, llm),
            description="FinAgent interview/demo prompt",
            tags=["fin-agent", "interview-demo"],
        )
        print(f"[ok] {name}: {url}")


if __name__ == "__main__":
    main()
