from graph1 import build_graph
from schema import State
from langchain_core.messages import HumanMessage

def main():
    graph = build_graph()

    original_query = input().strip()
    current_query = original_query

    while True:
        init_state: State = {
            "messages": [HumanMessage(content=current_query)],
        }
        result_state = graph.invoke(init_state)

        interrupts = result_state.get("__interrupt__") or []
        if not interrupts:
            break

        interrupt_value = getattr(interrupts[0], "value", {}) or {}
        question = (
            interrupt_value.get("question")
            or result_state.get("human_question")
            or "추가 정보를 입력해 주세요."
        )
        print(f"\n=== 추가 질문 ===\n{question}")

        clarification = input().strip()
        if not clarification:
            print("\n추가 입력이 없어 실행을 종료합니다.")
            return
        current_query = f"{original_query}\n추가 조건: {clarification}"

    print("\n=== 답변 ===")
    answers = result_state.get("answer") or []
    print("\n".join(answers) if isinstance(answers, list) else answers)

if __name__ == "__main__":
    main()
