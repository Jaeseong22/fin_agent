# sim.py
from graph1 import build_graph
from schema import State
from langchain_core.messages import HumanMessage

def main():
    # 그래프 빌드
    graph = build_graph()

    user_query = "2025-01-17에 셀트리온의 거래량 순위는?"
    init_state: State = {
        "messages": [HumanMessage(content=user_query)],
        "task": None,
        "answer": [],
        "task_name": "",
        "question": [user_query],
    }

    # 그래프 실행 (invoke는 동기 실행, 결과 state 반환)
    result_state = graph.invoke(init_state)

    print("\n=== 답변 ===")
    print(result_state.get("answer"))

if __name__ == "__main__":
    main()