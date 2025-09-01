from langgraph.graph import START, END, StateGraph
from agent import *
from schema import State

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("task_classifier", task_classifier)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("query_parsing", query_parsing)
    graph_builder.add_node("db_check", db_check)
    graph_builder.add_node("task1", task1)
    graph_builder.add_node("task2", task2)
    graph_builder.add_node("task3", task3)
    graph_builder.add_node("ask_human", ask_human)
    graph_builder.add_node("ambiguity_handler", ambiguity_handler)
    graph_builder.add_node("llm_answer", llm_answer)

    graph_builder.add_edge(START, "task_classifier")
    graph_builder.add_edge("llm_answer", END)

    return graph_builder.compile()
