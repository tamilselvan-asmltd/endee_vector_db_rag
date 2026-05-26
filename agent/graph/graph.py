from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from agent.state import AgentState
from agent.graph.nodes import (
    intent_router_node,
    rag_node,
    sql_node,
    chart_node,
    synthesis_node,
    format_response_node,
)
from agent.memory.checkpointer import create_checkpointer
from agent.config import agent_settings
from agent.utils.logger import logger


def build_agent() -> StateGraph:
    llm = ChatOllama(
        base_url=agent_settings.ollama_url,
        model=agent_settings.llm_model,
        temperature=agent_settings.llm_temperature,
        keep_alive="5m",
    )

    builder = StateGraph(AgentState)

    builder.add_node("intent_router", lambda state: intent_router_node(state, llm))
    builder.add_node("rag", rag_node)
    builder.add_node("sql", sql_node)
    builder.add_node("chart", chart_node)
    builder.add_node("synthesis", lambda state: synthesis_node(state, llm))
    builder.add_node("format_response", format_response_node)

    builder.add_edge(START, "intent_router")

    def route_from_intent(state: AgentState) -> Literal["rag", "sql", "synthesis"]:
        intent = state.get("intent", "general")
        if intent == "rag":
            return "rag"
        elif intent in ("sql", "chart", "mixed"):
            return "sql"
        return "synthesis"

    builder.add_conditional_edges("intent_router", route_from_intent, {
        "rag": "rag",
        "sql": "sql",
        "synthesis": "synthesis",
    })

    def route_from_rag(state: AgentState) -> Literal["synthesis", "sql"]:
        intent = state.get("intent", "general")
        if intent == "mixed":
            return "sql"
        return "synthesis"

    builder.add_conditional_edges("rag", route_from_rag, {
        "synthesis": "synthesis",
        "sql": "sql",
    })

    def route_from_sql(state: AgentState) -> Literal["synthesis", "chart"]:
        intent = state.get("intent", "general")
        if intent == "chart":
            return "chart"
        sql_results = state.get("sql_results")
        if sql_results and len(sql_results) > 0:
            return "chart"
        return "synthesis"

    builder.add_conditional_edges("sql", route_from_sql, {
        "synthesis": "synthesis",
        "chart": "chart",
    })

    builder.add_edge("chart", "synthesis")
    builder.add_edge("synthesis", "format_response")
    builder.add_edge("format_response", END)

    checkpointer = create_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("LangGraph agent compiled successfully")
    return graph


agent_graph = build_agent()
