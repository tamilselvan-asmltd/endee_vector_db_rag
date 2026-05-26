import json
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from agent.state import AgentState, UIComponent
from agent.tools.rag_tool import rag_retrieval
from agent.tools.sql_tool import sql_query
from agent.tools.chart_tool import generate_chart
from agent.graph.router import classify_intent
from agent.utils.logger import logger
from agent.utils.security import audit_logger, detect_prompt_injection


SYSTEM_PROMPT = """You are an intelligent health monitoring assistant with access to:
1. A RAG knowledge base of technical/medical documentation
2. A personal health metrics database (blood sugar, BP, heart rate, sleep, etc.)
3. Chart/visualization generation capabilities

Answer the user's question using the tool outputs provided below.

When you have SQL results and a chart would help explain the data:
- Include a clear analysis of the numbers
- Tell the user a chart has been generated (the chart will appear below your response)

When you have RAG context:
- Cite the source document name
- Quote relevant passages

CRITICAL RULES:
- Respond ONLY in plain natural language.
- Do NOT include raw JSON, SQL queries, tool names, tool schemas, or technical specifications in your response.
- Do NOT output anything that looks like a data structure or code block.
- Your entire response must be conversational text that a non-technical user would understand.
"""


SYNTHESIS_PROMPT = PromptTemplate.from_template("""{system_prompt}

=== DATA FOR YOUR ANALYSIS ===

SQL Results (use these to answer the user's question):
{sql_results}

RAG Context (use if relevant to the question):
{rag_context}

Chart Generated (a chart has been created and will appear below your text response):
{chart_summary}

=== USER QUERY ===
{user_query}

=== YOUR RESPONSE ===
""")


def intent_router_node(state: AgentState, llm: ChatOllama) -> dict:
    query = state.get("user_query", "")
    session_id = state.get("session_id", "unknown")

    if detect_prompt_injection(query):
        logger.warning(f"Prompt injection detected: {query[:100]}")
        return {"intent": "general", "error": "Query blocked by security filter"}

    intent = classify_intent(query, llm)
    logger.info(f"Intent classified: {intent} for query: {query[:100]}")

    audit_logger.log_tool_call(session_id, "intent_router", query)

    return {"intent": intent}


def rag_node(state: AgentState) -> dict:
    query = state.get("user_query", "")
    session_id = state.get("session_id", "unknown")

    logger.info(f"Executing RAG tool for: {query[:100]}")
    result = rag_retrieval.invoke({"query": query})

    audit_logger.log_tool_call(session_id, "rag_retrieval", query)

    docs = result.get("documents", [])
    citations = result.get("citations", [])

    return {
        "retrieved_docs": docs,
        "tool_trace": (state.get("tool_trace") or []) + [
            {"tool": "rag", "doc_count": len(docs), "citation_count": len(citations)}
        ],
    }


def sql_node(state: AgentState) -> dict:
    query = state.get("user_query", "")
    session_id = state.get("session_id", "unknown")

    logger.info(f"Executing SQL tool for: {query[:100]}")
    result = sql_query.invoke({"natural_language_query": query})

    audit_logger.log_tool_call(session_id, "sql_query", query)

    rows = result.get("rows", [])
    executed_sql = result.get("query_executed", "")
    error = result.get("error")

    return {
        "sql_results": rows,
        "sql_query": executed_sql,
        "tool_trace": (state.get("tool_trace") or []) + [
            {"tool": "sql", "row_count": len(rows), "error": error}
        ],
    }


def chart_node(state: AgentState) -> dict:
    sql_results = state.get("sql_results", [])
    if not sql_results:
        return {"chart_payloads": []}

    columns = list(sql_results[0].keys()) if sql_results else []
    if len(columns) < 2:
        return {"chart_payloads": []}

    x_col = columns[0]
    y_col = columns[1] if len(columns) > 1 else columns[0]

    chart_result = generate_chart.invoke({
        "chart_type": "line",
        "data": sql_results,
        "x_column": x_col,
        "y_column": y_col,
        "title": f"{y_col} by {x_col}",
    })

    payloads = [chart_result] if "error" not in chart_result else []

    return {
        "chart_payloads": (state.get("chart_payloads") or []) + payloads,
        "tool_trace": (state.get("tool_trace") or []) + [
            {"tool": "chart", "payloads": len(payloads)}
        ],
    }


def synthesis_node(state: AgentState, llm: ChatOllama) -> dict:
    user_query = state.get("user_query", "")
    retrieved_docs = state.get("retrieved_docs") or []
    sql_results = state.get("sql_results") or []
    chart_payloads = state.get("chart_payloads") or []

    rag_context = ""
    if retrieved_docs:
        rag_context = "\n\n".join(
            [d.get("page_content", str(d))[:500] if isinstance(d, dict) else d.page_content[:500]
             for d in retrieved_docs[:5]]
        )

    sql_context = json.dumps(sql_results[:20], indent=2) if sql_results else "No SQL data"

    chart_summary = "No chart generated"
    if chart_payloads:
        names = []
        for cp in chart_payloads:
            ctype = cp.get("chart_type", "chart")
            title = cp.get("title", "")
            names.append(f"{ctype} chart: {title}" if title else f"{ctype} chart")
        chart_summary = "; ".join(names)

    prompt = SYNTHESIS_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        rag_context=rag_context,
        sql_results=sql_context,
        chart_summary=chart_summary,
        user_query=user_query,
    )

    response = llm.invoke(prompt)

    message = response.content if hasattr(response, "content") else str(response)

    return {"final_response": message, "messages": [AIMessage(content=message)]}


def format_response_node(state: AgentState) -> dict:
    message = state.get("final_response", "")
    chart_payloads = state.get("chart_payloads") or []

    ui_components = []
    for cp in chart_payloads:
        if isinstance(cp, dict) and "component" in cp:
            ui_components.append(UIComponent(
                type=cp["component"],
                chart_type=cp.get("chart_type"),
                payload=cp.get("payload", {}),
                title=cp.get("title", ""),
            ))

    return {
        "ui_components": ui_components,
    }
