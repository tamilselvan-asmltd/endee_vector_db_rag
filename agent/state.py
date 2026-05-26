from typing import TypedDict, List, Optional, Any, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import MessagesState


class UIComponent(TypedDict):
    type: str
    chart_type: Optional[str]
    payload: dict
    title: Optional[str]


class AgentState(MessagesState):
    user_query: str
    intent: Optional[str]
    retrieved_docs: Optional[List[Document]]
    sql_results: Optional[List[dict]]
    sql_query: Optional[str]
    chart_payloads: Optional[List[UIComponent]]
    final_response: Optional[str]
    ui_components: Optional[List[UIComponent]]
    tool_trace: Optional[List[dict]]
    error: Optional[str]
    session_id: Optional[str]
