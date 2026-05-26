from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
from agent.config import agent_settings
from agent.utils.logger import logger


_retriever_instance = None


def set_retriever(retriever) -> None:
    global _retriever_instance
    _retriever_instance = retriever


@tool
def rag_retrieval(query: str) -> dict:
    """Retrieve relevant documents from the RAG knowledge base.

    Use this tool when the user asks about technical documentation,
    operating procedures, maintenance guides, or any information
    that might be in uploaded PDFs.
    """
    if _retriever_instance is None:
        return {"documents": [], "citations": [], "error": "Retriever not initialized"}

    try:
        docs: List[Document] = _retriever_instance.invoke(query)
        citations = []
        seen = set()
        for d in docs:
            link = d.metadata.get("link", "")
            if link and link not in seen:
                seen.add(link)
                citations.append({
                    "text": d.page_content[:300],
                    "filename": d.metadata.get("filename", "Unknown"),
                    "page": d.metadata.get("page", "?"),
                    "link": link,
                    "score": d.metadata.get("rerank_score", 0),
                })
        return {
            "documents": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            "citations": citations,
        }
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return {"documents": [], "citations": [], "error": str(e)}
