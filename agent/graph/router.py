from langchain_ollama import ChatOllama
from agent.config import agent_settings


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a health monitoring assistant.

Given a user query, classify it into ONE of these intents:

- "rag": Questions about technical documentation, manuals, SOPs, maintenance guides, or engineering knowledge (from uploaded PDFs).
- "sql": Questions about personal health metrics data, body measurements, vitals, trends, or any stored health data.
- "chart": Requests for visualizations, charts, graphs, or plots of health data. Always paired with "sql" intent.
- "mixed": Questions that need BOTH RAG documents AND health database data. E.g., comparing SOP recommendations with actual health readings.
- "general": General conversation, greetings, or questions that don't need tools.

Examples:
- "What does the SOP say about compressor maintenance?" -> rag
- "Show my average blood sugar for the last 30 days" -> sql
- "Plot my weight trend this month" -> sql (because we need data first, then chart)
- "Compare the manual's recommended BP range with my actual readings" -> mixed
- "Who are you?" -> general

Return ONLY the intent keyword, nothing else.
"""


def classify_intent(query: str, llm: ChatOllama) -> str:
    response = llm.invoke(
        [
            ("system", INTENT_CLASSIFICATION_PROMPT),
            ("human", query),
        ]
    )
    intent = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()
    valid = {"rag", "sql", "chart", "mixed", "general"}
    if intent not in valid:
        return "general"
    if intent == "chart":
        return "sql"
    return intent
