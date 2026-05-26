import json
import time
from datetime import datetime
from pathlib import Path
from agent.config import agent_settings


class AuditLogger:
    def __init__(self):
        self.enabled = agent_settings.audit_log_enabled
        self.log_path = Path(agent_settings.audit_log_path)

    def log(
        self,
        event: str,
        session_id: str,
        user_query: str = "",
        intent: str = "",
        sql_query: str = "",
        tool: str = "",
        status: str = "success",
        error: str = "",
    ):
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "session_id": session_id,
            "user_query": user_query[:500],
            "intent": intent,
            "sql_query": sql_query[:1000],
            "tool": tool,
            "status": status,
            "error": error[:500],
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_tool_call(self, session_id: str, tool: str, query: str, status: str = "success", error: str = ""):
        self.log(
            event="tool_call",
            session_id=session_id,
            user_query=query,
            tool=tool,
            status=status,
            error=error,
        )

    def log_sql(self, session_id: str, sql: str, status: str = "success", error: str = ""):
        self.log(
            event="sql_execution",
            session_id=session_id,
            sql_query=sql,
            status=status,
            error=error,
        )


audit_logger = AuditLogger()


SENSITIVE_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "forget your instructions",
    "you are now",
    "system prompt",
    "you have been",
]


def detect_prompt_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in SENSITIVE_PATTERNS:
        if pattern in text_lower:
            return True
    return False
