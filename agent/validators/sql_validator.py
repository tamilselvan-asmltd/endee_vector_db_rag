import re
from typing import Optional
from agent.config import agent_settings


READ_ONLY_KEYWORDS = {"select"}
DISALLOWED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "replace", "grant", "revoke", "exec", "execute",
    "call", "import", "load", "attach", "detach", "pragma",
}

WHITELISTED_TABLES = set(agent_settings.table_whitelist)
MAX_ROWS = agent_settings.sql_max_rows


class SQLValidationError(Exception):
    pass


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().lower())


def extract_statements(sql: str) -> list[str]:
    parts = re.split(r";", sql)
    return [s.strip() for s in parts if s.strip()]


def validate_sql_statement(stmt: str) -> str:
    stmt_lower = stmt.lower().strip()

    has_disallowed = any(kw in stmt_lower.split() for kw in DISALLOWED_KEYWORDS)
    if has_disallowed:
        raise SQLValidationError("Only SELECT statements are allowed")

    first_token = stmt_lower.split()[0] if stmt_lower.split() else ""
    if first_token not in READ_ONLY_KEYWORDS:
        raise SQLValidationError(f"Statement must begin with SELECT, got '{first_token}'")

    tokens = re.split(r"[^a-zA-Z_0-9]", stmt_lower)
    mentioned_tables = {t for t in tokens if t in WHITELISTED_TABLES}

    if not mentioned_tables:
        allowed = ", ".join(sorted(WHITELISTED_TABLES))
        raise SQLValidationError(
            f"Query must reference at least one whitelisted table ({allowed})"
        )

    return stmt


def enforce_row_limit(sql: str) -> str:
    sql_stripped = sql.strip().rstrip(";").strip()
    sql_lower = sql_stripped.lower()

    if re.search(r"\blimit\s+\d+", sql_lower):
        return sql_stripped + ";"
    else:
        return f"{sql_stripped} LIMIT {MAX_ROWS};"


def validate_and_sanitize(sql: str) -> str:
    statements = extract_statements(sql)
    validated = []
    for stmt in statements:
        if stmt:
            validated.append(validate_sql_statement(stmt))
    if not validated:
        raise SQLValidationError("No valid SQL statements found")

    joined = "; ".join(validated)
    return enforce_row_limit(joined)
