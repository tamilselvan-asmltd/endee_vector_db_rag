import json
import sqlite3
import time
import re
from typing import Optional
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from agent.config import agent_settings
from agent.validators.sql_validator import validate_and_sanitize, SQLValidationError
from agent.utils.logger import logger
from agent.utils.security import audit_logger


_llm_instance: Optional[ChatOllama] = None


def set_llm(llm: ChatOllama) -> None:
    global _llm_instance
    _llm_instance = llm


SCHEMA_DESCRIPTION = """
Table: daily_body_metrics
Columns: id, user_id, measurement_date, blood_sugar_fasting, blood_sugar_post_meal,
         systolic_bp, diastolic_bp, heart_rate_bpm, oxygen_saturation_percent,
         body_temperature_celsius, body_weight_kg, bmi, sleep_hours,
         water_intake_liters, steps_count, exercise_minutes,
         calories_consumed, calories_burned, stress_level, mood_status,
         medications_taken, symptoms_notes, created_at, updated_at

Table: users
Columns: id, email, display_name, created_at, updated_at

Foreign Key: daily_body_metrics.user_id -> users.id

Useful date patterns (SQLite):
- 'last 7 days' -> measurement_date >= date('now', '-7 days')
- 'this month' -> strftime('%%Y-%%m', measurement_date) = strftime('%%Y-%%m', 'now')
- 'last 30 days' -> measurement_date >= date('now', '-30 days')
- 'today' -> measurement_date = date('now')
"""


NL2SQL_SYSTEM_PROMPT = f"""You are a SQL generator for a health metrics database.
Given a natural language question, generate a safe SQLite SELECT query.

Rules:
- SELECT only, read-only
- Only tables: daily_body_metrics, users
- Use date('now') for current date
- Always GROUP BY when using aggregation
- Use CAST for division when needed
- Return ONLY the raw SQL, no markdown, no explanation

Schema:
{SCHEMA_DESCRIPTION}
"""


def _nl_to_sql(natural_query: str) -> str:
    if _llm_instance is None:
        msg = "LLM not initialized for NL-to-SQL conversion"
        logger.error(msg)
        return f"SELECT 1 WHERE 0 -- {msg}"

    try:
        response = _llm_instance.invoke(
            [
                ("system", NL2SQL_SYSTEM_PROMPT),
                ("human", natural_query),
            ]
        )
        sql = response.content if hasattr(response, "content") else str(response)
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```", "", sql).strip()
        sql = re.sub(r";\s*$", "", sql).strip()
        return sql
    except Exception as e:
        logger.error(f"NL-to-SQL conversion failed: {e}")
        return f"SELECT 1 WHERE 0 -- conversion error"


@tool
def sql_query(natural_language_query: str) -> dict:
    """Query the health metrics database using natural language.

    Use this tool when the user asks about health data, trends,
    averages, or any metrics stored in the daily_body_metrics table.
    Converts natural language to safe, read-only SQL and executes it.
    """

    raw_sql = ""
    try:
        raw_sql = _nl_to_sql(natural_language_query)
        validated_sql = validate_and_sanitize(raw_sql)
        logger.info(f"SQL generated: {validated_sql}")

        audit_logger.log_sql("agent_session", validated_sql)

        db_path = agent_settings.health_db_path
        timeout = agent_settings.sql_timeout_seconds
        max_rows = agent_settings.sql_max_rows

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        start = time.perf_counter()
        cur = conn.execute(validated_sql)
        rows = [dict(row) for row in cur.fetchmany(max_rows)]
        elapsed = time.perf_counter() - start

        conn.close()

        def serialize(val):
            if isinstance(val, (int, float)):
                return val
            return str(val)

        serialized_rows = [
            {k: serialize(v) for k, v in row.items()} for row in rows
        ]

        return {
            "query_executed": validated_sql,
            "rows": serialized_rows,
            "row_count": len(serialized_rows),
            "execution_time_seconds": round(elapsed, 4),
        }

    except SQLValidationError as e:
        logger.warning(f"SQL validation failed: {e}")
        return {
            "query_executed": raw_sql,
            "rows": [],
            "row_count": 0,
            "error": f"SQL validation error: {e}",
        }
    except sqlite3.Error as e:
        logger.error(f"SQL execution error: {e}")
        return {
            "query_executed": raw_sql,
            "rows": [],
            "row_count": 0,
            "error": f"Database error: {e}",
        }
    except Exception as e:
        logger.error(f"SQL tool error: {e}")
        return {
            "query_executed": raw_sql,
            "rows": [],
            "row_count": 0,
            "error": str(e),
        }
