from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class AgentSettings(BaseSettings):
    llm_model: str = "gpt-oss:120b-cloud"
    llm_temperature: float = 0.0
    ollama_url: str = "http://localhost:11434"

    checkpointer_type: str = "sqlite"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/agent_memory"
    sqlite_checkpoint_path: str = str(Path(__file__).parent.parent / "agent_memory.db")

    health_db_path: str = str(Path(__file__).parent.parent / "config" / "health.db")
    sql_max_rows: int = 100
    sql_timeout_seconds: int = 10

    table_whitelist: list[str] = ["daily_body_metrics", "users"]
    schema_whitelist: list[str] = ["main"]

    max_chart_data_points: int = 500
    top_k: int = 5
    max_tool_retries: int = 2
    agent_timeout_seconds: int = 60

    audit_log_enabled: bool = True
    audit_log_path: str = str(Path(__file__).parent.parent / "agent_audit.log")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


agent_settings = AgentSettings()
