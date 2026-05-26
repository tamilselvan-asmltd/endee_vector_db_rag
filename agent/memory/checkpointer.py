import sqlite3
from contextlib import closing
from agent.config import agent_settings
from agent.utils.logger import logger


def create_checkpointer():
    checkpointer_type = agent_settings.checkpointer_type

    if checkpointer_type == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection

            conn = Connection.connect(agent_settings.postgres_dsn)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            logger.info("PostgreSQL checkpointer initialized")
            return checkpointer
        except Exception as e:
            logger.warning(f"PostgreSQL checkpointer failed ({e}), falling back to SQLite")

    if checkpointer_type == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver
            import redis

            r = redis.Redis(
                host=getattr(agent_settings, "redis_host", "localhost"),
                port=getattr(agent_settings, "redis_port", 6379),
                password=getattr(agent_settings, "redis_password", ""),
                decode_responses=False,
            )
            checkpointer = RedisSaver(r)
            checkpointer.setup()
            logger.info("Redis checkpointer initialized")
            return checkpointer
        except Exception as e:
            logger.warning(f"Redis checkpointer failed ({e}), falling back to SQLite")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(
            agent_settings.sqlite_checkpoint_path,
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(conn)
        try:
            checkpointer.setup()
        except Exception:
            pass
        logger.info(f"SQLite checkpointer initialized at {agent_settings.sqlite_checkpoint_path}")
        return checkpointer
    except Exception as e:
        logger.warning(f"SQLite checkpointer failed ({e}), falling back to in-memory")

    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()
    logger.info("In-memory checkpointer initialized (no persistence)")
    return checkpointer
