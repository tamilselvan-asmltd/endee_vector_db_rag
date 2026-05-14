import logging
import redis
from typing import List, Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """
    Manages session-based chat history using Redis.
    Uses RedisChatMessageHistory from LangChain for persistence.
    Ensures history is trimmed to a maximum number of messages.
    """

    def __init__(self, max_messages: int = settings.max_history_messages):
        self.redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
        if settings.redis_password:
            self.redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}"
        
        self.max_messages = max_messages
        self.key_prefix = settings.redis_chat_history_prefix
        
        # Redis client for administrative tasks (like clear all)
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True
        )
        logger.info(f"[*] ChatHistoryManager initialized with prefix: {self.key_prefix}, max_messages: {self.max_messages}")

    def _get_redis_history(self, session_id: str) -> RedisChatMessageHistory:
        """
        Internal helper to get the RedisChatMessageHistory instance for a session.
        Session ID format: user_id:conversation_id
        """
        return RedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url,
            key_prefix=self.key_prefix
        )

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """
        Retrieves the full chat history for a given session.
        """
        try:
            history = self._get_redis_history(session_id)
            return history.messages
        except Exception as e:
            logger.error(f"[-] Error retrieving history for session {session_id}: {str(e)}")
            return []

    def add_user_message(self, session_id: str, message: str):
        """
        Adds a user message to the session history and trims if necessary.
        """
        try:
            history = self._get_redis_history(session_id)
            history.add_user_message(message)
            self._trim_history(session_id)
        except Exception as e:
            logger.error(f"[-] Error adding user message for session {session_id}: {str(e)}")

    def add_ai_message(self, session_id: str, message: str):
        """
        Adds an AI message to the session history and trims if necessary.
        """
        try:
            history = self._get_redis_history(session_id)
            history.add_ai_message(message)
            self._trim_history(session_id)
        except Exception as e:
            logger.error(f"[-] Error adding AI message for session {session_id}: {str(e)}")

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[BaseMessage]:
        """
        Returns the last 'limit' messages from the session history.
        """
        messages = self.get_history(session_id)
        return messages[-limit:] if messages else []

    def clear_history(self, session_id: str):
        """
        Deletes all history for the given session.
        """
        try:
            history = self._get_redis_history(session_id)
            history.clear()
            logger.info(f"[*] History cleared for session: {session_id}")
        except Exception as e:
            logger.error(f"[-] Error clearing history for session {session_id}: {str(e)}")

    def clear_all_histories(self):
        """Clears ALL chat histories across all sessions."""
        cursor = 0
        pattern = f"{self.key_prefix}*"
        count = 0
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self.redis_client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break
        logger.info(f"[*] Cleared {count} total chat history keys.")
        return count

    def _trim_history(self, session_id: str):
        """
        Trims the history to keep only the last max_messages.
        Note: RedisChatMessageHistory doesn't have a built-in trim method in older versions,
        so we manually manage it by getting all and slicing if needed, or using Redis commands.
        Since we want to be production-grade and efficient:
        """
        try:
            history = self._get_redis_history(session_id)
            messages = history.messages
            if len(messages) > self.max_messages:
                # If history exceeds limit, we clear and re-add only the last N
                # This is a bit heavy but ensures consistency with LangChain's abstraction.
                # Alternatively, we could use history.redis_client.ltrim if we had direct access.
                recent_messages = messages[-self.max_messages:]
                history.clear()
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        history.add_user_message(msg.content)
                    elif isinstance(msg, AIMessage):
                        history.add_ai_message(msg.content)
                logger.info(f"[*] Trimmed history for session {session_id} to {self.max_messages} messages.")
        except Exception as e:
            logger.error(f"[-] Error trimming history for session {session_id}: {str(e)}")

    def register_user(self, user_id: str):
        """Registers a user in a persistent global set."""
        self.redis_client.sadd("users:registry", user_id)
        logger.info(f"[*] Registered user: {user_id}")

    def list_all_users(self) -> List[str]:
        """Discovers all unique users from Redis registry and active keys."""
        # 1. Get from registry
        registered_users = [u.decode('utf-8') if isinstance(u, bytes) else u 
                           for u in self.redis_client.smembers("users:registry")]
        
        # 2. Discover from active chat keys (backup/legacy)
        cursor = 0
        pattern = f"{self.key_prefix}*"
        active_users = set()
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                # Key format: chat:user_id:conversation_id
                # Use split with maxsplit to preserve user_id if it contains colons (unlikely)
                # but definitely handle conversation_id containing colons.
                parts = key.split(":", 2)
                if len(parts) >= 2:
                    active_users.add(parts[1])
            if cursor == 0:
                break
        
        # Combine and sort
        all_users = set(registered_users).union(active_users)
        return sorted(list(all_users))

    def list_user_sessions(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a specific user."""
        cursor = 0
        pattern = f"{self.key_prefix}{user_id}:*"
        sessions = set()
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                # Key format: chat:user_id:conversation_id
                # We want the 3rd part onwards.
                parts = key.split(":", 2)
                if len(parts) >= 3:
                    sessions.add(parts[2])
            if cursor == 0:
                break
        return sorted(list(sessions))

    def rename_session(self, user_id: str, old_conv_id: str, new_conv_id: str):
        """Renames a conversation ID by changing the Redis key."""
        old_key = f"{self.key_prefix}{user_id}:{old_conv_id}"
        new_key = f"{self.key_prefix}{user_id}:{new_conv_id}"
        
        if self.redis_client.exists(old_key):
            self.redis_client.rename(old_key, new_key)
            logger.info(f"[*] Renamed session: {old_key} -> {new_key}")
            return True
        return False

    def delete_user(self, user_id: str):
        """Removes a user from the registry and wipes all their data."""
        # 1. Remove from registry
        self.redis_client.srem("users:registry", user_id)
        
        # 2. Wipe chat keys
        keys = self.redis_client.keys(f"chat:{user_id}:*")
        if keys:
            self.redis_client.delete(*keys)
        
        logger.info(f"[*] Deleted user {user_id} and purged all data.")
        return True
