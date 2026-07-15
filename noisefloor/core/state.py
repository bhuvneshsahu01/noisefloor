import logging
import threading
from typing import Dict, Any, Optional
import redis

logger = logging.getLogger("noisefloor.state")

class StateStore:
    """Interface for distributed state management."""
    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError
        
    def set(self, key: str, value: str, expire: Optional[int] = None) -> None:
        raise NotImplementedError
        
    def incr(self, key: str) -> int:
        raise NotImplementedError

class InMemoryStateStore(StateStore):
    """Thread-safe in-memory state store for fallback operations."""
    def __init__(self):
        self._store: Dict[str, str] = {}
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._store.get(key)
            
    def set(self, key: str, value: str, expire: Optional[int] = None) -> None:
        with self._lock:
            self._store[key] = str(value)
            
    def incr(self, key: str) -> int:
        with self._lock:
            curr = int(self._store.get(key, 0))
            new_val = curr + 1
            self._store[key] = str(new_val)
            return new_val

class RedisStateStore(StateStore):
    """Redis-backed distributed state store."""
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, db: int = 0):
        self.client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            decode_responses=True,
            socket_timeout=1.0,
            socket_connect_timeout=1.0
        )
        
    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)
        
    def set(self, key: str, value: str, expire: Optional[int] = None) -> None:
        self.client.set(key, value, ex=expire)
        
    def incr(self, key: str) -> int:
        return self.client.incr(key)

def get_state_store(host: str = "127.0.0.1", port: int = 6379) -> StateStore:
    """
    Factory function that attempts to connect to Redis.
    Falls back gracefully to InMemoryStateStore if Redis is unavailable.
    """
    try:
        store = RedisStateStore(host=host, port=port)
        # Ping test to verify connection
        store.client.ping()
        logger.info("Connected to Redis distributed state store.")
        return store
    except Exception as e:
        logger.warning(f"Redis is unavailable ({str(e)}). Falling back to InMemoryStateStore.")
        return InMemoryStateStore()
