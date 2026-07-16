import logging
import threading
from typing import Dict, Any, Optional
import redis

logger = logging.getLogger("risklayer.state")

class StateStore:
    """Interface for distributed state management."""
    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError
        
    def set(self, key: str, value: str, expire: Optional[int] = None) -> None:
        raise NotImplementedError
        
    def incr(self, key: str) -> int:
        raise NotImplementedError
        
    def rpush(self, key: str, value: str) -> int:
        raise NotImplementedError
        
    def lrange(self, key: str, start: int, end: int) -> list:
        raise NotImplementedError
        
    def delete(self, key: str) -> None:
        raise NotImplementedError

class InMemoryStateStore(StateStore):
    """Thread-safe in-memory state store for fallback operations."""
    def __init__(self):
        self._store: Dict[str, str] = {}
        self._lists: Dict[str, list] = {}
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
            
    def rpush(self, key: str, value: str) -> int:
        with self._lock:
            if key not in self._lists:
                self._lists[key] = []
            self._lists[key].append(str(value))
            return len(self._lists[key])
            
    def lrange(self, key: str, start: int, end: int) -> list:
        with self._lock:
            lst = self._lists.get(key, [])
            if end == -1:
                return lst[start:]
            return lst[start:end+1]
            
    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
            self._lists.pop(key, None)

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
        
    def rpush(self, key: str, value: str) -> int:
        return self.client.rpush(key, value)
        
    def lrange(self, key: str, start: int, end: int) -> list:
        return self.client.lrange(key, start, end)
        
    def delete(self, key: str) -> None:
        self.client.delete(key)

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
