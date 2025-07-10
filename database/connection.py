"""
Database connection manager with connection pooling.
"""

import asyncpg
from typing import Optional
from functools import wraps
from .config import get_db_url, get_pool_config
from cachetools import TTLCache
from .config import get_cache_config

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

# Initialize cache
cache_config = get_cache_config()
_cache = TTLCache(
    maxsize=cache_config['max_size'],
    ttl=cache_config['ttl']
)

async def get_pool() -> asyncpg.Pool:
    """Get or create the database connection pool."""
    global _pool
    if _pool is None:
        pool_config = get_pool_config()
        _pool = await asyncpg.create_pool(
            get_db_url(),
            min_size=pool_config['min_size'],
            max_size=pool_config['max_size'],
            max_queries=pool_config['max_queries'],
            max_inactive_connection_lifetime=pool_config['max_inactive_connection_lifetime'],
        )
    return _pool

async def close_pool():
    """Close the database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

def cached(key_prefix: str):
    """Decorator for caching database query results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a cache key from the function arguments
            cache_key = f"{key_prefix}:{':'.join(str(arg) for arg in args)}:{':'.join(f'{k}={v}' for k, v in kwargs.items())}"
            
            # Try to get the result from cache
            if cache_key in _cache:
                return _cache[cache_key]
            
            # Execute the function and cache the result
            result = await func(*args, **kwargs)
            _cache[cache_key] = result
            return result
        return wrapper
    return decorator

def paginate(page: int = 1, page_size: Optional[int] = None):
    """Decorator for adding pagination to database queries."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, page=page, page_size=page_size, **kwargs):
            from .config import get_pagination_config
            
            pagination_config = get_pagination_config()
            if page_size is None:
                page_size = pagination_config['default_page_size']
            
            # Ensure page_size doesn't exceed the maximum
            page_size = min(page_size, pagination_config['max_page_size'])
            
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Add pagination parameters to the query
            kwargs['limit'] = page_size
            kwargs['offset'] = offset
            
            # Execute the query
            result = await func(*args, **kwargs)
            
            return {
                'data': result,
                'page': page,
                'page_size': page_size,
                'has_more': len(result) == page_size
            }
        return wrapper
    return decorator
