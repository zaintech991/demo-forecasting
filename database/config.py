"""
Centralized database configuration with connection pooling and caching.
"""

import os
from typing import Dict, Any
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT") or 5432),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# Check required variables
required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
var_mapping = {"DB_HOST": "host", "DB_NAME": "database", "DB_USER": "user", "DB_PASSWORD": "password"}
for var in required_vars:
    config_key = var_mapping[var]
    if not DB_CONFIG[config_key]:
        raise ValueError(f"{var} environment variable is required")

# Connection pool settings
POOL_CONFIG = {
    "min_size": int(os.getenv("DB_POOL_MIN_SIZE", "5")),
    "max_size": int(os.getenv("DB_POOL_MAX_SIZE", "20")),
    "max_queries": int(os.getenv("DB_POOL_MAX_QUERIES", "50000")),
    "max_inactive_connection_lifetime": float(
        os.getenv("DB_POOL_MAX_INACTIVE_CONNECTION_LIFETIME", "300")
    ),
}

# Cache settings
CACHE_CONFIG = {
    "ttl": int(os.getenv("CACHE_TTL", "300")),  # Time to live in seconds
    "max_size": int(
        os.getenv("CACHE_MAX_SIZE", "1000")
    ),  # Maximum number of items in cache
}

# Pagination settings
PAGINATION_CONFIG = {
    "default_page_size": int(os.getenv("DEFAULT_PAGE_SIZE", "50")),
    "max_page_size": int(os.getenv("MAX_PAGE_SIZE", "100")),
}


@lru_cache()
def get_db_url() -> str:
    """Get the database URL with credentials."""
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"


@lru_cache()
def get_db_config() -> Dict[str, Any]:
    """Get the database configuration."""
    return DB_CONFIG


@lru_cache()
def get_pool_config() -> Dict[str, Any]:
    """Get the connection pool configuration."""
    return POOL_CONFIG


@lru_cache()
def get_cache_config() -> Dict[str, Any]:
    """Get the cache configuration."""
    return CACHE_CONFIG


@lru_cache()
def get_pagination_config() -> Dict[str, Any]:
    """Get the pagination configuration."""
    return PAGINATION_CONFIG
