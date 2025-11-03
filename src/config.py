"""Database configuration sourced from environment variables."""

import os


def _get_env_var(key: str, default=None, required: bool = False):
    """Fetch an environment variable, optionally enforcing its presence."""
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Environment variable '{key}' is required but not set.")
    return value


DB_CONFIG = {
    'host': _get_env_var('MAPLE_DB_HOST', required=True),
    'port': int(_get_env_var('MAPLE_DB_PORT', default='3306')),
    'user': _get_env_var('MAPLE_DB_USER', required=True),
    'password': _get_env_var('MAPLE_DB_PASSWORD', required=True),
    'database': _get_env_var('MAPLE_DB_NAME', required=True),
    'charset': _get_env_var('MAPLE_DB_CHARSET', default='utf8mb4')
}

TABLE_NAME = _get_env_var('MAPLE_TABLE_NAME', default='auction_history')
