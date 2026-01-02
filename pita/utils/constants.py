"""
Redis configuration constants for the PITA package.

This module defines Redis connection parameters that can be overridden via
environment variables. Redis is used for inter-process communication between
the main process and logits processors during inference.

Environment Variables:
    REDIS_HOST: Redis server hostname (default: 'localhost')
    REDIS_PORT: Redis server port (default: 6379)
    REDIS_PASSWORD: Redis authentication password (default: None)
"""
import os

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
