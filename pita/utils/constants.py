"""
Valkey configuration constants for the PITA package.

This module defines Valkey connection parameters that can be overridden via
environment variables. Valkey is used for inter-process communication between
the main process and logits processors during inference.

Environment Variables:
    VALKEY_HOST: Valkey server hostname (default: 'localhost')
    VALKEY_PORT: Valkey server port (default: 6379)
    VALKEY_PASSWORD: Valkey authentication password (default: None)
"""
import os

VALKEY_HOST = os.environ.get('VALKEY_HOST', 'localhost')
VALKEY_PORT = int(os.environ.get('VALKEY_PORT', 6379))
VALKEY_PASSWORD = os.environ.get('VALKEY_PASSWORD', None)
