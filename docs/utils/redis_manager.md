# Redis Manager

::: pita.utils.redis_manager

## Overview

The `redis_manager` module provides utilities for managing Redis connections and data storage, primarily used for inter-process communication between the main inference process and logits processors in vLLM and TensorRT backends.

## Use Cases

Redis is used in PITA for:

- **Logits Processor Communication**: vLLM and TensorRT run logits processors in separate threads. These processors calculate normalization constants and entropy values, storing them in Redis for retrieval by the main process.
- **Temporary Data Storage**: Metrics are stored temporarily and automatically cleaned up after retrieval.
- **Multi-threaded Coordination**: Enables safe data sharing between the inference engine threads and the main application.

## Configuration

Redis connection parameters are configured via environment variables (see [constants](constants.md)):

- `REDIS_HOST`: Redis server hostname (default: "localhost")
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)

## Related Components

- [vLLM Logits Processor](../inference/vllm_logits_processor.md): Uses Redis to store computed metrics
- [TensorRT Backend](../inference/tensorRT_backend.md): Uses Redis for logits processing
- [Constants](constants.md): Redis configuration constants

## Requirements

Redis is required for the following backends when `logits_processor=True`:

- vLLM backend with entropy or power distribution metrics
- TensorRT backend with entropy or power distribution metrics

Install Redis:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Or via pip for Python client
pip install redis>=4.0.0
```

Start Redis server:
```bash
redis-server
```

## Example Usage

The Redis manager is typically used internally by the logits processors, but you can access it directly if needed:

```python
from pita.utils.redis_manager import RedisManager
from pita.utils.constants import REDIS_HOST, REDIS_PORT, REDIS_DB

# Initialize Redis connection (usually done internally)
redis_client = RedisManager(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB
)

# The logits processors use Redis to store metrics
# These are automatically retrieved and cleaned up by the sampler
```

For most users, Redis integration is transparent - simply ensure Redis is running when using vLLM or TensorRT backends with `logits_processor=True`.
