# Valkey Manager

::: pita.utils.valkey_manager

## Overview

The `valkey_manager` module provides utilities for managing Valkey connections and data storage, primarily used for inter-process communication between the main inference process and logits processors in vLLM and TensorRT backends.

## Use Cases

Valkey is used in PITA for:

- **Logits Processor Communication**: vLLM and TensorRT run logits processors in separate threads. These processors calculate normalization constants and entropy values, storing them in Valkey for retrieval by the main process.
- **Temporary Data Storage**: Metrics are stored temporarily and automatically cleaned up after retrieval.
- **Multi-threaded Coordination**: Enables safe data sharing between the inference engine threads and the main application.

## Configuration

Valkey connection parameters are configured via environment variables (see [constants](constants.md)):

- `VALKEY_HOST`: Valkey server hostname (default: "localhost")
- `VALKEY_PORT`: Valkey server port (default: 6379)
- `VALKEY_DB`: Valkey database number (default: 0)

## Related Components

- [vLLM Logits Processor](../inference/vllm_logits_processor.md): Uses Valkey to store computed metrics
- [TensorRT Backend](../inference/tensorRT_backend.md): Uses Valkey for logits processing
- [Constants](constants.md): Valkey configuration constants

## Requirements

Valkey is required for the following backends when `logits_processor=True`:

- vLLM backend with entropy or power distribution metrics
- TensorRT backend with entropy or power distribution metrics

Install Valkey:
```bash
# Ubuntu/Debian
sudo apt-get install valkey-server

# macOS
brew install valkey

# Or via pip for Python client
pip install valkey
```

Start Valkey server:
```bash
valkey-server
```

## Example Usage

The Valkey manager is typically used internally by the logits processors, but you can access it directly if needed:

```python
from pita.utils.valkey_manager import ValkeyManager
from pita.utils.constants import VALKEY_HOST, VALKEY_PORT, VALKEY_DB

# Initialize Valkey connection (usually done internally)
valkey_client = ValkeyManager(
    host=VALKEY_HOST,
    port=VALKEY_PORT,
    db=VALKEY_DB
)

# The logits processors use Valkey to store metrics
# These are automatically retrieved and cleaned up by the sampler
```

For most users, Valkey integration is transparent - simply ensure Valkey is running when using vLLM or TensorRT backends with `logits_processor=True`.
