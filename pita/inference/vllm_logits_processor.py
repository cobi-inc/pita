import torch
from torch.distributions import Categorical
import redis
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor, 
    BatchUpdate, 
    MoveDirectionality
)
from vllm.config import VllmConfig
from typing import Dict, Optional, List
from dataclasses import dataclass

from pita.utils.constants import REDIS_HOST, REDIS_PORT

@dataclass
class sampling_params:
    """
    Sampling parameters for logits processing requests.

    Attributes:
        req_id: Unique identifier for the request.
        normalization_constants: Whether to calculate normalization constants.
        temperature: Sampling temperature value.
        entropy: Whether to calculate entropy.
        entropy_inference: Whether entropy is used for inference decisions.
        gradient_steps: Number of gradient steps for optimization.
        learning_rate: Learning rate for optimization.
        delta: Delta value for optimization adjustments.
    """
    req_id: str
    normalization_constants: bool
    temperature: float
    entropy: bool
    entropy_inference: bool
    gradient_steps: int
    learning_rate: float
    delta: float

class LogitsLoggingProcessor(LogitsProcessor):
    """
    Custom vLLM logits processor that logs normalization constants and entropy to Redis.

    This processor intercepts logits during generation to calculate and store normalization
    constants and entropy values. These values are stored in Redis for retrieval by the
    main process after generation completes.

    Attributes:
        active_req_ids: Dictionary mapping request indices to their sampling parameters.
        redis_client: Redis client for storing computed values.
        temperature: Default temperature value.
    """
    def __init__(
         self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool
    ) -> None:
        """
        Initialize the LogitsLoggingProcessor.

        Args:
            vllm_config: vLLM configuration object.
            device: PyTorch device for tensor operations.
            is_pin_memory: Whether to use pinned memory for tensors.
        """
        self.active_req_ids: Dict[int, sampling_params] = {}
        self.redis_client = None
        self.temperature = 1.0  # Default temperature, can be configured per request

    def _ensure_redis(self) -> None:
        """
        Ensure Redis client is initialized and connected.

        Lazily initializes the Redis connection on first use to avoid connection
        issues during processor instantiation.
        """
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
                )
            except Exception as e:
                # If we can't log to Redis, we are flying blind, but try printing just in case
                print(f"CRITICAL WORKER ERROR: Redis connect failed: {e}")

    def is_argmax_invariant(self) -> bool:
        """
        Indicate whether this processor changes which token has the highest probability.

        Returns:
            False to ensure apply() is always called, even when it doesn't change argmax.
        """
        return False  # Must be False to ensure apply() is called

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Update processor state when requests are added, removed, or moved in the batch.

        This method is called by vLLM to notify the processor of batch changes. It tracks
        request IDs and their associated sampling parameters.

        Args:
            batch_update: Information about requests added, removed, or moved in the batch.
                Can be None if no updates occurred.
        """
        self._ensure_redis()
        
        if batch_update is None:
            return

        for req_index, params, _, _ in batch_update.added:
            # Debug: Check if extra_args survived the trip
            args_str = str(params.extra_args) if params.extra_args else "None"
            
            # Update the req_id map
            if params.extra_args and "req_id" in params.extra_args:
                req_id = params.extra_args["req_id"]
                self.active_req_ids[req_index] = sampling_params(
                    req_id, 
                    params.extra_args.get("normalization_constants", False), 
                    params.temperature, 
                    params.extra_args.get("entropy", False), 
                    params.extra_args.get("entropy_inference", False), 
                    params.extra_args.get("gradient_steps", 0), 
                    params.extra_args.get("learning_rate", 0.0), 
                    params.extra_args.get("delta", 0.0)
                )
            else:
                print(f"WARNING: No req_id found in extra_args for req_index {req_index}. extra_args: {args_str}. Logits logging will be skipped for this request.")

        # Handle removals to keep map clean
        for req_index in batch_update.removed:
            if req_index in self.active_req_ids:
                self.active_req_ids.pop(req_index)

        # Handle index movements 
        for from_idx, to_idx, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self.active_req_ids[to_idx], self.active_req_ids[from_idx] = (
                    self.active_req_ids[from_idx], self.active_req_ids[to_idx]
                )
            else:
                if from_idx in self.active_req_ids:
                    self.active_req_ids[to_idx] = self.active_req_ids[from_idx]
                    del self.active_req_ids[from_idx]
 
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Process logits to calculate and log normalization constants and entropy.

        This method is called by vLLM for each token generation step. It calculates
        normalization constants (logsumexp) and entropy values, then stores them in
        Redis for later retrieval.

        Args:
            logits: Raw logits tensor of shape (batch_size, vocab_size).

        Returns:
            The unmodified logits tensor (this processor only observes, doesn't modify).
        """
        self._ensure_redis()
        
        if not self.active_req_ids:
            print("WARNING: active_req_ids is empty in apply()!")
            return logits
        
        # Store the max_logits and shift_logits of each request
        log_norm_constant = torch.zeros(len(self.active_req_ids), device=logits.device)
        log_norm_constant_temp_scaled = torch.zeros(len(self.active_req_ids), device=logits.device)
        entropy = torch.zeros(len(self.active_req_ids), device=logits.device)
        
        # Calculate the Normalization Constants if normalization_constants = True or entropy = True
        for row_idx, params in self.active_req_ids.items():
            if params.normalization_constants:
                # Calculate the Normalization Constants if required
                log_norm_constant[row_idx] = torch.logsumexp(logits[row_idx], dim=-1)
                log_norm_constant_temp_scaled[row_idx] = torch.logsumexp(logits[row_idx] / params.temperature, dim=-1)
            # If entropy = True, calculate the entropy
            if params.entropy:
                entropy[row_idx] = Categorical(logits=logits[row_idx]).entropy()
    

        # Prepare pipeline for batch Redis operations
        pipe = self.redis_client.pipeline()
        
        found_any = False
        for row_idx, params in self.active_req_ids.items():
            req_id = params.req_id
            if row_idx < logits.size(0):
                # Store as JSON-like string with all normalization info
                data = f"{log_norm_constant[row_idx]},{log_norm_constant_temp_scaled[row_idx]},{entropy[row_idx]}"
                pipe.rpush(req_id, data)
                found_any = True
            else:
                print(f"row_idx {row_idx} >= batch size {logits.size(0)}, skipping")
        
        # Push values to REDIS if we found any valid ones
        if found_any:
            pipe.execute()
 
        return logits
