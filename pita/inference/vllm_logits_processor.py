import torch
import redis
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor, 
    BatchUpdate, 
    MoveDirectionality
)
from vllm.config import VllmConfig
from typing import Dict, Optional, List

from pita.utils.constants import REDIS_HOST, REDIS_PORT

class LogitsLoggingProcessor(LogitsProcessor):
    def __init__(
         self, 
        vllm_config: VllmConfig, 
        device: torch.device, 
        is_pin_memory: bool
    ) -> None:
        self.active_req_ids: Dict[int, str] = {}
        self.redis_client = None
        self.temperature = 1.0  # Default temperature, can be configured per request

    def _ensure_redis(self):
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
                )
            except Exception as e:
                # If we can't log to Redis, we are flying blind, but try printing just in case
                print(f"CRITICAL WORKER ERROR: Redis connect failed: {e}")

    def is_argmax_invariant(self) -> bool:
        return False  # Must be False to ensure apply() is called

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        self._ensure_redis()
        
        if batch_update is None:
            return

        for req_index, params, _, _ in batch_update.added:
            # Debug: Check if extra_args survived the trip
            args_str = str(params.extra_args) if params.extra_args else "None"

            if params.extra_args and "req_id" in params.extra_args:
                req_id = params.extra_args["req_id"]
                self.active_req_ids[req_index] = req_id
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
        self._ensure_redis()
        
        if not self.active_req_ids:
            print("WARNING: active_req_ids is empty in apply()!")
            return logits
        
        # Get the max logit (this is what vLLM uses to shift the logits)
        max_logit = logits.max(dim=-1).values
        shifted_logits = logits - max_logit.unsqueeze(-1)

        # Calculate normalization constant (LogSumExp) for the current token's distribution
        # dim=-1 ensures we sum over the vocabulary dimension
        norm_constant = torch.logsumexp(shifted_logits, dim=-1)
        
        # Scale by temperature if needed
        norm_constant_temp_scaled = torch.logsumexp(shifted_logits / self.temperature, dim=-1)

        # Prepare pipeline for batch Redis operations
        pipe = self.redis_client.pipeline()
        
        found_any = False
        for row_idx, req_id in self.active_req_ids.items():
            if row_idx < logits.size(0):
                norm_val = norm_constant[row_idx].item()
                norm_temp_val = norm_constant_temp_scaled[row_idx].item()
                max_val = max_logit[row_idx].item()
                
                # Store as JSON-like string with all normalization info
                data = f"{norm_val},{norm_temp_val},{max_val}"
                pipe.rpush(req_id, data)
                print(f"Pushed norm_constant={norm_val}, norm_temp_scaled={norm_temp_val}, max_logit={max_val} for req_id {req_id}")
                found_any = True
            else:
                print(f"row_idx {row_idx} >= batch size {logits.size(0)}, skipping")
        
        # Push values to REDIS if we found any valid ones
        if found_any:
            pipe.execute()
 
        return logits
