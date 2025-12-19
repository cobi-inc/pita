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
    req_id: str
    normalization_constants: bool
    temperature: float
    entropy: bool
    entropy_inference: bool
    gradient_steps: int
    learning_rate: float
    delta: float

class LogitsLoggingProcessor(LogitsProcessor):
    def __init__(
         self, 
        vllm_config: VllmConfig, 
        device: torch.device, 
        is_pin_memory: bool
    ) -> None:
        self.active_req_ids: Dict[int, sampling_params] = {}
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
        self._ensure_redis()
        
        if not self.active_req_ids:
            print("WARNING: active_req_ids is empty in apply()!")
            return logits
        
        # Store the max_logits and shift_logits of each request
        log_norm_constant = torch.zeros(len(self.active_req_ids), device=logits.device)
        log_norm_constant_temp_scaled = torch.zeros(len(self.active_req_ids), device=logits.device)
        entropy = torch.zeros(len(self.active_req_ids), device=logits.device)
        
        # Calculate the Normalization Constants if normalization_constants = True or entropy = True
        for(req_id, params) in self.active_req_ids.items():
            if(params.normalization_constants):
                # Calculate the Normalization Constants if requireds
                log_norm_constant[req_id] = torch.logsumexp(logits[req_id], dim=-1)
                log_norm_constant_temp_scaled[req_id] = torch.logsumexp(logits[req_id] / params.temperature, dim=-1)                
            #If entropy = True, calculate the entropy
            if(params.entropy):
                entropy[req_id] = Categorical(logits=logits[req_id]).entropy()
    

        # Prepare pipeline for batch Redis operations
        pipe = self.redis_client.pipeline()
        
        found_any = False
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
