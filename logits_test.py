import torch
import redis
import os
import time
from typing import Dict, Optional, List
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor, 
    BatchUpdate, 
    MoveDirectionality
)

# --- Configuration ---
# Ensure this matches your local redis server
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
TEST_REQ_ID = "my_request_001"
DEBUG_KEY = "vllm_worker_debug_log"

class RedisDebugProcessor(LogitsProcessor):
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        device: torch.device, 
        is_pin_memory: bool
    ) -> None:
        self.device = device
        self.active_req_ids: Dict[int, str] = {}
        self.redis_client = None
        self.temperature = 1.0  # Default temperature, can be configured per request

    def _log(self, message: str):
        """Helper to push debug logs to Redis from the worker."""
        try:
            if self.redis_client:
                self.redis_client.rpush(DEBUG_KEY, f"[Worker-PID-{os.getpid()}] {message}")
        except:
            pass

    def _ensure_redis(self):
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
                )
                self._log("Successfully connected to Redis inside Worker!")
            except Exception as e:
                # If we can't log to Redis, we are flying blind, but try printing just in case
                print(f"CRITICAL WORKER ERROR: Redis connect failed: {e}")

    def is_argmax_invariant(self) -> bool:
        return False  # Must be False to ensure apply() is called

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        self._ensure_redis()
        
        if batch_update is None:
            return

        # Debug: Log every time we receive new requests
        if batch_update.added:
            self._log(f"Received {len(batch_update.added)} new requests.")

        for req_index, params, _, _ in batch_update.added:
            # Debug: Check if extra_args survived the trip
            args_str = str(params.extra_args) if params.extra_args else "None"
            self._log(f"New Request at BatchIdx {req_index}. extra_args: {args_str}")

            if params.extra_args and "req_id" in params.extra_args:
                req_id = params.extra_args["req_id"]
                self.active_req_ids[req_index] = req_id
                self._log(f"Mapped BatchIdx {req_index} -> ReqID {req_id}")
            else:
                self._log(f"WARNING: 'req_id' missing in extra_args for BatchIdx {req_index}")

        # Handle removals to keep map clean
        for req_index in batch_update.removed:
            if req_index in self.active_req_ids:
                self.active_req_ids.pop(req_index)

        # Handle moves
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
        
        # Debug: ALWAYS log apply being called
        self._log(f"Apply called. Batch size: {logits.size(0)}. active_req_ids: {self.active_req_ids}")
        
        if not self.active_req_ids:
            self._log("WARNING: active_req_ids is empty in apply()!")
            return logits
        
        # Get the max logit (this is what vLLM uses to shift the logits)
        max_logit = logits.max(dim=-1).values
        shifted_logits = logits - max_logit.unsqueeze(-1)

        # Calculate normalization constant (LogSumExp) for the current token's distribution
        # dim=-1 ensures we sum over the vocabulary dimension
        norm_constant = torch.logsumexp(shifted_logits, dim=-1)
        
        # Scale by temperature if needed
        norm_constant_temp_scaled = torch.logsumexp(shifted_logits / self.temperature, dim=-1)

        pipe = self.redis_client.pipeline()
        
        found_any = False
        for row_idx, req_id in self.active_req_ids.items():
            self._log(f"Checking row_idx={row_idx}, logits.size(0)={logits.size(0)}")
            if row_idx < logits.size(0):
                norm_val = norm_constant[row_idx].item()
                norm_temp_val = norm_constant_temp_scaled[row_idx].item()
                max_val = max_logit[row_idx].item()
                
                # Store as JSON-like string with all normalization info
                data = f"{norm_val},{norm_temp_val},{max_val}"
                pipe.rpush(req_id, data)
                self._log(f"Pushed norm_constant={norm_val}, norm_temp_scaled={norm_temp_val}, max_logit={max_val} for req_id {req_id}")
                found_any = True
            else:
                self._log(f"row_idx {row_idx} >= batch size {logits.size(0)}, skipping")
        
        if found_any:
            pipe.execute()
        else:
            self._log("No values were pushed to Redis!")
            
        return logits

def run_debug_test():
    # 1. Setup Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.delete(TEST_REQ_ID)
        r.delete(DEBUG_KEY)
        print(f"Main Process: Connected to Redis. Cleared keys.")
    except redis.ConnectionError:
        print("Main Process: Could not connect to Redis.")
        return

    # 2. Init vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[RedisDebugProcessor],
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        logprobs_mode='raw_logits'
    )

    # 3. Generate
    prompt = "Hello world"
    sampling_params = SamplingParams(
        temperature=0.5, 
        max_tokens=5,
        logprobs=1,
        extra_args={"req_id": TEST_REQ_ID} 
    )

    print(f"Generating prompt...")
    outputs = llm.generate(prompt, sampling_params)

    # 4. Dump Debug Logs
    print("\n--- WORKER DEBUG LOGS (From Redis) ---")
    debug_logs = r.lrange(DEBUG_KEY, 0, -1)
    if not debug_logs:
        print("NO LOGS RECEIVED. The worker likely failed to connect to Redis or crashed.")
    else:
        for log in debug_logs:
            print(log)
    print("--------------------------------------\n")

    # 5. Check Results
    stored_values = r.lrange(TEST_REQ_ID, 0, -1)
    print(f"Values extracted: {stored_values}")
    
    # Parse and display the normalization constants
    if stored_values:
        print("\nParsed normalization constants:")
        for i, val in enumerate(stored_values):
            parts = val.split(',')
            if len(parts) == 3:
                print(f"  Token {i}: norm_constant={parts[0]}, norm_temp_scaled={parts[1]}, max_logit={parts[2]}")

    if len(stored_values) == 5:
        print("\nSUCCESS")
    else:
        print("\nFAILURE")

if __name__ == "__main__":
    run_debug_test()