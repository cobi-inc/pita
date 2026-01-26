"""
TensorRT-LLM Logits Processor for calculating normalization constants and entropy.

This module provides a LogitsProcessor class that can be passed per-request to
TensorRT-LLM's generate() function. Results are stored in Valkey for retrieval
after generation completes, enabling IPC across MPI process boundaries.
"""

import logging
import torch
import valkey
from torch.distributions import Categorical
from typing import List, Optional

from pita.utils.constants import VALKEY_HOST, VALKEY_PORT

# Module-level logger
logger = logging.getLogger(__name__)


class TensorRTLogitsProcessor:
    """
    Logits processor for TensorRT-LLM that calculates and stores normalization constants
    and entropy for each generated token via Valkey IPC.
    
    This processor is designed to be instantiated fresh for each sample() call.
    Results are written to Valkey using the req_id as the key, allowing the main
    process to retrieve them after generation completes.
    
    Attributes:
        req_id (str): Unique request ID used as Valkey key.
        temperature (float): Temperature for scaling logits.
        calculate_entropy (bool): Whether to calculate entropy.
        calculate_normalization (bool): Whether to calculate normalization constants.
        valkey_client: Valkey client for storing computed values.
    """
    
    def __init__(
        self, 
        req_id: str,
        temperature: float = 1.0, 
        calculate_normalization: bool = True,
        calculate_entropy: bool = False
    ):
        """
        Initialize the TensorRTLogitsProcessor.
        
        Args:
            req_id: Unique request ID used as Redis key for storing results.
            temperature: Temperature for scaling logits. Defaults to 1.0.
            calculate_normalization: Whether to calculate normalization constants. Defaults to True.
            calculate_entropy: Whether to calculate entropy. Defaults to False.
        """
        self.req_id: str = req_id
        self.temperature: float = temperature
        self.calculate_normalization: bool = calculate_normalization
        self.calculate_entropy: bool = calculate_entropy
        self.valkey_client = None
        
    def _ensure_valkey(self) -> None:
        """
        Ensure Valkey client is initialized and connected.
        
        Lazily initializes the Valkey connection on first use to avoid connection
        issues during processor instantiation (which may happen in a different process).
        """
        if self.valkey_client is None:
            try:
                self.valkey_client = valkey.Valkey(
                    host=VALKEY_HOST, port=VALKEY_PORT, db=0, decode_responses=True
                )
            except Exception as e:
                logger.error(f"Valkey connection failed: {e}")
                raise ConnectionError(f"Failed to connect to Valkey at {VALKEY_HOST}:{VALKEY_PORT}") from e

    def __call__(
        self, 
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int] = None
    ) -> None:
        """
        Process logits for a single token generation step.
        
        Calculates logsumexp of the current logits for normalization constants
        and optionally calculates Shannon entropy. Results are pushed to Redis.
        
        Args:
            req_id: The ID of the request (from TensorRT-LLM, not used - we use self.req_id).
            logits: A torch.Tensor containing the raw logits.
            token_ids: A list of lists of token IDs generated so far.
            stream_ptr: A pointer to the CUDA stream (required for synchronization).
            client_id: An optional client ID.
            
        Returns:
            None (this processor only observes, doesn't modify logits).
        """
        self._ensure_valkey()
        
        # Synchronize with the CUDA stream before reading logits values
        if stream_ptr is not None:
            stream = torch.cuda.ExternalStream(stream_ptr)
            stream.synchronize()
        else:
            torch.cuda.synchronize()
        
        # Handle 1D, 2D, and 3D logits tensors
        # TensorRT-LLM returns 3D tensors: [batch_size, seq_len, vocab_size]
        if logits.dim() == 1:
            logits_1d = logits
        elif logits.dim() == 2:
            logits_1d = logits[0]
        else:
            batch_size, seq_len = logits.size(0), logits.size(1)
            if batch_size != 1 or seq_len != 1:
                raise ValueError(
                    f"TensorRTLogitsProcessor expects 3D logits with batch_size == 1 and seq_len == 1, "
                    f"but got shape {tuple(logits.shape)}. Multi-batch or multi-step processing is not supported "
                    f"by this processor; ensure generation is configured for a single sequence and step per call."
                )
            logits_1d = logits[0, 0]
        
        # Calculate normalization constants
        if self.calculate_normalization:
            log_norm_constant = torch.logsumexp(logits_1d, dim=-1).item()
            log_norm_constant_temp_scaled = torch.logsumexp(logits_1d / self.temperature, dim=-1).item()
        else:
            log_norm_constant = 0.0
            log_norm_constant_temp_scaled = 0.0
        
        # Calculate entropy
        if self.calculate_entropy:
            token_entropy = Categorical(logits=logits_1d).entropy().item()
        else:
            token_entropy = 0.0
        
        # Push to Valkey: format "norm,norm_temp,entropy"
        if self.valkey_client is None:
            raise RuntimeError("Valkey client is not initialized; cannot push logits data.")
        data = f"{log_norm_constant},{log_norm_constant_temp_scaled},{token_entropy}"
        self.valkey_client.rpush(self.req_id, data)


def create_logits_processor(
    req_id: str,
    temperature: float = 1.0,
    calculate_normalization: bool = True,
    calculate_entropy: bool = False
) -> TensorRTLogitsProcessor:
    """
    Create a TensorRTLogitsProcessor for use with TensorRT-LLM.
    
    Args:
        req_id: Unique request ID used as Valkey key for storing results.
        temperature: Temperature for scaling logits.
        calculate_normalization: Whether to calculate normalization constants.
        calculate_entropy: Whether to calculate entropy.
        
    Returns:
        A TensorRTLogitsProcessor instance that can be passed to generate().
    """
    return TensorRTLogitsProcessor(
        req_id=req_id,
        temperature=temperature,
        calculate_normalization=calculate_normalization,
        calculate_entropy=calculate_entropy
    )