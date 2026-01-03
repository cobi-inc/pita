"""
TensorRT-LLM Logits Processor for calculating normalization constants and entropy.

This module provides a LogitsProcessor class that can be passed per-request to
TensorRT-LLM's generate() function, following the same pattern as llama_cpp_backend.
Results are stored in instance variables for retrieval after generation completes.
"""

import torch
from torch.distributions import Categorical
from typing import List, Optional
import numpy as np


class TensorRTLogitsProcessor:
    """
    Logits processor for TensorRT-LLM that calculates and stores normalization constants
    and entropy for each generated token.
    
    This processor is designed to be instantiated fresh for each sample() call,
    accumulating per-token statistics that can be retrieved after generation completes.
    
    Unlike the vLLM logits processor which uses Redis for IPC, this processor
    stores results directly in instance variables since TensorRT-LLM supports
    per-request logits processors.
    
    Attributes:
        log_norm_constants (list[float]): Log normalization constants (logsumexp of raw logits) per token.
        log_norm_constants_temp_scaled (list[float]): Log normalization constants after temperature scaling per token.
        entropy (list[float]): Shannon entropy per token.
        temperature (float): Temperature for scaling logits.
        calculate_entropy (bool): Whether to calculate entropy.
        calculate_normalization (bool): Whether to calculate normalization constants.
    """
    
    def __init__(
        self, 
        temperature: float = 1.0, 
        calculate_normalization: bool = True,
        calculate_entropy: bool = False
    ):
        """
        Initialize the TensorRTLogitsProcessor.
        
        Args:
            temperature: Temperature for scaling logits. Defaults to 1.0.
            calculate_normalization: Whether to calculate normalization constants. Defaults to True.
            calculate_entropy: Whether to calculate entropy. Defaults to False.
        """
        self.log_norm_constants: list[float] = []
        self.log_norm_constants_temp_scaled: list[float] = []
        self.entropy: list[float] = []
        self.temperature: float = temperature
        self.calculate_normalization: bool = calculate_normalization
        self.calculate_entropy: bool = calculate_entropy
        
    def reset(self) -> None:
        """Reset all accumulated lists for a new generation."""
        self.log_norm_constants = []
        self.log_norm_constants_temp_scaled = []
        self.entropy = []
        
    def __call__(
        self, 
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: int,
        client_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Process logits for a single token generation step.
        
        Calculates logsumexp of the current logits for normalization constants
        and optionally calculates Shannon entropy.
        
        Args:
            req_id: The ID of the request.
            logits: A torch.Tensor containing the raw logits (batch_size, vocab_size).
            token_ids: A list of lists of token IDs generated so far.
            stream_ptr: A pointer to the CUDA stream.
            client_id: An optional client ID.
            
        Returns:
            The unmodified logits tensor (this processor only observes, doesn't modify).
        """
        # Process each request in the batch
        # For simplicity, we assume batch_size=1 for now (single request processing)
        # If batch processing is needed, this can be extended
        
        if self.calculate_normalization:
            # Calculate log normalization constant (logsumexp of raw logits)
            # Handle both 1D and 2D logits tensors
            if logits.dim() == 1:
                log_norm_constant = torch.logsumexp(logits, dim=-1).item()
                log_norm_constant_temp_scaled = torch.logsumexp(logits / self.temperature, dim=-1).item()
            else:
                # For 2D (batch), take first row
                log_norm_constant = torch.logsumexp(logits[0], dim=-1).item()
                log_norm_constant_temp_scaled = torch.logsumexp(logits[0] / self.temperature, dim=-1).item()
            
            self.log_norm_constants.append(float(log_norm_constant))
            self.log_norm_constants_temp_scaled.append(float(log_norm_constant_temp_scaled))
        else:
            # Append zeros if not calculating
            self.log_norm_constants.append(0.0)
            self.log_norm_constants_temp_scaled.append(0.0)
        
        if self.calculate_entropy:
            # Calculate Shannon entropy from the probability distribution
            if logits.dim() == 1:
                token_entropy = Categorical(logits=logits).entropy().item()
            else:
                token_entropy = Categorical(logits=logits[0]).entropy().item()
            self.entropy.append(float(token_entropy))
        else:
            self.entropy.append(0.0)
        
        # Return logits unchanged so generation continues normally
        return logits


def create_logits_processor(
    temperature: float = 1.0,
    calculate_normalization: bool = True,
    calculate_entropy: bool = False
) -> TensorRTLogitsProcessor:
    """
    Create a TensorRTLogitsProcessor for use with TensorRT-LLM.
    
    Args:
        temperature: Temperature for scaling logits.
        calculate_normalization: Whether to calculate normalization constants.
        calculate_entropy: Whether to calculate entropy.
        
    Returns:
        A TensorRTLogitsProcessor instance that can be passed to generate().
    """
    return TensorRTLogitsProcessor(
        temperature=temperature,
        calculate_normalization=calculate_normalization,
        calculate_entropy=calculate_entropy
    )