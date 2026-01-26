import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy as scipy_entropy
from llama_cpp import LogitsProcessorList
import numpy.typing as npt


class LogSumExpProcessor:
    """
    Logits processor for llama.cpp that calculates and stores normalization constants
    and entropy for each generated token.
    
    This processor is designed to be instantiated fresh for each sample() call,
    accumulating per-token statistics that can be retrieved after generation completes.
    
    Unlike the vLLM logits processor which uses Redis for IPC, this processor
    stores results directly in instance variables since llama.cpp runs in-process.
    
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
        Initialize the LogSumExpProcessor.
        
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
        input_ids: npt.NDArray[np.intc], 
        scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        """
        Process logits for a single token generation step.
        
        Calculates logsumexp of the current scores (logits) for normalization constants
        and optionally calculates Shannon entropy.
        
        Args:
            input_ids: Tokens generated so far (npt.NDArray[np.intc]).
            scores: The current logits (npt.NDArray[np.single]).
            
        Returns:
            The unmodified scores array (this processor only observes, doesn't modify).
        """
        if self.calculate_normalization:
            # Calculate log normalization constant (logsumexp of raw logits)
            log_norm_constant = logsumexp(scores)
            self.log_norm_constants.append(float(log_norm_constant))
            
            # Calculate temperature-scaled log normalization constant
            log_norm_constant_temp_scaled = logsumexp(scores / self.temperature)
            self.log_norm_constants_temp_scaled.append(float(log_norm_constant_temp_scaled))
        else:
            # Append zeros if not calculating
            self.log_norm_constants.append(0.0)
            self.log_norm_constants_temp_scaled.append(0.0)
        
        if self.calculate_entropy:
            # Calculate Shannon entropy from the probability distribution
            # First convert logits to probabilities using softmax
            # For numerical stability, subtract max before exp
            scores_shifted = scores - np.max(scores)
            probs = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted))
            # Calculate entropy (scipy uses natural log by default)
            token_entropy = scipy_entropy(probs)
            self.entropy.append(float(token_entropy))
        else:
            self.entropy.append(0.0)
        
        # Return scores unchanged so generation continues normally
        return scores


def create_logits_processor_list(
    temperature: float = 1.0,
    calculate_normalization: bool = True,
    calculate_entropy: bool = False
) -> tuple[LogitsProcessorList, LogSumExpProcessor]:
    """
    Create a LogitsProcessorList with a LogSumExpProcessor for use with llama.cpp.
    
    Args:
        temperature: Temperature for scaling logits.
        calculate_normalization: Whether to calculate normalization constants.
        calculate_entropy: Whether to calculate entropy.
        
    Returns:
        A tuple of (LogitsProcessorList, LogSumExpProcessor) where the processor
        can be used to retrieve accumulated statistics after generation.
    """
    processor = LogSumExpProcessor(
        temperature=temperature,
        calculate_normalization=calculate_normalization,
        calculate_entropy=calculate_entropy
    )
    processor_list = LogitsProcessorList([processor])
    return processor_list, processor
