import numpy as np
from scipy.special import logsumexp
from llama_cpp import Llama, LogitsProcessorList

class LogSumExpProcessor:
    def __init__(self):
        # We store the results in a list to track them for every generated token
        self.log_norm_constants: float = 0.0
        self.log_norm_constants_temp_scaled: float = 0.0
        self.temperature: float = 1.0
        
    def __call__(self, input_ids, scores):
        """
        Calculates logsumexp of the current scores (logits).
        input_ids: npt.NDArray[np.intc] - tokens generated so far
        scores: npt.NDArray[np.single] - the current logits
        """
        # Manual logsumexp for precision/efficiency with numpy
        log_norm_constant = logsumexp(scores)
        log_norm_constant_temp_scaled = logsumexp(scores / self.temperature)
        # Store the result for the current step
        
        # Return scores unchanged so generation continues normally
        return scores
