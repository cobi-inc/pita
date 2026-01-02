# Math Libraries
import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

# Custom Libraries
from pita.inference.LLM_backend import AutoregressiveSampler, Output
from pita.sampling.token_metrics import calc_sequence_length_normalized_logprob


class Best_of_N:
    """
    Best of N Sampling Class that samples N sequences and selects the best one based on a token metric.
    
    Attributes:
        sequence_n (int): Number of sequences to sample and choose the best from.
        sequence_top_k (int): Number of top_k sequences to choose from (top_k <= sequence_n). If top_k = 1, greedy selection is used.
        token_metric (str): The token metric to use for evaluation. Can be "logprobs", "power_distribution", "entropy", or "likelihood_confidence".
    """
    def __init__(
        self, 
        sequence_n: int = 10,  # Number of sequences to sample and choose the best from
        sequence_top_k: int = 1,  # Number of top_k sequences to choose from (top_k <= sequences). If top_k = 1, greedy selection is used.
        token_metric: str = "logprobs"  # The token metric to use for evaluation
    ):  
        self.sequence_n = sequence_n
        self.sequence_top_k = sequence_top_k
        self.token_metric = token_metric

    def select_sequence(
        self,
        sequence_scores: npt.NDArray[np.float64]
    ) -> int:
        """
        Select the best sequence based on the sequence scores.

        Args:
            sequence_scores (npt.NDArray[np.float64]): The array of sequence scores.

        Returns:
            int: The index of the best sequence.
        """
        # Select the top_k sequences based on scores
        if self.sequence_top_k == self.sequence_n:
            # If top_k equals the number of sequences, sort all
            top_k_indices = np.argsort(-sequence_scores)[:self.sequence_top_k]
        else:
            top_k_indices = np.argpartition(-sequence_scores, self.sequence_top_k)[:self.sequence_top_k]

        # Find the scores of the top_k sequences
        top_k_scores = sequence_scores[top_k_indices]
        
        # Convert to relative probabilities using log-sum-exp trick for numerical stability
        top_k_relative_probs = np.exp(top_k_scores - logsumexp(top_k_scores))

        # Take a weighted random choice from the top_k sequences based on their relative probabilities
        best_index = np.random.choice(top_k_indices, p=top_k_relative_probs)
        
        # Return the index of the best sequence
        return best_index

    def sample(
        self,
        sampler: AutoregressiveSampler,
        prompt: str
    ) -> Output:
        """
        Samples N sequences and selects the best one based on cumulative token metric.

        Args:
            sampler (AutoregressiveSampler): The sampler object containing sampling parameters and the LLM engine.
            prompt (str): The prompt to sample from.
            
        Returns:
            Output: Standard output object for the PITA library containing the best sequence.
        """
        # Sample sequence_n sequences from the LLM
        outputs = []
        for _ in range(self.sequence_n):
            output = sampler.sample(prompt)
            outputs.append(output)

        # Calculate the cumulative token metric score for each sequence
        sequence_scores = np.zeros(self.sequence_n)
        for seq_index in range(self.sequence_n):
            # Calculate the token metric for the sequence
            sequence_scores[seq_index] = calc_sequence_length_normalized_logprob(outputs[seq_index], sampler, 0, len(outputs[seq_index].tokens), self.token_metric)

        # Select the best sequence
        best_index = self.select_sequence(sequence_scores)

        # Return the best sequence
        return outputs[best_index]
