#Math Libraries
import numpy as np
from scipy.special import logsumexp

# TODO Update the Best_of_N Function to look similar to the other sampling functions
class Best_of_Params:
    def __init__(
        self, 
        sequence_n, # Number of sequences to sample and choose the best from
        sequence_top_k, # Number of top_k sequences to choose from (top_k <= sequences). If top_k = 1, greedy selection is used.
    ):  
        self.sequence_n = sequence_n
        self.sequence_top_k = sequence_top_k

# Default logprob evaluation function
# Context: The prompt in string format that will be used for generation (Should be in Chat Template format if using chat models)
# Sampler: An AutoregressiveSampler object with best_of sampling enabled
# Returns: The best sequence in string format based on logprob evaluation
def best_of_n_logprob(context, sampler):
    # Sample sequence_n sequences from the LLM
    token_sequences, chosen_logit_token, top_k_logits = sampler.sample(
        [context] * sampler.best_of_sampling_params.sequence_n, 
        sampler.sampling_params.max_tokens
    )

    # Calculate the cumulative log probability for each sequence
    sequence_cumulative_logprobs = np.zeros(sampler.best_of_sampling_params.sequence_n)
    for seq_index in range(len(top_k_logits)):
        for token_index in range(len(top_k_logits[seq_index])):
            # Calculate the log probability of the chosen logit logit log(e^logit_i / sum(e^logit_j)) = logit_i - logsumexp(logits)
            token_logprob = chosen_logit_token[seq_index][token_index] - logsumexp(top_k_logits[seq_index][token_index])
            sequence_cumulative_logprobs[seq_index] += token_logprob
        
        # Normalize by sequence length to prevent bias towards longer sequences
        sequence_cumulative_logprobs[seq_index] /= len(top_k_logits[seq_index])

    # Select the top_k sequences based on cumulative logprobs
    sequence_top_k = sampler.best_of_sampling_params.sequence_top_k
    sequence_n = len(sequence_cumulative_logprobs)
    if sequence_top_k == sequence_n:
        # If top_k equals the number of sequences, sort all
        top_k_sequence_cum_logprobs_indices = np.argsort(-sequence_cumulative_logprobs)
    else:
        top_k_sequence_cum_logprobs_indices = np.argpartition(-sequence_cumulative_logprobs, sequence_top_k)[:sequence_top_k]

    # Add the top_k sequence_cumulative_logprobs together and choose them based off their probabilities
    # Find the log probabilities of the top_k sequences
    top_k_cum_logprobs = sequence_cumulative_logprobs[top_k_sequence_cum_logprobs_indices[:sampler.best_of_sampling_params.sequence_top_k]]
    # Convert to relative probabilities using log-sum-exp trick for numerical stability
    top_k_cum_relative_probs = np.exp(top_k_cum_logprobs - logsumexp(top_k_cum_logprobs))

    # Take a weighted random choice from the top_k sequences based on their relative probabilities
    best_index = np.random.choice(top_k_sequence_cum_logprobs_indices[:sampler.best_of_sampling_params.sequence_top_k], p=top_k_cum_relative_probs)
    
    return sampler.tokenizer.decode(token_sequences[best_index], skip_special_tokens=False)


def enable_best_of_sampling(
    sampler, # The sampling parameters object to modify
    sequence_n, # Number of sequences to sample and choose the best from
    sequence_top_k # Number of top_k sequences to choose from (top_k <= sequences)
):      
    # Check if sampler is valid
    if(sampler is None):
        raise ValueError("Sampler object is None. Please provide a valid sampler object.")

    # Check if best_of sampling is already enabled
    if(sampler.best_of_sampling_params is not None):
        raise ValueError("Best_of sampling is already enabled in the sampler.")

    # Check if you are trying to choose from more sequences than generated
    if(sequence_top_k > sequence_n):
        raise ValueError("sequence_top_k must be less than or equal to sequences in best_of sampling.")

    # Initalize the best_of sampling parameters
    best_of_sampler = Best_of_Params(
        sequence_n = sequence_n,
        sequence_top_k = sequence_top_k
    )

    # Store the best_of sampling parameters in the sampler object
    sampler.best_of_sampling_params = best_of_sampler
    print(f"Best Of Sampling Enabled: Sequences = {sequence_n}, Top K = {sequence_top_k}")
