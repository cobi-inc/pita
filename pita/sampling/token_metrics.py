# Math Libraries
import numpy as np
import numpy.typing as npt

# process the top_k_probs to find just the probs of the chosen tokens
# can be either the logits or logprobs outputted by the LLM inference engine
# top_k_probs is a np.array of the top_k_probs returned from a LLM inference engine
# tokens_list is a np.array of the chosen tokens
def process_top_k_probs(
    top_k_probs: npt.NDArray[np.float64],
    tokens_list: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    # Initialize chosen_token_prob_list with zeros (shape: num_tokens)
    chosen_token_prob_list = np.zeros(len(tokens_list))
    # Iterate through the tokens_list and find the probs of the chosen tokens
    for i in range(len(tokens_list)):
        if(len(top_k_probs[i]) > 1):
            chosen_token_prob_list[i] = top_k_probs[i][0] - np.max(top_k_probs[i][:2])
    # Return the chosen token logits
    return chosen_token_prob_list

# Takes in an array of raw logits of a chosen token sequence, an array of the unprocessed normalization constants, and the power sampling temperature
# Returns the power sampling log probabilities of the chosen token sequence
def power_sampling_logprobs(
    token_logit_list: npt.NDArray[np.float64],
    unprocessed_normalization_constant: float,
    power_sampling_temperature: float
) -> npt.NDArray[np.float64]:
    # P(x) ^ 1/T = P(x) ^ alpha = 1/T * softmax(logit_selected)
    return (1/power_sampling_temperature) * (token_logit_list - unprocessed_normalization_constant)

# Takes in an array of raw logits of a chosen token sequence, an array of the low-temperature processed normalization constants, and the low temperature sampling temperature
# Returns the low temperature log probabilities of the chosen token sequence
def low_temp_logprobs(
    token_logit_list: npt.NDArray[np.float64],
    temp_processed_normalization_constant: float,
    low_temp_sampling_temperature: float
) -> npt.NDArray[np.float64]:
    # P(x/T) = softmax(logit_selected / T)
    return (1/low_temp_sampling_temperature) * token_logit_list - temp_processed_normalization_constant
