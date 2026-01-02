# Math Libraries
import numpy as np
import numpy.typing as npt

# PITA Libraries
from pita.inference.LLM_backend import AutoregressiveSampler, Output

def calc_token_metric(
    output: Output,
    sampler: AutoregressiveSampler,
    metric: str
) -> npt.NDArray[np.float64]:
    """
    Calculate the token metric for the given output and sampler.

    Args:
        output (Output): The output object containing the token metrics.
        sampler (AutoregressiveSampler): The sampler object containing the sampling parameters.
        metric (str): The metric to calculate. Can be "logprobs", "power_distribution", or "entropy".

    Returns:
        npt.NDArray[np.float64]: The calculated token metric.
    """
    if metric == "logprobs":
        return output.top_k_logprobs[:, 0]
    elif metric == "power_distribution":
        return (1 / sampler.sampling_params.temperature) * (
            output.top_k_logits[:, 0] - np.asarray(output.unprocessed_log_normalization_constant)
        )
    elif metric == "entropy":
        return output.entropy
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Expected one of 'logprobs', 'power_distribution', or 'entropy'."
        )

def calc_sequence_prob(
    output: Output,
    sampler: AutoregressiveSampler,
    starting_index: int,
    ending_index: int,
    metric: str
) -> float:
    """
    Calculate the probability of a sequence of tokens based on the token metric.

    Args:
        output (Output): The output object containing the token metrics.
        sampler (AutoregressiveSampler): The sampler object containing the sampling parameters.
        starting_index (int): The starting index of the sequence.
        ending_index (int): The ending index of the sequence.
        metric (str): The metric to calculate. Can be "logprobs", "power_distribution", or "entropy".

    Returns:
        float: The calculated sequence probability.
    """
    if metric == "logprobs":
        # Sum the log probabilities of the tokens in the sequence and exponentiate
        return np.exp(np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]))
    elif metric == "power_distribution":
        # Sum the log power distribution of the tokens in the sequence and exponentiate
        return np.exp(np.sum((1 / sampler.sampling_params.temperature) * (
            output.top_k_logits[:, 0][starting_index:ending_index] - np.asarray(output.unprocessed_log_normalization_constant)[starting_index:ending_index]
        )))
    elif metric == "entropy":
        # Exponentiate the average negative entropy of the tokens in the sequence
        return np.exp(-np.mean(output.entropy[starting_index:ending_index]))
    elif metric == "likelihood_confidence":
        # Multiply the probability of the sequence by the confidence of the sequence
        return np.exp(np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index])) * np.exp(-np.mean(output.entropy[starting_index:ending_index]))
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Expected one of 'logprobs', 'power_distribution', 'entropy', or 'likelihood_confidence'."
        )
        
def calc_sequence_logprob(
    output: Output,
    sampler: AutoregressiveSampler,
    starting_index: int,
    ending_index: int,
    metric: str
) -> float:
    """
    Calculate the log probability of a sequence of tokens based on the token metric.

    Args:
        output (Output): The output object containing the token metrics.
        sampler (AutoregressiveSampler): The sampler object containing the sampling parameters.
        starting_index (int): The starting index of the sequence.
        ending_index (int): The ending index of the sequence.
        metric (str): The metric to calculate. Can be "logprobs", "power_distribution", "entropy", or "likelihood_confidence".

    Returns:
        float: The calculated sequence log probability.
    """
    if(metric == "logprobs"):
        # Sum the log probabilities of the tokens in the sequence
        return np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index])
    elif(metric == "power_distribution"):
        # Sum the log power distribution of the tokens in the sequence
        return np.sum((1 / sampler.sampling_params.temperature) * (
            output.top_k_logits[:, 0][starting_index:ending_index] - np.asarray(output.unprocessed_log_normalization_constant)[starting_index:ending_index]
        ))
    elif(metric == "entropy"):
        # Exponentiate the average negative entropy of the tokens in the sequence
        return -np.mean(output.entropy[starting_index:ending_index])
    elif(metric == "likelihood_confidence"):
         # Add the log probability of the sequence and the negative entropy of the sequence log( p(x) * e^(-H(x)) ) = log(p(x)) + log(e^(-H(x))) = log(p(x)) - H(x)
         return np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]) - np.mean(output.entropy[starting_index:ending_index])
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Expected one of 'logprobs', 'power_distribution', 'entropy', or 'likelihood_confidence'."
        )

def calc_sequence_length_normalized_prob(
    output: Output,
    sampler: AutoregressiveSampler,
    starting_index: int,
    ending_index: int,
    metric: str
) -> float:
    """
    Calculate the probability of a sequence of tokens based on the token metric.

    Args:
        output (Output): The output object containing the token metrics.
        sampler (AutoregressiveSampler): The sampler object containing the sampling parameters.
        starting_index (int): The starting index of the sequence.
        ending_index (int): The ending index of the sequence.
        metric (str): The metric to calculate. Can be "logprobs", "power_distribution", or "entropy".

    Returns:
        float: The calculated sequence probability.
    """
    if(metric == "logprobs"):
        # Sum the log probabilities of the tokens in the sequence and exponentiate
        return np.exp(np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]) / (ending_index - starting_index))
    elif(metric == "power_distribution"):
        # Sum the log power distribution of the tokens in the sequence and exponentiate
        return np.exp(np.sum((1 / sampler.sampling_params.temperature) * (
            output.top_k_logits[:, 0][starting_index:ending_index] - np.asarray(output.unprocessed_log_normalization_constant)[starting_index:ending_index]
        )) / (ending_index-starting_index))
    elif(metric == "entropy"):
        # Exponentiate the average negative entropy of the tokens in the sequence
        return np.exp(-np.mean(output.entropy[starting_index:ending_index]))
    elif(metric == "likelihood_confidence"):
         # Multiply the probability of the sequence by the confidence of the sequence
         return np.exp(np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]) / (ending_index-starting_index)) * np.exp(-np.mean(output.entropy[starting_index:ending_index]))
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Expected one of 'logprobs', 'power_distribution', 'entropy', or 'likelihood_confidence'."
        )

def calc_sequence_length_normalized_logprob(
    output: Output,
    sampler: AutoregressiveSampler,
    starting_index: int,
    ending_index: int,
    metric: str
) -> float:
    """
    Calculate the log probability of a sequence of tokens based on the token metric.

    Args:
        output (Output): The output object containing the token metrics.
        sampler (AutoregressiveSampler): The sampler object containing the sampling parameters.
        starting_index (int): The starting index of the sequence.
        ending_index (int): The ending index of the sequence.
        metric (str): The metric to calculate. Can be "logprobs", "power_distribution", or "entropy".

    Returns:
        float: The calculated sequence log probability.
    """
    if(metric == "logprobs"):
        # Sum the log probabilities of the tokens in the sequence and exponentiate
        return np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]) / (ending_index - starting_index)
    elif(metric == "power_distribution"):
        # Sum the log power distribution of the tokens in the sequence and exponentiate
        return np.sum((1 / sampler.sampling_params.temperature) * (
            output.top_k_logits[:, 0][starting_index:ending_index] - np.asarray(output.unprocessed_log_normalization_constant)[starting_index:ending_index]
        )) / (ending_index-starting_index)
    elif(metric == "entropy"):
        # Exponentiate the average negative entropy of the tokens in the sequence
        return -np.mean(output.entropy[starting_index:ending_index])
    elif(metric == "likelihood_confidence"):
        # Multiply the probability of the sequence by the confidence of the sequence
        return np.sum(output.top_k_logprobs[:, 0][starting_index:ending_index]) / (ending_index-starting_index) - np.mean(output.entropy[starting_index:ending_index])
    else:
        raise ValueError(
            f"Invalid metric: {metric}. Expected one of 'logprobs', 'power_distribution', 'entropy', or 'likelihood_confidence'."
        )