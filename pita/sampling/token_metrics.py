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
        return (1/sampler.sampling_params.temperature) * (output.top_k_logits[:, 0] - output.unprocessed_normalization_constant)
    elif metric == "entropy":
        return output.entropy