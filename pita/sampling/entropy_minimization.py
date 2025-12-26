# TODO Implement entropy minimization Inference

class Entropy_Minimization:
    """
    Entropy minimization class that stores the parameters used for entropy-based sampling/inference.

    Attributes:
        block_size (int): How many blocks to divide the total output tokens into. Smaller block sizes = better quality but slower.
        MCMC_steps (int): Number of MCMC steps to perform per block. More steps = better quality but slower.
        token_metric (str): Metric to use for token selection. Can be "logprobs", "power_distribution", or "entropy".
    """
    def __init__(
        self, 
        block_size: int = 192, # How many blocks to divide the total output tokens into. Smaller block sizes = better quality but slower
        MCMC_steps: int = 8, # Number of MCMC steps to perform per block. More steps = better quality but slower
        token_metric: str = "power_distribution"
    ):
