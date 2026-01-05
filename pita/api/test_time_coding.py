# Test Time Coding - Encode/Decode for Inference Time Scaling Parameters
# Format: ITS_<chain_sampling>_<chain_params...>_<token_sampling>_<token_params...>
#
# Chain Sampling Methods (operate on full sequences):
#   - SMC_<num_particles>_<tokens_per_step>_<stop_on_eos> (Sequential Monte Carlo)
#   - BO_<sequence_n>_<sequence_top_k> (Best-of-N)
#   - NONE (no chain sampling)
#
# Token Sampling Methods (modify individual token generation):
#   - PS_<block_size>_<MCMC_steps> (Power Sampling)
#   - NONE (no token sampling)
#
# Examples:
#   ITS_SMC_10_5_1_NONE         -> SMC with 10 particles, 5 tokens/step, stop_on_eos=True
#   ITS_BO_5_2_PS_192_8         -> Best-of-N (5 sequences, top 2) with Power Sampling (192 block, 8 MCMC)
#   ITS_NONE_PS_192_8           -> Just Power Sampling
#   ITS_SMC_10_5_1_PS_192_8     -> SMC with Power Sampling for token generation

from pita.sampling.power_sample import Power_Sampling
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.sampling.best_of import Best_of_N
from typing import Optional


def encode(
    chain_sampling: Optional[Sequential_Monte_Carlo | Best_of_N] = None,
    token_sampling: Optional[Power_Sampling] = None
) -> str:
    """
    Encode test-time scaling parameters into a string for embedding in system prompts.

    Args:
        chain_sampling: Chain sampling configuration (SMC or Best-of-N). Only one allowed.
        token_sampling: Token sampling configuration (Power Sampling).

    Returns:
        A formatted string: "ITS_<chain>_<chain_params>_<token>_<token_params>"
        Returns empty string if no parameters are provided.
    """
    if chain_sampling is None and token_sampling is None:
        return ""
    
    parts = ["ITS"]
    
    # Encode chain sampling method
    if chain_sampling is None:
        parts.append("NONE")
    elif isinstance(chain_sampling, Sequential_Monte_Carlo):
        parts.extend(["SMC", str(chain_sampling.num_particles), 
                      str(chain_sampling.tokens_per_step), 
                      str(int(chain_sampling.stop_on_eos))])
    elif isinstance(chain_sampling, Best_of_N):
        parts.extend(["BO", str(chain_sampling.sequence_n), 
                      str(chain_sampling.sequence_top_k)])
    else:
        raise ValueError(f"Unknown chain sampling type: {type(chain_sampling)}")
    
    # Encode token sampling method
    if token_sampling is None:
        parts.append("NONE")
    elif isinstance(token_sampling, Power_Sampling):
        parts.extend(["PS", str(token_sampling.block_size), 
                      str(token_sampling.MCMC_steps)])
    else:
        raise ValueError(f"Unknown token sampling type: {type(token_sampling)}")
    
    return "_".join(parts)


def decode(system_string: str) -> tuple[Optional[Sequential_Monte_Carlo | Best_of_N], Optional[Power_Sampling]]:
    """
    Decode test-time scaling parameters from a system prompt string.

    Args:
        system_string: The encoded string containing test-time scaling parameters.
            Format: "ITS_<chain>_<chain_params...>_<token>_<token_params...>"

    Returns:
        A tuple of (chain_sampling, token_sampling) where each element is either
        a parameter object or None if that sampling technique was not specified.

    Raises:
        ValueError: If the format is invalid or parameter values are non-numeric.
    """
    # Split the string into parts (only first token before space)
    parts = system_string.split(" ")[0].split("_")
    
    if len(parts) < 2 or parts[0] != "ITS":
        raise ValueError("Invalid system string format. Must start with 'ITS'.")
    
    chain_sampling = None
    token_sampling = None
    
    i = 1  # Start after "ITS"
    
    # Parse chain sampling method
    if parts[i] == "NONE":
        i += 1
    elif parts[i] == "SMC":
        # SMC requires 3 params: num_particles, tokens_per_step, stop_on_eos
        if i + 3 >= len(parts):
            raise ValueError("SMC requires 3 parameters: num_particles, tokens_per_step, stop_on_eos")
        if not all(parts[i+j].isdigit() for j in range(1, 4)):
            raise ValueError(f"Invalid SMC parameters: expected 3 integers after 'SMC'")
        chain_sampling = Sequential_Monte_Carlo(
            num_particles=int(parts[i+1]),
            tokens_per_step=int(parts[i+2]),
            stop_on_eos=bool(int(parts[i+3]))
        )
        i += 4
    elif parts[i] == "BO":
        # BO requires 2 params: sequence_n, sequence_top_k
        if i + 2 >= len(parts):
            raise ValueError("BO requires 2 parameters: sequence_n, sequence_top_k")
        if not all(parts[i+j].isdigit() for j in range(1, 3)):
            raise ValueError(f"Invalid BO parameters: expected 2 integers after 'BO'")
        chain_sampling = Best_of_N(
            sequence_n=int(parts[i+1]),
            sequence_top_k=int(parts[i+2])
        )
        i += 3
    else:
        raise ValueError(f"Unknown chain sampling method: '{parts[i]}'. Expected 'SMC', 'BO', or 'NONE'.")
    
    # Parse token sampling method
    if i >= len(parts):
        raise ValueError("Missing token sampling specification. Expected 'PS' or 'NONE'.")
    
    if parts[i] == "NONE":
        pass  # token_sampling stays None
    elif parts[i] == "PS":
        # PS requires 2 params: block_size, MCMC_steps
        if i + 2 >= len(parts):
            raise ValueError("PS requires 2 parameters: block_size, MCMC_steps")
        if not all(parts[i+j].isdigit() for j in range(1, 3)):
            raise ValueError(f"Invalid PS parameters: expected 2 integers after 'PS'")
        token_sampling = Power_Sampling(
            block_size=int(parts[i+1]),
            MCMC_steps=int(parts[i+2])
        )
    else:
        raise ValueError(f"Unknown token sampling method: '{parts[i]}'. Expected 'PS' or 'NONE'.")
    
    return chain_sampling, token_sampling
