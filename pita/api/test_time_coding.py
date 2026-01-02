from pita.sampling.power_sample import Power_Sampling
from pita.sampling.smc import Sequential_Monte_Carlo
from pita.sampling.best_of import Best_of_N
from typing import Optional

# Encode the parameters into a single string that can be passed as context (e.g., system prompt) during the API call
def encode(
    Power_Sampling_Params: Optional[Power_Sampling] = None,
    SMC_Sampling_Params: Optional[Sequential_Monte_Carlo] = None,
    Best_of_Sampling_Params: Optional[Best_of_N] = None
) -> str:
    """
    Encode test-time scaling parameters into a string for embedding in system prompts.

    This function creates a compact string representation of sampling parameters that can be
    embedded in the system prompt during API calls. The string is prefixed with "ITS" (Inference
    Time Scaling) and contains parameter values for various sampling techniques.

    Args:
        Power_Sampling_Params: Power sampling configuration object containing block_size and MCMC_steps.
        SMC_Sampling_Params: Sequential Monte Carlo configuration object containing num_particles,
            tokens_per_step, and stop_on_eos.
        Best_of_Sampling_Params: Best-of-N sampling configuration object containing sequence_n
            and sequence_top_k.

    Returns:
        A formatted string starting with "ITS" followed by underscore-separated parameter values.
        For example: "ITS_PS_192_8" for power sampling with block_size=192 and MCMC_steps=8.
        Returns empty string if no parameters are provided.
    """
    system_string =""

    # Check if the block size and MCMC steps are provided for Power Sampling
    if(Power_Sampling_Params is not None):
        system_string += f"ITS_PS_{Power_Sampling_Params.block_size}_{Power_Sampling_Params.MCMC_steps}"
    
    # Check if the parameters are provided for SMC Sampling
    # stop_oni_eos is a boolean, so we convert it to int for easier encoding/decoding
    if(SMC_Sampling_Params is not None):
        if(system_string == ""):
            system_string += "ITS"
        system_string += f"_SMC_{SMC_Sampling_Params.num_particles}_{SMC_Sampling_Params.tokens_per_step}_{int(SMC_Sampling_Params.stop_on_eos)}"
    
    # Check if the parameters are provided for Best Of Sampling
    if(Best_of_Sampling_Params is not None):
        if(system_string == ""):
            system_string += "ITS"
        system_string += f"_BO_{Best_of_Sampling_Params.sequence_n}_{Best_of_Sampling_Params.sequence_top_k}"
    
    return system_string

# Decode the parameters from the system prompt string to be used during inference
def decode(system_string: str) -> tuple[Optional[Power_Sampling], Optional[Sequential_Monte_Carlo], Optional[Best_of_N]]:
    """
    Decode test-time scaling parameters from a system prompt string.

    This function parses a string encoded by the `encode()` function and reconstructs the
    sampling parameter objects. The string should start with "ITS" followed by parameter
    specifications.

    Args:
        system_string: The encoded string containing test-time scaling parameters.
            Expected format: "ITS_PS_<block_size>_<MCMC_steps> [additional content...]"
            or similar patterns for SMC and Best-of-N sampling.

    Returns:
        A tuple of (ps_params, smc_params, best_of_params) where each element is either
        a parameter object or None if that sampling technique was not specified.

    Raises:
        ValueError: If the system_string doesn't start with "ITS" or if parameter values
            are invalid (non-numeric or wrong count).
    """
    # Initialize the objects to None
    ps_params = None
    smc_params = None
    best_of_params = None

    # Split the string into parts and parse
    parts = system_string.split(" ")
    parts = parts[0].split("_")  # Only consider the first part before any spaces
    if(parts[0] != "ITS"):
        raise ValueError("Invalid system string format. Must start with 'ITS'.")
    
    i = 1  # Start after the initial "ITS"
    while i < len(parts):
        # Check to see if the test time coding parts are present
        if parts[i] == "PS" :
            # Check if the next 3 parts are valid digits (non-negative integers)
            if not (parts[i+1].isdigit() and parts[i+2].isdigit()):
                raise ValueError(f"Invalid parameters for Power Sampling: expected 2 integers after 'PS', got '{parts[i+1]}', '{parts[i+2]}'.")
            
            # If valide, create the Power_Sampling_Params object
            ps_params = Power_Sampling_Params(
                block_size=int(parts[i+1]),
                MCMC_steps=int(parts[i+2])
            )
            i += 3
        elif parts[i] == "SMC":
            # Check if the next 3 parts are valid digits (non-negative integers)
            if not (parts[i+1].isdigit() and parts[i+2].isdigit() and parts[i+3].isdigit()):
                raise ValueError(f"Invalid parameters for SMC: expected 3 integers after 'SMC', got '{parts[i+1]}', '{parts[i+2]}', '{parts[i+3]}'.")
            
            # If valid, create the SMC_Sampling_Params object
            smc_params = SequentialMonteCarlo_Params(
                num_particles=int(parts[i+1]),
                tokens_per_step=int(parts[i+2]),
                stop_on_eos=bool(int(parts[i+3]))
            )
            i += 4
        elif parts[i] == "BO":
            # Check if the next 2 parts are valid digits (non-negative integers)
            if not (parts[i+1].isdigit() and parts[i+2].isdigit()):
                raise ValueError(f"Invalid parameters for Best Of Sampling: expected 2 integers after 'BO', got '{parts[i+1]}', '{parts[i+2]}'.")

            # If valid, create the Best_of_Sampling_Params object
            best_of_params = Best_of_Params(
                sequence_n=int(parts[i+1]),
                sequence_top_k=int(parts[i+2])
            )
            i += 3
        else:
            i += 1
    
    return ps_params, smc_params, best_of_params
