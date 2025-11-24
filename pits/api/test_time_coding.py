from pits.inference.autoregressive_sampler_backend import Power_Sampling_Params, SMC_Sampling_Params, Best_of_Sampling_Params

# Encode the parameters into a single string that can be passed as context (e.g., system prompt) during the API call
def encode(Power_Sampling_Params = None, SMC_Sampling_Params = None, Best_of_Sampling_Params = None):
    system_string =""

    if(Power_Sampling_Params is not None):
        system_string += f"ITS_PS_{Power_Sampling_Params.total_output_tokens}_{Power_Sampling_Params.block_size}_{Power_Sampling_Params.MCMC_steps}"
    
    if(SMC_Sampling_Params is not None):
        if(system_string == ""):
            system_string += "ITS"
        system_string += f"_SMC_{SMC_Sampling_Params.particles}_{SMC_Sampling_Params.particle_length}_{SMC_Sampling_Params.resample_interval}"
    
    if(Best_of_Sampling_Params is not None):
        if(system_string == ""):
            system_string += "ITS"
        system_string += f"_BO_{Best_of_Sampling_Params.best_of}_{Best_of_Sampling_Params.n}"
    
    return system_string

# Decode the parameters from the system prompt string to be used during inference
def decode(system_string):
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
            if not (parts[i+1].isdigit() and parts[i+2].isdigit() and parts[i+3].isdigit()):
                raise ValueError(f"Invalid parameters for Power Sampling: expected 3 integers after 'PS', got '{parts[i+1]}', '{parts[i+2]}', '{parts[i+3]}'.")
            
            # If valide, create the Power_Sampling_Params object
            ps_params = Power_Sampling_Params(
                total_output_tokens=int(parts[i+1]),
                block_size=int(parts[i+2]),
                MCMC_steps=int(parts[i+3])
            )
            i += 4
        elif parts[i] == "SMC":
            # Check if the next 3 parts are valid digits (non-negative integers)
            if not (parts[i+1].isdigit() and parts[i+2].isdigit() and parts[i+3].isdigit()):
                raise ValueError(f"Invalid parameters for SMC: expected 3 integers after 'SMC', got '{parts[i+1]}', '{parts[i+2]}', '{parts[i+3]}'.")
            
            # If valide, create the SMC_Sampling_Params object
            smc_params = SMC_Sampling_Params(
                particles=int(parts[i+1]),
                particle_length=int(parts[i+2]),
                resample_interval=int(parts[i+3])
            )
            i += 4
        elif parts[i] == "BO":
            # Check if the next 2 parts are valid digits (non-negative integers)
            if not (parts[i+1].isdigit() and parts[i+2].isdigit()):
                raise ValueError(f"Invalid parameters for Best Of Sampling: expected 2 integers after 'BO', got '{parts[i+1]}', '{parts[i+2]}'.")

            # If valid, create the Best_of_Sampling_Params object
            best_of_params = Best_of_Sampling_Params(
                best_of=int(parts[i+1]),
                n=int(parts[i+2])
            )
            i += 3
        else:
            i += 1
    
    return ps_params, smc_params, best_of_params