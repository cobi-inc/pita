from src.api.test_time_coding import encode, decode
from src.inference.autoregressive_sampler_backend import Power_Sampling_Params, SMC_Sampling_Params, Best_of_Sampling_Params

def main():
    # Create parameter objects
    power_params = Power_Sampling_Params(total_output_tokens=1000, block_size=250, MCMC_steps=3)
    smc_params = SMC_Sampling_Params(particles=50, particle_length=200, resample_interval=50)
    best_of_params = Best_of_Sampling_Params(best_of=5, n=1)

    # Encode parameters into a string
    encoded_string = encode(Power_Sampling_Params=power_params, SMC_Sampling_Params=smc_params, Best_of_Sampling_Params=best_of_params)
    print(f"Encoded String: {encoded_string}")

    # Decode the string back into parameter objects
    decoded_power_params, decoded_smc_params, decoded_best_of_params = decode(encoded_string)

    # Verify the decoding
    print(f"Decoded Power Sampling Params: total_output_tokens={decoded_power_params.total_output_tokens}, block_size={decoded_power_params.block_size}, MCMC_steps={decoded_power_params.MCMC_steps}")
    print(f"Decoded SMC Sampling Params: particles={decoded_smc_params.particles}, particle_length={decoded_smc_params.particle_length}, resample_interval={decoded_smc_params.resample_interval}")
    print(f"Decoded Best Of Sampling Params: best_of={decoded_best_of_params.best_of}, n={decoded_best_of_params.n}")
    
    # Verify the decoding by comparing attributes
    assert decoded_power_params.total_output_tokens == power_params.total_output_tokens, "Power Sampling total_output_tokens do not match."
    assert decoded_power_params.block_size == power_params.block_size, "Power Sampling block_size do not match."
    assert decoded_power_params.MCMC_steps == power_params.MCMC_steps, "Power Sampling MCMC_steps do not match."
    
    assert decoded_smc_params.particles == smc_params.particles, "SMC Sampling particles do not match."
    assert decoded_smc_params.particle_length == smc_params.particle_length, "SMC Sampling particle_length do not match."
    assert decoded_smc_params.resample_interval == smc_params.resample_interval, "SMC Sampling resample_interval do not match."
    
    assert decoded_best_of_params.best_of == best_of_params.best_of, "Best Of Sampling best_of do not match."
    assert decoded_best_of_params.n == best_of_params.n, "Best Of Sampling n do not match."

    print("All parameters decoded correctly.")

if __name__ == "__main__":
    main()