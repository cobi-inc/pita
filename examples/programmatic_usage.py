from pita.inference.LLM_backend import AutoregressiveSampler
def main():
    # Configuration
    # You can change these values to match your environment
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
    ENGINE = "vllm" # Options: "vllm", "llama_cpp", "tensorrt"
    DTYPE = "auto"
    GPU_MEMORY_UTILIZATION = 0.85
    CONTEXT_LENGTH = 1024
    MAX_PROBS = 100
    
    print(f"Initializing Sampler with model: {MODEL_NAME} using engine: {ENGINE}...")

    # Initialize the Sampler
    sampler = AutoregressiveSampler(
        engine=ENGINE,
        model=MODEL_NAME,
        dtype=DTYPE,
        tokenizer_path=None,  # Use model's tokenizer
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=CONTEXT_LENGTH,
        max_probs=MAX_PROBS,
        logits_processor=True, # Enable logits processor for metrics
        trust_remote_code=True,
        sampling_params=None # Use default sampling parameters
    )

    # Prepare a prompt
    prompt = "Explain the concept of 'pita' in the context of bread."
    
    # Format prompt using the chat template if available, otherwise raw string
    if hasattr(sampler.tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = sampler.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    print(f"\nPrompting model with:\n{formatted_prompt}\n")
    print("-" * 50)

    # Generate response
    # You can override sampling parameters here if needed
    sampler.sampling_params.max_tokens = 256
    sampler.sampling_params.temperature = 0.7
    sampler.sampling_params.enable_entropy = True # Enable entropy calculation

    output = sampler.sample(formatted_prompt)
    
    # Decode the output
    generated_text = sampler.tokenizer.decode(output.tokens, skip_special_tokens=True)
    
    print("\nGenerated Response:")
    print(generated_text)
    print("-" * 50)

    # Token Metrics Demonstration
    print("\nToken Metrics (First 5 tokens):")
    if output.entropy is not None:
        print(f"Entropy: {output.entropy[:5]}")
    if output.top_k_logprobs is not None:
        # top_k_logprobs is a list of lists (or 2D array), where each inner list contains logprobs for top_k tokens
        # The first element of each inner list is the chosen token's logprob if we just want that
        print(f"Top-k Logprobs (First choice): {[k[0] for k in output.top_k_logprobs[:5]]}")
    print("-" * 50)

    # Example: Enable Power Sampling (if supported)
    # Note: Power Sampling requires compatible token metrics
    print("\nEnabling Power Sampling...")
    try:
        sampler.enable_power_sampling(
            block_size=5,
            MCMC_steps=10,
            token_metric="entropy" # Example metric
        )
        output_power = sampler.token_sample(formatted_prompt)
        generated_text_power = sampler.tokenizer.decode(output_power.tokens, skip_special_tokens=True)
        print("\nGenerated Response (Power Sampling):")
        print(generated_text_power)
    except Exception as e:
        print(f"Power Sampling skipped: {e}")

if __name__ == "__main__":
    main()
