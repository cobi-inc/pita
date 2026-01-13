import json
import gc
import numpy as np
from pita.inference.LLM_backend import AutoregressiveSampler

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def instantiate_vllm_instance():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Initializing vLLM instance with model: {model_name}")
    
    # Instantiate AutoregressiveSampler with vLLM
    sampler = AutoregressiveSampler(
        engine="vllm",
        model=model_name,
        dtype="auto",
        tokenizer_path=None,
        gpu_memory_utilization=0.6,
        max_model_len=1024,
        max_probs=5,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )

    # Configure Sampling Params
    print("Configuring sampling parameters...")
    sampler.sampling_params.temperature = 1.0
    sampler.sampling_params.top_p = 1.0
    sampler.sampling_params.top_k = 1
    sampler.sampling_params.enable_normalization_constants = True
    sampler.sampling_params.enable_entropy = True
    
    return sampler

def prompt_and_log(sampler):
    prompt = "Hello, how are you?"
    print(f"Prompting model with: '{prompt}'")
    
    output = sampler.sample(context=prompt)
    
    output_data = {
        "tokens": output.tokens,
        "top_k_logits": output.top_k_logits,
        "top_k_logprobs": output.top_k_logprobs,
        "entropy": output.entropy,
        "unprocessed_log_normalization_constant": output.unprocessed_log_normalization_constant,
        "temp_processed_log_normalization_constant": output.temp_processed_log_normalization_constant
    }

    output_file = "vllm_output.json"
    print(f"Writing output to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, cls=NumpyEncoder)

def cleanup(sampler):
    print("Cleanup...")
    del sampler
    gc.collect()
    print("Done.")

import argparse

def instantiate_llama_cpp_instance():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct-GGUF" # HuggingFace repo ID
    print(f"Initializing LlamaCPP instance with model: {model_name}")
    
    # Instantiate AutoregressiveSampler with llama_cpp
    sampler = AutoregressiveSampler(
        engine="llama_cpp",
        model=model_name,
        dtype="fp16",
        tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct", # Use non-GGUF repo for tokenizer
        gpu_memory_utilization=0.6,
        max_model_len=1024,
        max_probs=5,
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )

    # Configure Sampling Params
    print("Configuring sampling parameters...")
    sampler.sampling_params.temperature = 1.0 # Greedy compatible
    sampler.sampling_params.top_p = 1.0
    sampler.sampling_params.top_k = 1
    sampler.sampling_params.enable_normalization_constants = True
    sampler.sampling_params.enable_entropy = True
    
    return sampler

def prompt_and_log_llama_cpp(sampler):
    prompt = "Hello, how are you?"
    print(f"Prompting model with: '{prompt}'")
    
    output = sampler.sample(context=prompt)
    
    output_data = {
        "tokens": output.tokens,
        "top_k_logits": output.top_k_logits,
        "top_k_logprobs": output.top_k_logprobs,
        "entropy": output.entropy,
        "unprocessed_log_normalization_constant": output.unprocessed_log_normalization_constant,
        "temp_processed_log_normalization_constant": output.temp_processed_log_normalization_constant
    }

    output_file = "llama_cpp_output.json"
    print(f"Writing output to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, cls=NumpyEncoder)

def cleanup_llama_cpp(sampler):
    print("Cleanup...")
    del sampler
    gc.collect()
    print("Done.")

def instantiate_tensorrt_instance():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Initializing TensorRT instance with model: {model_name}")
    
    # Instantiate AutoregressiveSampler with tensorrt
    # User constraint: max_probs = 1
    sampler = AutoregressiveSampler(
        engine="tensorrt",
        model=model_name,
        dtype="auto", 
        tokenizer_path=None,
        gpu_memory_utilization=0.6,
        max_model_len=1024,
        max_probs=1, # User constraint
        logits_processor=True,
        trust_remote_code=True,
        sampling_params=None
    )

    # Configure Sampling Params
    print("Configuring sampling parameters...")
    sampler.sampling_params.temperature = 1.0 
    sampler.sampling_params.top_p = 1.0
    sampler.sampling_params.top_k = 1
    # User requested max_probs=1, ensuring per-token params respect this
    sampler.sampling_params.logprobs_per_token = 1
    sampler.sampling_params.logits_per_token = 1
    
    sampler.sampling_params.enable_normalization_constants = True
    sampler.sampling_params.enable_entropy = True
    
    return sampler

def prompt_and_log_tensorrt(sampler):
    prompt = "Hello, how are you?"
    print(f"Prompting model with: '{prompt}'")
    
    output = sampler.sample(context=prompt)
    
    output_data = {
        "tokens": output.tokens,
        "top_k_logits": output.top_k_logits,
        "top_k_logprobs": output.top_k_logprobs,
        "entropy": output.entropy,
        "unprocessed_log_normalization_constant": output.unprocessed_log_normalization_constant,
        "temp_processed_log_normalization_constant": output.temp_processed_log_normalization_constant
    }

    output_file = "tensorrt_output.json"
    print(f"Writing output to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, cls=NumpyEncoder)

def cleanup_tensorrt(sampler):
    print("Cleanup...")
    del sampler
    gc.collect()
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Run engine check for vLLM, LlamaCPP, or TensorRT")
    parser.add_argument("--engine", type=str, choices=["vllm", "llamacpp", "tensorrt"], required=True, help="Engine to check")
    args = parser.parse_args()

    if args.engine == "vllm":
        sampler = instantiate_vllm_instance()
        prompt_and_log(sampler)
        cleanup(sampler)
    elif args.engine == "llamacpp":
        sampler = instantiate_llama_cpp_instance()
        prompt_and_log_llama_cpp(sampler)
        cleanup_llama_cpp(sampler)
    elif args.engine == "tensorrt":
        sampler = instantiate_tensorrt_instance()
        prompt_and_log_tensorrt(sampler)
        cleanup_tensorrt(sampler)

if __name__ == "__main__":
    main()