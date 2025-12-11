
from vllm import LLM, SamplingParams
from pita.inference.autoregressive_sampler_backend import create_autoregressive_sampler
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

def main():
    _engine_name = "vllm"
    _model_name = "Qwen/Qwen3-4B-AWQ"
    _dtype = "auto"
    _tokenizer_path = None
    _gpu_memory_utilization = 0.25
    _max_model_len = 2048
    _max_logprobs = 2
    _logits_per_token = None

    sampler = create_autoregressive_sampler(
        engine=_engine_name, 
        model=_model_name, 
        dtype=_dtype,
        tokenizer_path=_tokenizer_path, 
        gpu_memory_utilization=_gpu_memory_utilization, 
        max_model_len=_max_model_len, 
        max_logprobs = _max_logprobs,
        logits_per_token = _logits_per_token,
        normalization_constants = False
    )

    tokens, top_k_logits, top_k_logprobs, unprocessed_normalization_constant, temp_processed_normalization_constant = sampler.sample("Hello, world!", max_new_tokens=10)
    
    print("Generated Output:", tokens)
    print("Top-K Logits:", top_k_logits)
    print("Top-K Logprobs:", top_k_logprobs)
    print("Unprocessed Normalization Constants:", unprocessed_normalization_constant)
    print("Temp Processed Normalization Constants:", temp_processed_normalization_constant)

# Run Main
if __name__ == "__main__":
    main()