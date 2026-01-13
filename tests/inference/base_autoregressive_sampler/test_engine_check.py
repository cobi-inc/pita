import pytest
import json
import os
import gc
import numpy as np
from pita.inference.LLM_backend import AutoregressiveSampler

# Constants
MODEL_NAME_VLLM = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME_LLAMACPP = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
PROMPT = "Hello, how are you?"

# Output files (hidden)
OUTPUT_FILES = {
    "vllm": ".vllm_test_output.json",
    "llamacpp": ".llamacpp_test_output.json",
    "tensorrt": ".tensorrt_test_output.json"
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_current_engine():
    """Detects which engine is installed/active in the environment."""
    # Check for vLLM
    try:
        import vllm
        return "vllm"
    except ImportError:
        pass

    # Check for LlamaCPP
    try:
        import llama_cpp
        return "llamacpp"
    except ImportError:
        pass
    
    # Check for TensorRT
    try:
        import tensorrt_llm
        return "tensorrt"
    except ImportError:
        pass

    return None

def instantiate_sampler(engine):
    if engine == "vllm":
        print(f"Initializing vLLM instance with model: {MODEL_NAME_VLLM}")
        sampler = AutoregressiveSampler(
            engine="vllm",
            model=MODEL_NAME_VLLM,
            dtype="auto",
            tokenizer_path=None,
            gpu_memory_utilization=0.6,
            max_model_len=1024,
            max_probs=5,
            logits_processor=True,
            trust_remote_code=True,
            sampling_params=None
        )
        sampler.sampling_params.temperature = 1.0
        sampler.sampling_params.top_p = 1.0
        sampler.sampling_params.top_k = 1
        sampler.sampling_params.enable_normalization_constants = True
        sampler.sampling_params.enable_entropy = True
        return sampler

    elif engine == "llamacpp":
        print(f"Initializing LlamaCPP instance with model: {MODEL_NAME_LLAMACPP}")
        sampler = AutoregressiveSampler(
            engine="llama_cpp",
            model=MODEL_NAME_LLAMACPP,
            dtype="fp16",
            tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct",
            gpu_memory_utilization=0.6,
            max_model_len=1024,
            max_probs=5,
            logits_processor=True,
            trust_remote_code=True,
            sampling_params=None
        )
        sampler.sampling_params.temperature = 1.0
        sampler.sampling_params.top_p = 1.0
        sampler.sampling_params.top_k = 1
        sampler.sampling_params.enable_normalization_constants = True
        sampler.sampling_params.enable_entropy = True
        return sampler

    elif engine == "tensorrt":
        print(f"Initializing TensorRT instance with model: {MODEL_NAME_VLLM}")
        sampler = AutoregressiveSampler(
            engine="tensorrt",
            model=MODEL_NAME_VLLM,
            dtype="auto", 
            tokenizer_path=None,
            gpu_memory_utilization=0.6,
            max_model_len=1024,
            max_probs=1, 
            logits_processor=True,
            trust_remote_code=True,
            sampling_params=None
        )
        sampler.sampling_params.temperature = 1.0 
        sampler.sampling_params.top_p = 1.0
        sampler.sampling_params.top_k = 1
        sampler.sampling_params.logprobs_per_token = 1
        sampler.sampling_params.logits_per_token = 1
        sampler.sampling_params.enable_normalization_constants = True
        sampler.sampling_params.enable_entropy = True
        return sampler
    
    return None

def test_generate_engine_output():
    engine = get_current_engine()
    if not engine:
        pytest.skip("No supported engine (vLLM, LlamaCPP, TensorRT) detected.")

    print(f"Detected engine: {engine}")
    sampler = instantiate_sampler(engine)
    
    try:
        print(f"Prompting model with: '{PROMPT}'")
        output = sampler.sample(context=PROMPT)
        
        output_data = {
            "tokens": output.tokens,
            "top_k_logits": output.top_k_logits,
            "top_k_logprobs": output.top_k_logprobs,
            "entropy": output.entropy,
            "unprocessed_log_normalization_constant": output.unprocessed_log_normalization_constant,
            "temp_processed_log_normalization_constant": output.temp_processed_log_normalization_constant
        }

        output_file = OUTPUT_FILES[engine]
        print(f"Writing output to {output_file}")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4, cls=NumpyEncoder)
            
    finally:
        if 'sampler' in locals() and sampler is not None:
            del sampler
        gc.collect()

def test_compare_outputs():
    # Check if all files exist
    missing_files = [f for f in OUTPUT_FILES.values() if not os.path.exists(f)]
    
    if missing_files:
        pytest.skip(f"Skipping comparison. Missing output files: {missing_files}")

    try:
        data = {}
        tokens = {}
        
        # Load data
        for engine, filename in OUTPUT_FILES.items():
            with open(filename, "r") as f:
                content = json.load(f)
                data[engine] = content
                tokens[engine] = content.get("tokens", [])

        # Compare tokens
        base_engine = "vllm" # Use lowercase key from OUTPUT_FILES
        if base_engine not in tokens:
             # If vllm isn't one of the keys (e.g. if we changed keys), pick arbitrary first
             base_engine = next(iter(tokens))

        base_tokens = tokens[base_engine]
        
        for engine, engine_tokens in tokens.items():
            if engine == base_engine:
                continue
                
            assert np.array_equal(base_tokens, engine_tokens), f"Mismatch between {base_engine} and {engine}"
            
    finally:
        # Cleanup
        print("Cleaning up output files...")
        for filename in OUTPUT_FILES.values():
            if os.path.exists(filename):
                os.remove(filename)
