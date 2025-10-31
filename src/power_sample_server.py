import subprocess
import sys

# --- Your LLM server Parameters ---
# model_name = "mistralai/Mistral-7B-v0.1" # <-- EDIT THIS with your model
# quantization_type = None # or "awq", "gptq", etc.
# gpu_mem_util = 0.90 # e.g., 0.90 for 90% GPU memory utilization
def start_vllm_server(model_name, quantization_type, gpu_mem_util):
    """
    Launches the vLLM API server as a subprocess.
    """
    
    # These are the command-line arguments for vllm.entrypoints.api_server
    cmd = [
        sys.executable,  # Gets the current Python interpreter
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--dtype", "auto",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--logprobs-mode", "raw_logits",
        "--max_logprobs", "200000",
    ]

    # Add quantization if it's specified
    if quantization_type:
        cmd.extend(["--quantization", quantization_type])

    print("--- Starting vLLM Server ---")
    print(f"Model: {model_name}")
    print(f"GPU Memory: {gpu_mem_util * 100}%")
    print(f"Command: {' '.join(cmd)}")
    print("\nServer will start loading. This may take a few minutes...")

    try:
        # subprocess.run will block until the server is stopped (e.g., with Ctrl+C)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Server process failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n--- Server shutting down ---")

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B-AWQ"
    quantization_type = "awq"
    gpu_mem_util = 0.90
    start_vllm_server(model_name, quantization_type, gpu_mem_util)