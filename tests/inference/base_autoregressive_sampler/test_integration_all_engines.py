import pytest
import subprocess
import os

# Environment names
ENV_NAMES = {
    "llamacpp": "pita_llamacpp_cuda",
    "tensorrt": "pita_tensorrt_cuda",
    "vllm": "pita_vllm_cuda"
}

TEST_FILE = "tests/inference/base_autoregressive_sampler/test_engine_check.py"

def run_test_in_env(env_name):
    """Runs the test file using the specified conda environment."""
    print(f"\n[{env_name}] Running test...")
    target_env = ENV_NAMES[env_name]
    
    # Use conda run to execute the test in the specific environment
    # This works across different machines and users
    cmd = ["conda", "run", "-n", target_env, "python", "-m", "pytest", TEST_FILE]
    
    try:
        # Run pytest in subprocess
        subprocess.check_call(cmd)
        print(f"[{env_name}] Success")
    except FileNotFoundError:
        pytest.fail("conda command not found. Please ensure conda is in your PATH.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[{env_name}] Test failed in environment '{target_env}' with exit code {e.returncode}")

def test_run_all_engines_integration():
    """
    Orchestrates the cross-environment integration test.
    The order matters because the final test (vLLM) performs the comparison and cleanup.
    """
    
    # 1. Run LlamaCPP (Generates .llamacpp_test_output.json)
    run_test_in_env("llamacpp")
    
    # 2. Run TensorRT (Generates .tensorrt_test_output.json)
    run_test_in_env("tensorrt")
    
    # 3. Run vLLM (Generates .vllm_test_output.json, Compares all 3, Cleans up)
    run_test_in_env("vllm")
