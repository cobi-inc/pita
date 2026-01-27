import requests
import json
import time

def main():
    # Configuration
    API_URL = "http://localhost:8001/v1/chat/completions"
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Ensure this matches the model running on the server
    
    print("Ensure the PITA server is running before executing this script.")
    print("Run: python -m pita.api.serve --model Qwen/Qwen2.5-0.5B-Instruct --engine vllm")
    print("-" * 50)

    # 1. Standard Chat Completion
    print("1. Sending Standard Chat Completion Request...")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "What is Inference Time Scaling?"}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        content = result['choices'][0]['message']['content']
        print("\nResponse:")
        print(content)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server. Is it running?")
        return
    except Exception as e:
        print(f"\nError: {e}")

    print("-" * 50)

    # 2. Advanced Usage with "ITS" (Inference Time Scaling) System Prompt
    # This feature allows you to configure chain/token sampling via a specially formatted system prompt.
    # Format: "ITS <ChainMethod> <TokenMethod>"
    # Example: "ITS Best-of-N Power-Sampling" (Conceptual)
    # Based on decode() in test_time_coding.py, let's use a valid string if known, 
    # or just demonstrate the mechanism. 
    # NOTE: The exact string format depends on pita.api.test_time_coding.decode implementation.
    # Assuming a hypothetical format for demonstration or "ITS None None" to just trigger the check.
    
    print("2. Sending Request with ITS System Prompt...")
    
    # We will use a system prompt that the server parses to configure test-time scaling.
    # Format: ITS_<ChainMethod>_<Params>_<TokenMethod>_<Params>
    # Example 1: ITS_NONE_PS_5_10 (No chain sampling, Power Sampling with block_size=5, MCMC_steps=10)
    # Example 2: ITS_NONE_NONE (Explicitly disable both)
    
    # Let's use a dummy valid string that disables both to demonstrate the mechanism without error.
    its_system_prompt = "ITS_NONE_NONE" 
    
    payload_its = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": its_system_prompt},
            {"role": "user", "content": "Explain the benefits of test-time compute."}
        ],
        "max_tokens": 128
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload_its)
        response.raise_for_status()
        result = response.json()
        
        content_its = result['choices'][0]['message']['content']
        print("\nResponse (ITS):")
        print(content_its)
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
