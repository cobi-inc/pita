#Library Code
from pits.sampling.power_sample import Power_Sampling_Params
from pits.api.test_time_coding import encode
# Standard Libraries
import requests
from openai import OpenAI

'''
Start the server before running this file with:
python -m src.api.serve_power_sampling.py
'''

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed-for-local"  # API key is often not required for local models
)

its_params = encode(
    Power_Sampling_Params = Power_Sampling_Params(
        total_output_tokens=10000,
        block_size=250,
        MCMC_steps=3
    )
)

messages = [
    {"role": "system", "content": f"{its_params} You are a personal assistant."},
    {"role": "user", "content": "Hello! How are you today? Is my message still incomplete? What is 2+2?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. Your message seems complete. To answer your question, 2 + 2 equals 4. Let me know if there's anything else I can assist you with!<|im_end|>"},
    {"role": "user", "content": "What is 1+1+1+1+2? Write a story about it."}
]

# Send the message with the OpenAI client
# response = client.chat.completions.create(
#     model="Qwen/Qwen3-4B-AWQ",
#     messages=messages,
#     temperature= 0.25,
#     max_tokens=1024,
#     n=1,
#     MCMC_steps=10,
#     block_size=400
# )
# print(response.choices[0].message.content)

# Without using the openAI client
response = requests.post(
    "http://localhost:8001//v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-4B-AWQ",
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 200,
        "n": 1
    }
)

response_data = response.json()
print(response_data["choices"][0]["message"]["content"])