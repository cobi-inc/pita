#Library Code
from pita.sampling.power_sample import Power_Sampling
from pita.api.test_time_coding import encode
# Standard Libraries
import requests
from openai import OpenAI

'''
Start the server before running this file with:
python -m src.api.serve_power_sampling.py
'''
# Create a Test Log
with open("api_test.log", "w") as log_file:
    log_file.write("Starting Test Log for PITA API with Test-Time Coding\n")

# Initialize the OpenAI client to interact with the local PITA server
local_url = "http://localhost:8001/v1"
local_api_key = "not-needed-for-local"
client = OpenAI(
    base_url=local_url,
    api_key=local_api_key  # API key is often not required for local models
)

with open("api_test.log", "a") as log_file:
    log_file.write(f"Local Url: {local_url}\n")
    log_file.write(f"Local API Key: {local_api_key}\n")

# Send just a regular message to verify server is working
messages = [
    {"role": "system", "content": f"You are a personal assistant."},
    {"role": "user", "content": "Hello! How are you today? Is my message still incomplete? What is 2+2?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. Your message seems complete. To answer your question, 2 + 2 equals 4. Let me know if there's anything else I can assist you with!<|im_end|>"},
    {"role": "user", "content": "What is 1+1+1+1+2? Write a story about it."}
]

# Log the messages being sent to the model
with open("api_test.log", "a") as log_file:
    log_file.write("Messages Sent to the Model:\n")
    for message in messages:
        log_file.write(f"  {message['role']}: {message['content']}\n")

# Send the message with the OpenAI client
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=messages,
    temperature= 0.25,
    max_tokens=1024,
    n=1
)

# Log the response from the OpenAI client
with open("api_test.log", "a") as log_file:
    log_file.write("Response from the Model using OpenAI Client w/o PITA:\n")
    log_file.write(f"  {response.choices[0].message.content}\n\n")

# Send a message with test-time coding for Power Sampling
block_size = 250
MCMC_steps = 3
its_params = encode(
    Power_Sampling_Params = Power_Sampling_Params(
        block_size=block_size,
        MCMC_steps=MCMC_steps
    )
)

# Message with the encoded ITS parameters in the system prompt
messages = [
    {"role": "system", "content": f"{its_params} You are a personal assistant."},
    {"role": "user", "content": "Hello! How are you today? Is my message still incomplete? What is 2+2?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. Your message seems complete. To answer your question, 2 + 2 equals 4. Let me know if there's anything else I can assist you with!<|im_end|>"},
    {"role": "user", "content": "What is 1+1+1+1+2? Write a story about it."}
]

# Log the encoded ITS parameters
with open("api_test.log", "a") as log_file:
    log_file.write("Power Sampling Parameters:\n")
    log_file.write(f"  Block Size: {block_size}\n")
    log_file.write(f"  MCMC Steps: {MCMC_steps}\n")
    log_file.write(f"Encoded ITS Params: {its_params}\n")

# Log the messages being sent to the model
with open("api_test.log", "a") as log_file:
    log_file.write("Messages Sent to the Model:\n")
    for message in messages:
        log_file.write(f"  {message['role']}: {message['content']}\n")

# Send the message with the OpenAI client
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=messages,
    temperature= 0.25,
    max_tokens=1024,
    n=1
)

# Log the response from the OpenAI client
with open("api_test.log", "a") as log_file:
    log_file.write("Response from the Model using OpenAI Client:\n")
    log_file.write(f"  {response.choices[0].message.content}\n\n")

# Without using the openAI client
# Check if the server can be used without the openAI client
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

# Log the response from requests.post
with open("api_test.log", "a") as log_file:
    log_file.write("Response from the Model without using OpenAI Client:\n")
    log_file.write(f"  {response.json()['choices'][0]['message']['content']}\n")
