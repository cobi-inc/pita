import requests
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed-for-local"  # API key is often not required for local models
)

messages = [
    {"role": "system", "content": "You are a personal assistant."},
    {"role": "user", "content": "Hello! How are you today? Is my message still incomplete? What is 2+2?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. Your message seems complete. To answer your question, 2 + 2 equals 4. Let me know if there's anything else I can assist you with!<|im_end|>"},
    {"role": "user", "content": "What is 1+1+1+1+2? Write a story about it."}

]

# Send the message with the OpenAI client
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=messages,
    temperature= 0.25,
    max_tokens=1024,
    n=1
)

# Without using the openAI client
# response = requests.post(
#     "http://localhost:8001//v1/chat/completions",
#     json={
#         "model": "Qwen/Qwen3-4B-AWQ",
#         "messages": messages,
#         "temperature": 0.25,
#         "max_tokens": 1024,
#         "n": 1
#     }
# )

print(response.choices[0].message.content)