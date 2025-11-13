import requests

messages = [
    {"role": "system", "content": "You are a personal assistant."},
    {"role": "user", "content": "Hello! How are you today? Is my message still incomplete? What is 2+2?"},
    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. Your message seems complete. To answer your question, 2 + 2 equals 4. Let me know if there's anything else I can assist you with!<|im_end|>"},
    {"role": "user", "content": "What is 1+1+1+1+2?"}

]

response = requests.post(
    "http://localhost:8001/power_sample",
    json={
        "model": "Qwen/Qwen3-4B-AWQ",
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 1024,
        "n": 1
    }
)

print(response.json()["content"])