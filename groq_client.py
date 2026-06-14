import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_chat(messages, model="llama-3.1-8b-instant", temperature=0.2, api_key=None):

    # Use passed key, then env var
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not key:
        return "Groq API key not found. Please set GROQ_API_KEY."

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            return f"Groq error: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Groq connection error: {str(e)}"
