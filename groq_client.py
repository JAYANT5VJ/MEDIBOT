import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq_chat(messages, model="llama3-8b-8192", temperature=0.2):
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return "Groq API key not found. Please set GROQ_API_KEY in your environment or Streamlit secrets."

    headers = {
        "Authorization": f"Bearer {api_key}",
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
