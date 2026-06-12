import requests

def ollama_chat(model, messages, base_url="http://localhost:11434", temperature=0.2, stream=False):

    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            return f"Ollama error: {response.text}"

        data = response.json()

        return data["message"]["content"]

    except Exception as e:
        return f"Ollama connection error: {str(e)}"