from clients.ollama_client import get_ollama_client
from config.settings import settings


def chat(prompt: str, model_key: str | None = None) -> str:
    client = get_ollama_client()

    model_name = settings.get_model(model_key)

    print(f"\n[사용 모델: {model_name}]\n")

    messages = [
        {"role": "user", "content": prompt}
    ]

    response_text = ""

    for chunk in client.chat(
        model=model_name,
        messages=messages,
        stream=True
    ):
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        response_text += content

    return response_text