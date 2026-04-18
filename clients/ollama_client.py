from ollama import Client
from config.settings import settings


def get_ollama_client():
    if settings.use_cloud:
        if not settings.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY is required for cloud mode")

        return Client(
            host=settings.ollama_host,
            headers={
                "Authorization": f"Bearer {settings.ollama_api_key}"
            }
        )
    else:
        return Client(host=settings.local_host)