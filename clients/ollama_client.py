from ollama import Client
from config.settings import settings


def get_ollama_client():
    if settings.use_cloud:
        if not settings.api_key:
            raise ValueError("OLLAMA_API_KEY is required for cloud mode")

        return Client(
            host=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}"
            }
        )
    else:
        return Client(host=settings.base_url)
