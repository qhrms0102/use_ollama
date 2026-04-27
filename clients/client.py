from ollama import Client
from config.settings import settings


def get_client():
    headers = (
        {"Authorization": f"Bearer {settings.api_key}"}
        if settings.api_key
        else None
    )

    return Client(
        host=settings.base_url,
        headers=headers,
    )