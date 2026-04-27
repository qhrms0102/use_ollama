from clients.client import get_client
from config.settings import settings
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


class FinalSafeChatModel(ChatOllama):
    """AIMessage 의 content 가 비어있는 경우를 방어하는 래퍼 클래스."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        for msg in messages:
            if isinstance(msg, AIMessage):
                if not msg.content:
                    msg.content = " "
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        for msg in messages:
            if isinstance(msg, AIMessage):
                if not msg.content:
                    msg.content = " "
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def create_llm(model_key: str = "model_1", temperature: float = 0.1) -> FinalSafeChatModel:
    model_name = settings.get_model(model_key)

    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else None
    client_kwargs = {"headers": headers} if headers else {}

    return FinalSafeChatModel(
        model=model_name,
        base_url=settings.base_url,
        temperature=temperature,
        disable_streaming="tool_calling",
        client_kwargs=client_kwargs,
        sync_client_kwargs=client_kwargs,
        async_client_kwargs=client_kwargs,
    )


def create_chat_model(model_key: str | None = None, temperature: float = 0.1) -> ChatOllama:
    return create_llm(model_key=model_key or "model_1", temperature=temperature)


def chat(prompt: str, model_key: str | None = None) -> str:
    client = get_client()
    model_name = settings.get_model(model_key)

    print(f"\n[사용 모델: {model_name}]\n")

    response_text = ""
    for chunk in client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        content = chunk["message"]["content"]
        if content:
            print(content, end="", flush=True)
            response_text += content

    return response_text
