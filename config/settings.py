import json
from pydantic_settings import BaseSettings
from typing import Dict


class OllamaSettings(BaseSettings):
    model_map_json: str
    active_model: str = "model_1"

    use_cloud: bool = True

    ollama_api_key: str | None = None
    ollama_host: str = "https://ollama.com"
    local_host: str = "http://localhost:11434"

    class Config:
        env_file = ".env"

    def _get_model_map(self) -> Dict[str, str]:
        return json.loads(self.model_map_json)

    # 선택 가능하게 확장
    def get_model(self, model_key: str | None = None) -> str:
        model_map = self._get_model_map()

        key = model_key or self.active_model

        if key not in model_map:
            raise ValueError(
                f"Invalid model key: {key}, available: {list(model_map.keys())}"
            )

        return model_map[key]

    @property
    def api_key(self) -> str | None:
        return self.ollama_api_key

    @property
    def base_url(self) -> str:
        return self.ollama_host if self.use_cloud else self.local_host


settings = OllamaSettings()
