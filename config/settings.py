import json
from pydantic_settings import BaseSettings
from typing import Dict


class LLMSettings(BaseSettings):
    model_map_json: str
    active_model: str = "model_1"

    api_key: str | None = None
    api_host: str = "http://localhost:11434"

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
    def base_url(self) -> str:
        return self.api_host


settings = LLMSettings()
