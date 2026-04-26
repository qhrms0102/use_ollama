OLLAMA CLOUD를 쉽게 사용할 수 있도록 만든 구조

추가로 `deepagent_demo/` 에 아래 예시를 포함했습니다.
- deepagents 기반 데이터프레임 분석 백엔드(FastAPI + SSE)
- Vite + React 프론트엔드(subagent 스트리밍 표시)

실행 방법: `deepagent_demo/README.md`

## .env

`.env`는 로컬에만 두고, 아래 키를 설정하면 됩니다.

```env
USE_CLOUD=true
MODEL_MAP_JSON={"model_1":"gemma4:31b","model_2":"minimax-m2.7:cloud"}
ACTIVE_MODEL=model_1
OLLAMA_API_KEY=your_api_key
OLLAMA_HOST=https://ollama.com
LOCAL_HOST=http://localhost:11434
```

설명:
- `USE_CLOUD`: `true`면 `OLLAMA_HOST`, `false`면 `LOCAL_HOST` 사용
- `MODEL_MAP_JSON`: 모델 키와 실제 모델명을 매핑한 JSON 문자열
- `ACTIVE_MODEL`: 기본으로 사용할 모델 키
- `OLLAMA_API_KEY`: Ollama Cloud 사용 시 API 키
