import json
import os
from typing import Any, Literal, TypedDict

import langchain
from deepagents import create_deep_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from services.llm_service import create_llm

os.environ["no_proxy"] = "*"
langchain.debug = True


RouteName = Literal["weather_analysis", "clothing_recommendation", "general_answer"]


class HybridAgentState(TypedDict, total=False):
    messages: list[BaseMessage | dict[str, str]]
    route: RouteName
    route_reason: str
    question: str
    extracted_scope: dict[str, Any]
    structured_context: dict[str, Any]
    rag_context: list[dict[str, str]]
    validation_notes: list[str]


# =========================
# Example Tools
# =========================


@tool(parse_docstring=True)
def get_weather(city: str) -> dict:
    """Look up weather data for a city.

    Args:
        city: City name in Korean or English
    """
    city_map = {"Seoul": "서울", "Tokyo": "도쿄", "New York": "뉴욕"}
    city = city_map.get(city, city)

    data = {
        "서울": {"cond": "맑음", "temp": 18, "unit": "C", "humidity": 60, "precipitation": 10},
        "도쿄": {"cond": "흐림", "temp": 22, "unit": "C", "humidity": 55, "precipitation": 5},
        "뉴욕": {"cond": "비", "temp": 59, "unit": "F", "humidity": 70, "precipitation": 80},
    }
    return data.get(city, {"error": "not_found"})


@tool(parse_docstring=True)
def convert_to_celsius(temp: float, unit: str) -> float:
    """Convert a temperature value to Celsius.

    Args:
        temp: Numeric temperature value
        unit: Temperature unit such as C or F
    """
    if unit == "F":
        return round((temp - 32) * 5 / 9, 1)
    return temp


@tool(parse_docstring=True)
def recommend_clothing(temp_c: float, cond: str) -> str:
    """Recommend clothing based on temperature and weather condition.

    Args:
        temp_c: Temperature in Celsius
        cond: Weather condition summary such as 맑음, 흐림, or 비
    """
    if temp_c >= 25:
        base = "반팔, 반바지"
    elif temp_c >= 15:
        base = "긴팔, 얇은 가디건"
    else:
        base = "니트, 패딩"

    if "비" in cond or "흐림" in cond:
        base += " + 우산 추천"
    return base


@tool(parse_docstring=True)
def make_user_action_tip(city: str, temp_c: float, cond: str, precipitation: int) -> str:
    """Create a practical one-line action tip for a city.

    Args:
        city: City name
        temp_c: Temperature in Celsius
        cond: Weather condition summary
        precipitation: Precipitation probability from 0 to 100
    """
    if precipitation >= 50 or "비" in cond:
        return f"{city}: 우산과 방수 신발을 우선 준비하세요."
    if temp_c >= 25:
        return f"{city}: 가벼운 옷차림과 수분 보충을 챙기세요."
    if temp_c <= 12:
        return f"{city}: 아침·저녁 보온용 겉옷을 준비하세요."
    return f"{city}: 얇은 외투를 챙기면 온도 변화에 대응하기 좋습니다."


# =========================
# Middleware
# =========================


@wrap_model_call
async def fix_all_messages_async(request, handler):
    """tool_calls만 있고 content가 비어 있는 메시지를 사내 API 호환 형태로 보정한다."""
    for msg in request.messages:
        if isinstance(msg, dict):
            if not msg.get("content"):
                msg["content"] = " "
        elif hasattr(msg, "content"):
            try:
                if not msg.content:
                    msg.content = " "
            except Exception:
                pass
    return await handler(request)


@wrap_tool_call
async def log_tool_calls_async(request, handler):
    tc = request.tool_call
    print(f"TOOL: {tc.get('name')} / {tc.get('args')}")
    return await handler(request)


COMMON_MIDDLEWARE = [fix_all_messages_async, log_tool_calls_async]


# =========================
# Deep Agents
# =========================


agent_llm = create_llm("model_2")


PLANNER_SYSTEM_PROMPT = """
너는 워크플로우 플래너다.

[역할]
1. 사용자 질문을 보고 어떤 LangGraph branch로 보내야 하는지 제안한다.
2. 실제 데이터 조회나 최종 분석은 하지 않는다.
3. 허용된 route 중 하나만 고른다.

[허용 route]
- weather_analysis: 날씨 조회, 날씨 비교, 강수/습도/온도 분석
- clothing_recommendation: 옷차림 추천이 포함된 요청
- general_answer: 위 범주가 아니거나 간단한 일반 대화

[출력]
반드시 JSON만 출력한다.
예:
{"route":"weather_analysis","reason":"서울과 도쿄 날씨 비교 요청"}
"""


ANALYZER_SYSTEM_PROMPT = """
너는 LangGraph가 수집, 정규화, 검증한 컨텍스트 안에서 최종 해석을 수행하는 deep agent다.

[규칙]
1. 날씨 조회, 단위 변환, 도시 간 수치 비교는 이미 LangGraph가 끝낸 것으로 간주한다.
2. get_weather 같은 원천 조회를 다시 시도하지 않는다.
3. structured_context, validation_notes, rag_context 밖의 사실을 단정하지 않는다.
4. 필요한 경우 clothing-advisor에게 옷차림과 실용 팁만 위임한다.
5. 최종 답변은 사용자에게 바로 보여줄 수 있게 자연스럽게 작성한다.
"""


ANALYZER_SUBAGENTS = [
    {
        "name": "weather-interpreter",
        "description": "LangGraph가 준비한 날씨 비교 결과를 사용자 관점으로 해석하는 전문 에이전트",
        "system_prompt": """
너는 날씨 해석 담당 서브 에이전트다.
이미 제공된 structured_context의 수치와 비교 결과만 사용한다.
새로운 날씨 조회나 단위 변환을 하지 않는다.
사용자에게 중요한 차이와 주의할 점을 짧게 정리한다.
""",
        "tools": [],
        "middleware": COMMON_MIDDLEWARE,
    },
    {
        "name": "clothing-advisor",
        "description": "정규화된 날씨 정보를 바탕으로 옷차림과 행동 팁을 추천하는 전문 에이전트",
        "system_prompt": """
너는 옷차림 추천 담당 서브 에이전트다.
입력된 온도, 날씨 상태, 강수 확률을 기준으로 recommend_clothing과 make_user_action_tip을 호출한다.
입력에 없는 날씨 데이터는 추정하지 않는다.
""",
        "tools": [recommend_clothing, make_user_action_tip],
        "middleware": COMMON_MIDDLEWARE,
    },
]


planner_agent = create_deep_agent(
    model=agent_llm,
    name="planner-agent",
    tools=[],
    subagents=[],
    middleware=COMMON_MIDDLEWARE,
    system_prompt=PLANNER_SYSTEM_PROMPT,
)


analyzer_agent = create_deep_agent(
    model=agent_llm,
    name="analysis-agent",
    tools=[],
    subagents=ANALYZER_SUBAGENTS,
    middleware=COMMON_MIDDLEWARE,
    system_prompt=ANALYZER_SYSTEM_PROMPT,
)


# =========================
# LangGraph Nodes
# =========================


def _last_user_text(messages: list[BaseMessage | dict[str, str]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            if message.get("role") == "user":
                return message.get("content", "")
            continue

        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))

    return ""


def _last_ai_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def _fallback_route(question: str) -> RouteName:
    if any(word in question for word in ("옷", "복장", "입", "clothing")):
        return "clothing_recommendation"
    if any(word in question for word in ("날씨", "온도", "습도", "비", "weather")):
        return "weather_analysis"
    return "general_answer"


def _to_celsius(weather: dict[str, Any]) -> float | None:
    temp = weather.get("temp")
    unit = weather.get("unit")
    if not isinstance(temp, (int, float)):
        return None
    if unit == "F":
        return convert_to_celsius.invoke({"temp": temp, "unit": unit})
    return float(temp)


async def planner_node(state: HybridAgentState) -> dict[str, Any]:
    # 초반 deep agent가 질문을 보고 어떤 branch로 보낼지 제안한다.
    question = _last_user_text(state.get("messages", []))
    result = await planner_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"질문을 route로 분류해줘.\n\n질문: {question}",
                }
            ]
        }
    )
    planner_text = _last_ai_text(result)

    try:
        parsed = json.loads(planner_text)
        route = parsed.get("route")
        if route not in ("weather_analysis", "clothing_recommendation", "general_answer"):
            route = _fallback_route(question)
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        route = _fallback_route(question)
        reason = "planner JSON 파싱 실패로 fallback 라우팅 적용"

    return {"question": question, "route": route, "route_reason": reason}


def extract_scope_node(state: HybridAgentState) -> dict[str, Any]:
    # LangGraph가 반드시 필요한 입력 조건을 구조화하는 단계다.
    question = state.get("question", "")
    city_candidates = ("서울", "도쿄", "뉴욕", "Seoul", "Tokyo", "New York")
    cities = [city for city in city_candidates if city in question]

    # 실제 제조 챗봇에서는 fab, line, tool_id, lot_id, 기간 등을 여기서 강제 추출한다.
    return {
        "extracted_scope": {
            "cities": cities or ["서울"],
            "needs_clothing": state.get("route") == "clothing_recommendation" or "옷" in question,
        }
    }


def weather_data_node(state: HybridAgentState) -> dict[str, Any]:
    # 정형 데이터 조회 노드 예시다. 실제로는 MES/FDC/SPC 조회로 바뀐다.
    scope = state.get("extracted_scope", {})
    cities = scope.get("cities", ["서울"])

    raw_weather_by_city = {}
    normalized_weather_by_city = {}
    for city in cities:
        raw_weather = get_weather.invoke({"city": city})
        raw_weather_by_city[city] = raw_weather
        normalized_weather_by_city[city] = {
            **raw_weather,
            "temp_c": _to_celsius(raw_weather),
        }

    return {
        "structured_context": {
            "raw_weather_by_city": raw_weather_by_city,
            "normalized_weather_by_city": normalized_weather_by_city,
        }
    }


def validate_weather_data_node(state: HybridAgentState) -> dict[str, Any]:
    # LangGraph가 분석 전에 데이터 품질과 누락 여부를 확인한다.
    context = state.get("structured_context", {})
    weather_by_city = context.get("normalized_weather_by_city", {})
    notes = []

    for city, weather in weather_by_city.items():
        if weather.get("error"):
            notes.append(f"{city}: 날씨 데이터를 찾지 못했습니다.")
        if weather.get("temp_c") is None:
            notes.append(f"{city}: 온도 값을 섭씨로 정규화하지 못했습니다.")

    return {"validation_notes": notes}


def compare_weather_node(state: HybridAgentState) -> dict[str, Any]:
    # 단순 수치 비교는 LLM이 아니라 코드로 고정해 재현성을 높인다.
    context = state.get("structured_context", {})
    weather_by_city = context.get("normalized_weather_by_city", {})
    valid_items = [
        (city, weather)
        for city, weather in weather_by_city.items()
        if isinstance(weather.get("temp_c"), (int, float))
    ]

    comparison: dict[str, Any] = {"city_count": len(valid_items)}
    if len(valid_items) >= 2:
        warmest = max(valid_items, key=lambda item: item[1]["temp_c"])
        coldest = min(valid_items, key=lambda item: item[1]["temp_c"])
        comparison = {
            **comparison,
            "warmest_city": warmest[0],
            "coldest_city": coldest[0],
            "temperature_gap_c": round(warmest[1]["temp_c"] - coldest[1]["temp_c"], 1),
        }

    return {
        "structured_context": {
            **context,
            "comparison": comparison,
        }
    }


def rag_context_node(state: HybridAgentState) -> dict[str, Any]:
    # 예시용 mock RAG다. 실제로는 vector DB나 검색 API를 호출하는 노드로 교체한다.
    route = state.get("route")
    if route == "clothing_recommendation":
        docs = [
            {
                "title": "봄철 옷차림 가이드",
                "content": "15도 이상 25도 미만에서는 긴팔과 얇은 외투를 권장한다.",
            }
        ]
    else:
        docs = [
            {
                "title": "날씨 비교 기준",
                "content": "온도, 습도, 강수 확률을 함께 비교하면 체감 조건을 설명하기 쉽다.",
            }
        ]
    return {"rag_context": docs}


async def analysis_node(state: HybridAgentState) -> dict[str, Any]:
    # 후반 deep agent는 LangGraph가 모은 근거 안에서만 분석한다.
    question = state.get("question", "")
    payload = {
        "route": state.get("route"),
        "route_reason": state.get("route_reason"),
        "extracted_scope": state.get("extracted_scope", {}),
        "structured_context": state.get("structured_context", {}),
        "validation_notes": state.get("validation_notes", []),
        "rag_context": state.get("rag_context", []),
    }

    result = await analyzer_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "아래 LangGraph 컨텍스트만 근거로 답변해줘.\n\n"
                        f"사용자 질문:\n{question}\n\n"
                        f"컨텍스트 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
                    ),
                }
            ]
        }
    )
    answer = _last_ai_text(result)

    return {"messages": [AIMessage(content=answer)]}


async def general_answer_node(state: HybridAgentState) -> dict[str, Any]:
    # 정형 조회가 필요 없는 질문은 가벼운 답변 branch로 보낸다.
    result = await analyzer_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "아래 질문은 정형 분석 branch가 필요하지 않다. "
                        "간단하고 자연스럽게 답해줘.\n\n"
                        f"질문: {state.get('question', '')}"
                    ),
                }
            ]
        }
    )
    return {"messages": [AIMessage(content=_last_ai_text(result))]}


def route_after_planner(state: HybridAgentState) -> RouteName:
    # planner 제안을 LangGraph 분기 키로 변환한다.
    return state.get("route", "general_answer")


# =========================
# Graph Factory
# =========================


def create_agent_async():
    graph = StateGraph(HybridAgentState)

    # 노드 이름은 trace와 디버깅에서 보이므로 역할이 드러나게 둔다.
    graph.add_node("planner_deep_agent", planner_node)
    graph.add_node("extract_required_scope", extract_scope_node)
    graph.add_node("load_weather_data", weather_data_node)
    graph.add_node("validate_weather_data", validate_weather_data_node)
    graph.add_node("compare_weather_metrics", compare_weather_node)
    graph.add_node("retrieve_rag_context", rag_context_node)
    graph.add_node("analysis_deep_agent", analysis_node)
    graph.add_node("general_answer", general_answer_node)

    graph.set_entry_point("planner_deep_agent")
    # planner deep agent의 제안 이후에는 LangGraph가 허용된 branch로만 이동시킨다.
    graph.add_conditional_edges(
        "planner_deep_agent",
        route_after_planner,
        {
            "weather_analysis": "extract_required_scope",
            "clothing_recommendation": "extract_required_scope",
            "general_answer": "general_answer",
        },
    )

    # branch 내부는 조건 추출 -> 조회/정규화 -> 검증 -> 비교 -> RAG -> 분석 순서로 고정한다.
    graph.add_edge("extract_required_scope", "load_weather_data")
    graph.add_edge("load_weather_data", "validate_weather_data")
    graph.add_edge("validate_weather_data", "compare_weather_metrics")
    graph.add_edge("compare_weather_metrics", "retrieve_rag_context")
    graph.add_edge("retrieve_rag_context", "analysis_deep_agent")
    graph.add_edge("analysis_deep_agent", END)
    graph.add_edge("general_answer", END)

    return graph.compile()
