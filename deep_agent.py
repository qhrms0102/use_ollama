import os
os.environ["no_proxy"] = "*"

from langchain_core.tools import tool
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent
from services.llm_service import create_llm

import langchain
langchain.debug = True


# =========================
# Tools
# =========================

@tool(parse_docstring=True)
def get_weather(city: str) -> dict:
    """Look up weather data for a city.

    Use this when the agent needs current weather information for a city.
    The returned payload can include temperature, unit, humidity, and
    precipitation values.

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

    Use this when a fetched weather result is expressed in Fahrenheit and
    needs to be normalized before comparison.

    Args:
        temp: Numeric temperature value
        unit: Temperature unit such as 'C' or 'F'
    """
    if unit == "F":
        return round((temp - 32) * 5 / 9, 1)
    return temp


@tool(parse_docstring=True)
def recommend_clothing(temp_c: float, cond: str) -> str:
    """Recommend clothing based on temperature and weather condition.

    Use this after weather data has been normalized to Celsius.

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


# =========================
# Middleware
# =========================

@wrap_tool_call
async def log_tool_calls_async(request, handler):
    tc = request.tool_call
    print(f"🔧 TOOL: {tc.get('name')} / {tc.get('args')}")
    return await handler(request)


COMMON_MIDDLEWARE = [log_tool_calls_async]


agent_llm = create_llm("model_2")


# =========================
# Agent Config
# =========================

SUBAGENTS = [
    {
        "name": "weather-researcher",
        "description": "도시별 날씨를 조회하고 필요하면 화씨를 섭씨로 변환해 정리하는 전문 에이전트",
        "system_prompt": """
너는 날씨 조회와 비교를 담당하는 서브 에이전트다.

[역할]
1. 요청에 포함된 각 도시의 날씨를 get_weather로 조회한다.
2. 화씨 온도는 convert_to_celsius로 섭씨로 변환한다.
3. 도시별 온도, 날씨 상태, 습도, 강수 확률을 실제 조회값 기준으로 정리한다.
4. 도시 간 비교가 필요하면 가장 따뜻한 도시, 더 습한 도시, 비 가능성이 높은 도시 등을 직접 판단해 요약한다.
5. 같은 대화에서 이미 조회한 도시 날씨가 최근 메시지에 명시되어 있고, 사용자가 새로고침/최신 재조회/다시 조회를 요청하지 않았다면 그 값을 재사용한다.
6. 절대로 조회하지 않은 데이터를 지어내지 않는다.

[출력]
- 도시별 요약
- 필요한 비교 판단
- 옷차림 추천은 하지 않는다
""",
        "tools": [get_weather, convert_to_celsius],
    },
    {
        "name": "clothing-advisor",
        "description": "정규화된 날씨 정보를 바탕으로 도시별 옷차림을 추천하는 전문 에이전트",
        "system_prompt": """
너는 옷차림 추천 서브 에이전트다.

[역할]
1. 입력으로 받은 도시별 온도와 날씨 상태를 바탕으로 recommend_clothing을 호출한다.
2. 각 도시별로 짧고 실용적인 추천을 만든다.
3. 입력에 없는 날씨 데이터는 추정하지 않는다.

[출력]
- 도시별 추천만 간단히 정리한다
""",
        "tools": [recommend_clothing],
    },
]


MAIN_SYSTEM_PROMPT = """
너는 날씨 분석을 조정하는 메인 에이전트다.

[규칙]
1. 메인 에이전트는 직접 날씨/옷차림 tool을 호출하지 않는다.
2. 날씨 조회, 정규화, 비교 판단이 필요하면 반드시 weather-researcher 서브 에이전트에 위임한다.
3. 옷차림 추천이 필요하면 반드시 clothing-advisor 서브 에이전트에 위임한다.
4. 비교와 옷차림 요청이 함께 있으면 weather-researcher 결과를 바탕으로 clothing-advisor에 필요한 정보만 넘긴다.
5. 같은 대화에서 이미 조회된 도시 날씨가 최근 대화 기록에 있으면, 사용자가 최신 재조회를 요구하지 않는 한 그 정보를 우선 재사용한다.
6. 절대로 데이터를 지어내지 말고, 서브 에이전트가 조회하거나 계산한 실제 결과만 사용한다.
7. 최종 답변은 메인 에이전트가 사용자 친화적으로 종합한다.
8. 단순 인사처럼 delegation이 불필요한 경우만 직접 답하고, 날씨 관련 실질 작업은 항상 서브 에이전트에 위임한다.

[출력]
- 불필요한 내부 위임 과정을 설명하지 말고, 결과만 자연스럽게 정리한다.
- 비교 결과와 옷차림 요청이 함께 있으면 둘 다 자연스럽게 정리해서 답한다.
"""


# =========================
# Main Agent
# =========================

def create_agent_async():
    return create_deep_agent(
        model=agent_llm,
        name="main-agent",
        tools=[],
        subagents=SUBAGENTS,
        middleware=COMMON_MIDDLEWARE,
        system_prompt=MAIN_SYSTEM_PROMPT,
    )
