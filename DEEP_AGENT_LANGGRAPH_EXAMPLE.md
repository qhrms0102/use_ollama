# Deep Agent + LangGraph Example

이 문서는 `deep_agent_with_langgraph.py` 예시 구조를 설명한다.

목적은 Deep Agent에게 전체 흐름을 전부 맡기지 않고, LangGraph로 반드시 지나야 하는 절차를 고정한 뒤, 필요한 지점에서 Deep Agent를 분석기로 쓰는 패턴을 보여주는 것이다.

## 전체 흐름

```text
사용자 질문
  -> planner deep agent
  -> LangGraph route 결정
  -> 필수 조건 추출 및 후속 질문 범위 복원
  -> 정형 데이터 조회/정규화
  -> 데이터 검증
  -> 수치 비교
  -> RAG 조회
  -> analysis deep agent
  -> 최종 답변
```

핵심 역할 분리는 아래와 같다.

- `planner deep agent`: 질문 의도와 branch를 제안한다.
- `LangGraph`: 허용된 branch로만 이동시키고 필수 노드를 강제한다.
- `RAG node`: 문서, SOP, 과거 리포트 같은 비정형 근거를 가져온다.
- `structured data node`: DB, MES, FDC, SPC 같은 정형 데이터를 가져오고 표준 형태로 정규화한다.
- `validation/comparison node`: 누락 데이터와 단순 수치 비교를 코드로 처리한다.
- `analysis deep agent`: 모인 근거를 사용자 관점으로 해석하고 최종 답변을 수행한다.

## 왜 이렇게 나누는가

사내 제조 데이터 챗봇에서는 LLM이 바로 답을 생성하게 두기 어렵다.

예를 들어 반도체 제조업 데이터는 아래 조건이 중요하다.

- 사용자가 볼 수 있는 데이터 범위
- fab, line, tool_id, chamber, recipe, lot, wafer, 기간 같은 필수 조건
- 정형 데이터와 문서 근거의 구분
- 근거 없는 추론 차단
- 민감 정보 노출 방지

그래서 LangGraph가 업무 흐름, 조회, 정규화, 검증 절차를 잡고, Deep Agent는 그 결과를 해석하는 구조가 적합하다.

중요한 점은 LangGraph와 Deep Agent가 같은 일을 반복하지 않게 나누는 것이다.

- LangGraph: 반드시 필요한 데이터 처리, 정규화, 검증, 재현 가능한 계산
- Deep Agent: 컨텍스트 해석, 우선순위 판단, 사용자에게 설명할 답변 구성

## 노드별 역할

### `planner_node`

초반 Deep Agent가 사용자 질문을 보고 route를 제안한다.

현재 예시 route:

- `weather_analysis`
- `clothing_recommendation`
- `general_answer`

제조 도메인에서는 아래처럼 바꿀 수 있다.

- `yield_analysis`
- `equipment_alarm_analysis`
- `recipe_comparison`
- `quality_defect_analysis`
- `sop_document_qa`
- `general_answer`

### `route_after_planner`

planner의 제안을 LangGraph 분기 키로 변환한다.

중요한 점은 planner가 제안하더라도 실제 이동은 LangGraph가 허용한 branch 안에서만 일어난다는 것이다.

### `extract_scope_node`

질문에서 필수 조건을 구조화한다.

현재 예시는 도시명을 추출한다. 현재 질문에 도시가 없으면 이전 대화 history에서 최근에 함께 언급된 도시 목록을 재사용한다.

예:

```text
1차 질문: 서울, 도쿄, 뉴욕 중 가장 따뜻한 도시를 찾아줘
2차 질문: 그럼 각 도시의 옷차림도 추천해줘
```

2차 질문에는 도시명이 없지만, 이전 질문의 `서울`, `도쿄`, `뉴욕`을 다시 사용한다.

제조 도메인에서는 아래 항목을 추출하는 노드로 바꿀 수 있다.

- fab
- line
- tool_id
- chamber
- recipe
- product
- lot_id
- wafer_id
- 기간
- 분석 목적

누락된 조건이 있으면 바로 분석하지 말고 clarification branch로 보내는 방식이 좋다.

### `weather_data_node`

정형 데이터 조회와 정규화 노드 예시다.

현재 예시는 mock weather tool을 호출한 뒤 섭씨 기준으로 온도를 정규화한다.

제조 도메인에서는 아래 조회 노드로 바꿀 수 있다.

- MES lot 이력 조회
- FDC sensor trend 조회
- SPC control chart 조회
- alarm log 조회
- recipe version 조회
- maintenance 이력 조회

이 노드는 Deep Agent가 다시 원천 조회를 하지 않도록, 분석에 필요한 원본값과 표준화된 값을 함께 state에 넣는다.

### `validate_weather_data_node`

분석 전에 데이터 품질을 확인한다.

현재 예시는 아래 조건을 본다.

- 도시별 날씨 데이터 존재 여부
- 온도 정규화 성공 여부

제조 도메인에서는 아래 검증을 넣을 수 있다.

- 조회 기간 누락
- lot 수 부족
- 설비 ID 불명확
- 권한 때문에 빠진 데이터 존재
- 결측치 또는 이상치 비율 초과

### `compare_weather_node`

단순한 수치 비교를 코드로 처리한다.

현재 예시는 가장 따뜻한 도시, 가장 추운 도시, 온도 차이를 계산한다.

제조 도메인에서는 아래 계산을 넣을 수 있다.

- 수율 변화량
- chamber별 편차
- recipe 변경 전후 평균 차이
- alarm 발생 전후 센서값 변화
- control limit 초과 여부

이런 계산은 LLM에게 맡기기보다 코드로 고정하는 편이 재현성과 신뢰성이 좋다.

### `rag_context_node`

RAG 조회 노드 예시다.

현재 예시는 mock 문서를 반환한다.

제조 도메인에서는 아래 문서를 검색할 수 있다.

- SOP
- 장비 매뉴얼
- 공정 조건 변경 이력 문서
- 과거 장애 리포트
- 품질 이슈 리포트
- 사내 기술 문서

### `analysis_node`

후반 Deep Agent가 해석과 답변 생성을 수행한다.

이 노드는 LangGraph가 모은 아래 근거를 입력으로 받는다.

- route
- route_reason
- extracted_scope
- structured_context
- validation_notes
- rag_context

이 예시에서 Deep Agent는 날씨 조회나 단위 변환을 다시 하지 않는다. LangGraph가 준비한 컨텍스트 안에서 사용자에게 중요한 차이, 주의점, 추천을 정리한다.

현재 후반 Deep Agent의 subagent 역할:

- `weather-interpreter`: 이미 계산된 날씨 비교 결과를 사용자 관점으로 해석한다.
- `clothing-advisor`: 정규화된 날씨값을 바탕으로 옷차림과 행동 팁을 만든다.

### `general_answer_node`

정형 데이터 조회가 필요 없는 질문을 처리한다.

예를 들어 단순 인사, 기능 설명, 일반적인 안내는 무거운 분석 branch를 태우지 않는다.

## main.py와의 관계

`main.py`는 `create_agent_async()`가 반환하는 runnable의 `astream_events`를 소비한다.

따라서 `deep_agent_with_langgraph.py`도 같은 이름의 `create_agent_async()`를 제공한다.

운영에서 이 구조를 쓰려면 `main.py`의 import만 아래처럼 바꾸면 된다.

```python
from deep_agent_with_langgraph import create_agent_async
```

현재 `main.py`는 LangGraph와 Deep Agent 이벤트를 작업 기록으로 저장한다.

처리하는 주요 이벤트:

- `on_chain_start`
- `on_chain_end`
- `on_tool_start`
- `on_tool_end`
- `on_retriever_start`
- `on_retriever_end`

trace 표시 정책:

- 스트리밍 중에는 최신 raw trace 하나를 현재 진행 블록으로 보여준다.
- 응답 완료 후 `작업 기록 보기`에서는 같은 tool 호출의 시작/결과를 한 카드로 합친다.
- LangGraph chain은 바로 연달아 나온 시작/결과만 합친다.
- chain 중간에 tool 호출이 끼면 `chain 지시 -> tool 결과 -> chain 결과` 순서를 유지한다.
- tool 호출 카드에는 `via clothing-advisor`, `via load_weather_data`처럼 호출 주체를 표시한다.
- chain 카드에는 `via` 배지를 표시하지 않는다.

최종 답변 본문은 모든 model chunk를 바로 붙이지 않는다. `analysis_deep_agent` 또는 `general_answer` 노드의 최종 output에서 마지막 메시지만 꺼내 사용자 답변으로 보낸다. 이렇게 해야 `clothing-advisor에게 위임하겠습니다` 같은 내부 진행 멘트가 최종 답변에 섞이지 않는다.

## 작업 기록 예시

질문:

```text
서울과 도쿄 날씨를 비교하고 옷차림도 추천해줘
```

작업 기록은 대략 아래처럼 읽힌다.

```text
planner_deep_agent
  -> route=clothing_recommendation

extract_required_scope
  -> cities=["서울", "도쿄"], needs_clothing=true

load_weather_data
  -> 지시

get_weather
  -> via load_weather_data
  -> 서울 조회 결과

get_weather
  -> via load_weather_data
  -> 도쿄 조회 결과

load_weather_data
  -> 결과 취합

validate_weather_data
  -> validation_notes=[]

compare_weather_metrics
  -> warmest_city, coldest_city, temperature_gap_c 계산

retrieve_rag_context
  -> 관련 가이드 문서 조회

analysis_deep_agent
  -> LangGraph 컨텍스트 기반 최종 답변
```

## 제조 도메인으로 확장할 때 권장 구조

```text
planner deep agent
  -> route 선택 제안

LangGraph common nodes
  -> 권한 확인
  -> 보안 정책 확인
  -> 질문 유형 확정

branch graph
  -> 필수 조건 추출
  -> 누락 조건 확인
  -> 정형 데이터 조회
  -> 데이터 정규화
  -> RAG 조회
  -> 데이터 충분성 검증
  -> 재현 가능한 계산

analysis deep agent
  -> 계산된 결과 해석
  -> 원인 후보 분석
  -> 근거 기반 설명
  -> 권장 액션 제안

LangGraph final guardrail
  -> 근거 없는 표현 제거
  -> 민감 정보 필터링
  -> 최종 답변 반환
```

## 주의점

- planner deep agent는 결정권자가 아니라 route 제안자 역할로 두는 것이 좋다.
- branch 이동과 필수 검증은 LangGraph가 통제해야 한다.
- LangGraph와 Deep Agent가 같은 조회나 계산을 반복하지 않게 역할을 분리해야 한다.
- RAG 결과와 DB 조회 결과는 구분해서 state에 저장하는 편이 좋다.
- 제조 데이터 분석에서는 누락 조건을 무시하고 답변하지 않는 것이 중요하다.
- UI에 더 깊은 계층형 trace가 필요하면 `parent_id` 저장과 프런트 트리 렌더링을 추가로 설계해야 한다.
