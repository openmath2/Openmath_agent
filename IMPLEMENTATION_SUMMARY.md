# OpenMath Agents - ReAct vs Multi-Agent 비교 실험 구현 완료

## 구현 개요

3단계 ReAct vs Multi-Agent 비교 실험을 위한 전체 시스템을 구현했습니다.

## 구현된 컴포넌트

### 1. ReAct Agent (`src/agents/react_agent.py`)

- **기능**: 단일 에이전트가 SymPy 도구를 사용하여 수학 문제 해결
- **주요 변경사항**:
  - SymPy 도구 4개 연결 (sympy_solve, sympy_simplify, sympy_verify, sympy_differentiate)
  - LangGraph 최신 방식으로 `create_react_agent` 사용
  - 환경변수에서 OPENAI_BASE_URL, OPENAI_MODEL 읽기
  - System prompt를 통한 정확한 수학적 추론 가이드

### 2. Generator Agent (`src/agents/generator_agent.py`)

- **역할**: 수학 문제에 대한 해답 후보 생성
- **주요 특징**:
  - SymPy 도구를 활용한 ReAct 에이전트 내장
  - 단계별 추론 과정 기록
  - 환경변수 기반 LLM 설정
  - Temperature 0.7로 다양한 솔루션 생성

### 3. Verifier Agent (`src/agents/verifier_agent.py`)

- **역할**: 생성된 해답의 정확성 독립 검증
- **주요 특징**:
  - SymPy 도구를 사용한 기호적 검증
  - CORRECT/INCORRECT/UNCERTAIN 판정
  - 판정 근거 제공
  - Temperature 0.0으로 결정론적 검증

### 4. Multi-Agent Graph (`src/agents/multi_agent_graph.py`)

- **패턴**: Generator + Verifier Supervisor 패턴
- **워크플로우**:
  1. Generator가 해답 후보 생성
  2. Verifier가 해답 검증
  3. INCORRECT/UNCERTAIN이면 Generator로 재시도 (최대 3회)
  4. CORRECT이거나 최대 재시도 도달 시 종료
- **주요 특징**:
  - LangGraph의 조건부 엣지를 통한 동적 라우팅
  - 상태 관리를 통한 재시도 카운팅
  - 서브그래프 조합을 통한 모듈화

### 5. Dataset (`src/evaluation/dataset.py`)

- **벤치마크**: 중학교 3학년 수준 수학 문제 10개
- **문제 유형**:
  - 이차방정식 (인수분해, 완전제곱식, 제곱차)
  - 판별식
  - 일차함수
  - 연립방정식
  - 식의 전개
  - 인수분해
  - 근과 계수의 관계 (비에타 공식)
- **주요 기능**:
  - `get_middle_school_benchmark()`: 10개 문제 즉시 로드
  - `load_dataset()`: JSONL 파일에서 데이터셋 로드
  - 태그 기반 필터링 지원

## 환경 설정

모든 에이전트는 `.env` 파일의 다음 환경변수를 읽습니다:

```env
OPENAI_API_KEY=dummy-not-used
OPENAI_BASE_URL=http://127.0.0.1:8317/v1
OPENAI_MODEL=gemini-2.5-pro
```

## 사용 예시

### ReAct Agent 사용

```python
from src.agents.react_agent import build_react_agent

agent = build_react_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "x^2 - 5x + 6 = 0을 풀어라"}]
})
```

### Multi-Agent Graph 사용

```python
from src.agents.multi_agent_graph import build_multi_agent_graph

graph = build_multi_agent_graph()
result = graph.invoke({
    "problem": "x^2 - 5x + 6 = 0을 풀어라",
    "solution_candidate": "",
    "reasoning_steps": [],
    "verdict": "",
    "justification": "",
    "attempt": 0
})
```

### 벤치마크 데이터셋 사용

```python
from src.evaluation.dataset import get_middle_school_benchmark

dataset = get_middle_school_benchmark()
for problem in dataset:
    print(f"{problem.id}: {problem.problem}")
    print(f"Answer: {problem.answer}")
```

## 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 에이전트 테스트만 실행
python -m pytest tests/test_agents.py -v

# 구현 검증 스크립트 실행
python test_implementation.py
```

## 아키텍처 비교

### ReAct Agent (단일 에이전트)

- **장점**: 간단한 구조, 빠른 실행
- **단점**: 자체 검증 한계, 복잡한 문제에서 오류 가능성

### Multi-Agent (Generator + Verifier)

- **장점**: 독립적 검증, 재시도 메커니즘, 높은 정확도
- **단점**: 더 많은 LLM 호출, 느린 실행 시간

## 다음 단계

1. **실험 실행**: 두 접근법을 벤치마크 데이터셋에서 비교
2. **메트릭 수집**: 정확도, 실행 시간, 토큰 사용량 측정
3. **결과 분석**: notebooks/compare_results.ipynb에서 시각화
4. **최적화**: 성능 개선 및 하이퍼파라미터 튜닝

## 파일 구조

```
openmath-agents/
├── src/
│   ├── agents/
│   │   ├── react_agent.py          # ✅ 구현 완료
│   │   ├── generator_agent.py      # ✅ 구현 완료
│   │   ├── verifier_agent.py       # ✅ 구현 완료
│   │   └── multi_agent_graph.py    # ✅ 구현 완료
│   ├── evaluation/
│   │   └── dataset.py              # ✅ 구현 완료
│   └── tools/
│       └── sympy_tools.py          # ✅ 기존 도구 활용
├── tests/
│   └── test_agents.py              # ✅ 테스트 준비됨
└── test_implementation.py          # ✅ 검증 스크립트
```

## 구현 완료 체크리스트

- [x] ReAct Agent에 SymPy 도구 연결
- [x] LangGraph 최신 방식으로 import 수정
- [x] Generator Agent 구현 (SymPy 도구 포함)
- [x] Verifier Agent 구현 (독립 검증 로직)
- [x] Multi-Agent Graph Supervisor 패턴 구현
- [x] 중3 수준 벤치마크 문제 10개 추가
- [x] 환경변수 기반 설정 적용
- [x] 테스트 준비 완료
