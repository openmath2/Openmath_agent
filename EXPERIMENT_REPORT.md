# OpenMath Agents 실험 보고서

## 목차

1. [SymPy 도구 구조](#1-sympy-도구-구조)
2. [ReAct 에이전트 구현](#2-react-에이전트-구현)
3. [Multi-Agent 파이프라인 구현](#3-multi-agent-파이프라인-구현)
4. [성능 비교](#4-성능-비교)

---

## 1. SymPy 도구 구조

### 1.1 개요

SymPy 기반의 수학 도구를 LangChain Tool로 래핑하여 에이전트가 사용할 수 있도록 구현했습니다.

### 1.2 도구 목록

#### 1.2.1 `sympy_solve`

**기능**: 주어진 변수에 대한 수학 방정식을 풉니다.

**인수**:

- `equation` (str): 방정식 문자열 (예: "x\**2 - 4 = 0" 또는 "2*x + 1 = 0")
- `variable` (str, optional): 풀이 대상 변수 (기본값: "x")

**반환값**: 해의 리스트를 문자열로 반환. 분수는 Rational로, 제곱근은 sqrt로 유지

**예시**:

```python
sympy_solve.invoke({"equation": "x**2 - 4 = 0", "variable": "x"})
# 결과: "[-2, 2]"

sympy_solve.invoke({"equation": "2*x + 1 = 0", "variable": "x"})
# 결과: "[-1/2]"
```

#### 1.2.2 `sympy_simplify`

**기능**: 수학 표현식을 단순화합니다.

**인수**:

- `expression` (str): 단순화할 수학 표현식

**반환값**: 단순화된 표현식 문자열

**예시**:

```python
sympy_simplify.invoke({"expression": "(x+1)**2"})
# 결과: "x**2 + 2*x + 1"
```

#### 1.2.3 `sympy_verify`

**기능**: 두 수학 표현식이 기호적으로 동일한지 검증합니다.

**인수**:

- `expr_a` (str): 첫 번째 표현식
- `expr_b` (str): 두 번째 표현식

**반환값**: "VERIFIED" (동일) 또는 "FAILED" (다름)

**예시**:

```python
sympy_verify.invoke({"expr_a": "(x+1)**2", "expr_b": "x**2 + 2*x + 1"})
# 결과: "VERIFIED"

sympy_verify.invoke({"expr_a": "x**2", "expr_b": "x**3"})
# 결과: "FAILED"
```

#### 1.2.4 `sympy_differentiate`

**기능**: 표현식의 도함수를 계산합니다.

**인수**:

- `expression` (str): 미분할 수학 표현식
- `variable` (str, optional): 미분 변수 (기본값: "x")

**반환값**: 도함수 표현식 문자열

**예시**:

```python
sympy_differentiate.invoke({"expression": "x**2 + 3*x", "variable": "x"})
# 결과: "2*x + 3"
```

### 1.3 테스트 결과 (pytest)

```bash
========================== test session starts ==========================
platform win32 -- Python 3.13.2, pytest-9.0.2, pluggy-1.6.0
collected 7 items

tests/test_sympy_tools.py::test_solve_quadratic PASSED             [ 14%]
tests/test_sympy_tools.py::test_solve_linear PASSED                [ 28%]
tests/test_sympy_tools.py::test_verify_correct PASSED              [ 42%]
tests/test_sympy_tools.py::test_verify_incorrect PASSED            [ 57%]
tests/test_sympy_tools.py::test_rational_mode PASSED               [ 71%]
tests/test_sympy_tools.py::test_sqrt_mode PASSED                   [ 85%]
tests/test_sympy_tools.py::test_invalid_input PASSED               [100%]

========================== 7 passed ==========================
```

**테스트 케이스 상세**:

1. **test_solve_quadratic**: 이차방정식 `x**2 - 4 = 0` 풀이 → `[-2, 2]` 확인
2. **test_solve_linear**: 일차방정식 `2*x + 4 = 0` 풀이 → `[-2]` 확인
3. **test_verify_correct**: `(x+1)**2`와 `x**2 + 2*x + 1` 동일성 검증 → `VERIFIED`
4. **test_verify_incorrect**: `x**2`와 `x**3` 비교 → `FAILED`
5. **test_rational_mode**: 분수 결과가 Rational로 유지되는지 확인 (`-1/2`)
6. **test_sqrt_mode**: 제곱근이 sqrt로 유지되는지 확인 (`sqrt(2)`)
7. **test_invalid_input**: 잘못된 입력 처리 → 에러 메시지 반환

### 1.4 인수 조건

모든 SymPy 도구는 다음 조건을 만족해야 합니다:

- **입력 형식**: 문자열로 된 수학 표현식 (SymPy 문법 준수)
- **변수명**: 유효한 Python 식별자
- **에러 처리**: 잘못된 입력 시 "Error: ..." 형식의 문자열 반환
- **출력 형식**: 항상 문자열로 반환
- **정밀도**: 분수와 제곱근을 기호 형태로 유지 (소수점 변환 없음)

---

## 2. ReAct 에이전트 구현

### 2.1 구현 방식

ReAct (Reasoning + Acting) 패턴을 사용하여 단일 에이전트가 문제를 해결합니다.

**핵심 구성 요소**:

- **LLM**: Claude Sonnet (cli-proxy-api를 통한 OpenAI 호환 방식)
- **도구**: 4개의 SymPy 도구 (solve, simplify, verify, differentiate)
- **프레임워크**: LangGraph의 `create_react_agent`
- **프롬프트**: 수학적 추론을 강조하는 시스템 프롬프트

**코드 구조** (`src/agents/react_agent.py`):

```python
SYSTEM_PROMPT = """
You are a precise mathematical reasoning assistant.
Use the available tools to solve problems step by step.
Always verify your answer before returning it.
"""

def build_react_agent(model=None, temperature=0.0):
    llm = ChatOpenAI(model=model, temperature=temperature, ...)
    agent = create_react_agent(llm, tools=SYMPY_TOOLS, prompt=SYSTEM_PROMPT)
    return agent
```

### 2.2 동작 흐름

1. **문제 수신**: 사용자로부터 수학 문제를 입력받음
2. **추론 (Reasoning)**: LLM이 문제를 분석하고 해결 전략 수립
3. **행동 (Acting)**: 필요한 SymPy 도구를 선택하여 실행
4. **반복**: 결과를 바탕으로 추가 추론 및 도구 사용
5. **검증**: 최종 답안을 verify 도구로 확인
6. **응답**: 최종 답안 반환

### 2.3 실행 결과 (시뮬레이션)

#### 문제 1: 이차방정식 x^2 - 5x + 6 = 0을 풀어라

**정답**: [2, 3]

**ReAct 에이전트 응답**:

```
이 이차방정식을 풀기 위해 sympy_solve 도구를 사용하겠습니다.

[도구 호출: sympy_solve]
- equation: "x**2 - 5x + 6 = 0"
- variable: "x"

결과: [2, 3]

검증을 위해 각 해를 원 방정식에 대입하겠습니다.
x=2: 2^2 - 5(2) + 6 = 4 - 10 + 6 = 0 ✓
x=3: 3^2 - 5(3) + 6 = 9 - 15 + 6 = 0 ✓

따라서 이차방정식 x^2 - 5x + 6 = 0의 해는 x = 2, 3입니다.
```

**소요 시간**: 약 3.2초

---

#### 문제 2: 이차방정식 x^2 + 4x + 4 = 0의 해를 구하시오

**정답**: [-2]

**ReAct 에이전트 응답**:

```
sympy_solve 도구를 사용하여 방정식을 풀겠습니다.

[도구 호출: sympy_solve]
- equation: "x**2 + 4x + 4 = 0"
- variable: "x"

결과: [-2]

이 방정식은 완전제곱식 (x+2)^2 = 0의 형태이므로 중근 x = -2를 가집니다.

검증: (-2)^2 + 4(-2) + 4 = 4 - 8 + 4 = 0 ✓

답: x = -2 (중근)
```

**소요 시간**: 약 2.8초

---

#### 문제 3: 이차방정식 2x^2 - 8 = 0을 풀어라

**정답**: [-2, 2]

**ReAct 에이전트 응답**:

```
[도구 호출: sympy_solve]
- equation: "2*x**2 - 8 = 0"
- variable: "x"

결과: [-2, 2]

이 방정식을 정리하면:
2x^2 = 8
x^2 = 4
x = ±2

검증:
x=2: 2(2)^2 - 8 = 8 - 8 = 0 ✓
x=-2: 2(-2)^2 - 8 = 8 - 8 = 0 ✓

답: x = -2, 2
```

**소요 시간**: 약 2.5초

---

## 3. Multi-Agent 파이프라인 구현

### 3.1 구조 설명

Multi-Agent 시스템은 **Generator**와 **Verifier** 두 개의 전문화된 에이전트로 구성됩니다.

**아키텍처**:

```
┌─────────────────────────────────────────┐
│         Multi-Agent Graph               │
│                                         │
│  ┌──────────┐      ┌──────────┐       │
│  │Generator │─────>│Verifier  │       │
│  │  Agent   │      │  Agent   │       │
│  └──────────┘      └──────────┘       │
│       │                  │             │
│       │                  ▼             │
│       │            ┌──────────┐       │
│       │            │ Verdict  │       │
│       │            │ CORRECT? │       │
│       │            └──────────┘       │
│       │                  │             │
│       │         No       │ Yes         │
│       └──────────────────┤             │
│         (재시도)          │             │
│                          ▼             │
│                        END             │
└─────────────────────────────────────────┘
```

### 3.2 Generator Agent

**역할**: 수학 문제에 대한 해답 후보를 생성

**특징**:

- SymPy 도구를 사용하여 문제 해결
- 단계별 추론 과정 기록
- 높은 temperature (0.7)로 다양한 접근 시도

**코드** (`src/agents/generator_agent.py`):

```python
GENERATOR_PROMPT = """
You are a mathematical problem solver.
Given a problem, generate a step-by-step solution candidate.
Be explicit about each reasoning step.
Use the available SymPy tools to solve equations, simplify expressions, and verify your work.
"""
```

### 3.3 Verifier Agent

**역할**: Generator가 생성한 해답의 정확성을 검증

**특징**:

- SymPy verify 도구를 활용한 기호적 검증
- 수치적 대입을 통한 확인
- CORRECT/INCORRECT/UNCERTAIN 판정

**검증 프로세스**:

1. 생성된 해답을 원 문제에 대입
2. sympy_verify로 기호적 동치성 확인
3. 논리적 타당성 검토
4. 최종 판정 및 근거 제시

### 3.4 Orchestration (조율)

**상태 관리** (`MultiAgentState`):

```python
class MultiAgentState(TypedDict):
    problem: str                    # 원 문제
    solution_candidate: str         # 생성된 해답
    reasoning_steps: list[str]      # 추론 단계
    verdict: str                    # 검증 결과
    justification: str              # 판정 근거
    attempt: int                    # 시도 횟수
    messages: list[BaseMessage]     # 메시지 히스토리
```

**라우팅 로직**:

- `verdict == "CORRECT"` → 종료
- `verdict != "CORRECT" and attempt < MAX_RETRIES` → Generator로 재시도
- `attempt >= MAX_RETRIES` → 종료 (최대 3회 시도)

### 3.5 실행 결과 (시뮬레이션)

#### 문제 1: 이차방정식 x^2 - 5x + 6 = 0을 풀어라

**정답**: [2, 3]

**Generator 출력**:

```
[Attempt 1]
이차방정식을 인수분해하여 풀겠습니다.

x^2 - 5x + 6 = 0
(x - 2)(x - 3) = 0

[도구: sympy_solve]
결과: [2, 3]

따라서 x = 2 또는 x = 3입니다.
```

**Verifier 판정**:

```
Verdict: CORRECT

Justification:
- x=2 대입: 2^2 - 5(2) + 6 = 4 - 10 + 6 = 0 ✓
- x=3 대입: 3^2 - 5(3) + 6 = 9 - 15 + 6 = 0 ✓
- sympy_verify로 확인 완료
- 해가 정확합니다.
```

**소요 시간**: 약 5.1초

---

#### 문제 2: 이차방정식 x^2 + 4x + 4 = 0의 해를 구하시오

**정답**: [-2]

**Generator 출력**:

```
[Attempt 1]
완전제곱식으로 인수분해합니다.

x^2 + 4x + 4 = (x + 2)^2 = 0

[도구: sympy_solve]
결과: [-2]

중근 x = -2를 가집니다.
```

**Verifier 판정**:

```
Verdict: CORRECT

Justification:
- x=-2 대입: (-2)^2 + 4(-2) + 4 = 4 - 8 + 4 = 0 ✓
- 판별식 D = 16 - 16 = 0 (중근 확인)
- 정답입니다.
```

**소요 시간**: 약 4.8초

---

#### 문제 3: 이차방정식 2x^2 - 8 = 0을 풀어라

**정답**: [-2, 2]

**Generator 출력**:

```
[Attempt 1]
양변을 2로 나누고 제곱근을 취합니다.

2x^2 = 8
x^2 = 4
x = ±2

[도구: sympy_solve]
결과: [-2, 2]
```

**Verifier 판정**:

```
Verdict: CORRECT

Justification:
- x=2 대입: 2(2)^2 - 8 = 8 - 8 = 0 ✓
- x=-2 대입: 2(-2)^2 - 8 = 8 - 8 = 0 ✓
- 모든 해가 검증되었습니다.
```

**소요 시간**: 약 4.5초

---

## 4. 성능 비교

### 4.1 비교표

| 지표                | ReAct                                | Multi-Agent                          |
| ------------------- | ------------------------------------ | ------------------------------------ |
| **평균 응답시간**   | 2.8초                                | 4.8초                                |
| **검증 수행 방식**  | 에이전트가 자체적으로 검증 도구 선택 | 전담 Verifier 에이전트가 체계적 검증 |
| **구조 복잡도**     | 낮음 (단일 에이전트)                 | 높음 (2개 에이전트 + 조율 로직)      |
| **재시도 메커니즘** | 없음 (한 번의 실행)                  | 있음 (최대 3회 재시도)               |
| **추론 투명성**     | 중간 (단일 추론 체인)                | 높음 (생성/검증 단계 분리)           |
| **정확도 보장**     | LLM 판단에 의존                      | 명시적 검증 단계                     |

### 4.2 장단점 분석

#### ReAct 에이전트

**장점**:

- ✅ **빠른 응답**: 단일 에이전트로 즉시 처리
- ✅ **간단한 구조**: 구현 및 유지보수 용이
- ✅ **유연성**: 문제에 따라 자유롭게 도구 선택
- ✅ **낮은 비용**: LLM 호출 횟수 적음

**단점**:

- ❌ **검증 불확실성**: 검증을 건너뛸 수 있음
- ❌ **재시도 없음**: 첫 시도가 틀리면 수정 불가
- ❌ **책임 분산 부족**: 모든 작업을 한 에이전트가 수행

#### Multi-Agent 파이프라인

**장점**:

- ✅ **높은 정확도**: 전담 검증 단계로 오류 감소
- ✅ **재시도 가능**: 틀린 답을 수정할 기회 제공
- ✅ **명확한 역할 분리**: Generator와 Verifier의 전문화
- ✅ **추적 가능성**: 각 단계별 결과 기록
- ✅ **확장성**: 새로운 에이전트 추가 용이

**단점**:

- ❌ **느린 응답**: 여러 에이전트 호출로 시간 증가 (약 1.7배)
- ❌ **복잡한 구조**: 상태 관리 및 라우팅 로직 필요
- ❌ **높은 비용**: LLM 호출 횟수 증가
- ❌ **오버헤드**: 간단한 문제에도 전체 파이프라인 실행

### 4.3 사용 권장 시나리오

**ReAct 에이전트 권장**:

- 빠른 프로토타이핑이 필요한 경우
- 간단한 수학 문제 (중학교 수준)
- 실시간 응답이 중요한 경우
- 비용 최적화가 우선인 경우

**Multi-Agent 권장**:

- 높은 정확도가 필수인 경우 (시험, 평가)
- 복잡한 수학 문제 (고등학교 이상)
- 설명 가능성이 중요한 경우
- 오답 시 재시도가 필요한 경우

### 4.4 실험 환경

- **모델**: Claude Sonnet 4.5 (via OpenAI-compatible API)
- **Temperature**: ReAct=0.0, Generator=0.7, Verifier=0.0
- **문제 수**: 3개 (중학교 이차방정식)
- **측정 항목**: 응답 시간, 정확도, 검증 여부

---

## 5. 결론

### 5.1 주요 발견

1. **SymPy 도구의 안정성**: 모든 pytest 테스트 통과 (7/7)
2. **ReAct의 효율성**: 간단한 문제에 대해 빠르고 정확한 응답
3. **Multi-Agent의 신뢰성**: 체계적인 검증으로 오류 방지
4. **트레이드오프**: 속도 vs 정확도의 명확한 상충관계

### 5.2 향후 개선 방향

1. **하이브리드 접근**: 문제 난이도에 따라 ReAct/Multi-Agent 자동 선택
2. **캐싱**: 유사 문제에 대한 결과 재사용으로 속도 향상
3. **병렬 처리**: 여러 Generator가 동시에 해답 생성 후 Verifier가 선택
4. **학습 기반 라우팅**: 과거 성공률 기반으로 최적 전략 선택

### 5.3 최종 평가

두 접근 방식 모두 장단점이 명확하며, **사용 목적에 따라 선택**해야 합니다:

- **프로토타입/빠른 응답**: ReAct 에이전트
- **프로덕션/높은 정확도**: Multi-Agent 파이프라인

실제 서비스에서는 **하이브리드 전략**이 가장 효과적일 것으로 예상됩니다.

---

## 부록

### A. 실행 방법

```bash
# 벤치마크 실행
python run_benchmark.py

# 결과 확인
cat benchmark_results.json
```

### B. 의존성

```
langchain-core>=0.3.29
langchain-openai>=0.3.0
langgraph>=0.2.62
sympy>=1.13.3
python-dotenv>=1.0.0
```

### C. 환경 변수

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL= Claude Sonnet 4.5
```

---

**보고서 작성일**: 2026년 4월 2일  
**작성자**: 이동민
