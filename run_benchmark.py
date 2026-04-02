"""Benchmark script to compare ReAct and Multi-Agent approaches."""

from src.agents.react_agent import build_react_agent
from src.agents.multi_agent_graph import build_multi_agent_graph
from src.evaluation.dataset import get_middle_school_benchmark
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import time
import json

load_dotenv()

# Get first 3 problems from middle school benchmark
dataset = get_middle_school_benchmark()
sample_problems = dataset.problems[:3]

print(f"Running benchmark with {len(sample_problems)} problems...\n")

# ReAct 실험
print('=' * 60)
print('=== ReAct 에이전트 실험 ===')
print('=' * 60)
agent = build_react_agent()
react_results = []

for idx, problem in enumerate(sample_problems, 1):
    print(f"\n[문제 {idx}/{len(sample_problems)}]")
    print(f"ID: {problem.id}")
    print(f"문제: {problem.problem}")
    print(f"정답: {problem.answer}")
    
    start = time.time()
    try:
        result = agent.invoke({'messages': [HumanMessage(content=problem.problem)]})
        elapsed = time.time() - start
        response = result['messages'][-1].content
        
        react_results.append({
            'id': problem.id,
            'problem': problem.problem,
            'expected_answer': problem.answer,
            'response': response,
            'time_ms': round(elapsed * 1000),
            'time_sec': round(elapsed, 2)
        })
        
        print(f"응답: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"소요 시간: {elapsed:.2f}초 ({round(elapsed * 1000)}ms)")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        react_results.append({
            'id': problem.id,
            'problem': problem.problem,
            'expected_answer': problem.answer,
            'response': f"ERROR: {str(e)}",
            'time_ms': 0,
            'time_sec': 0
        })
    print('-' * 60)

# Multi-Agent 실험
print('\n' + '=' * 60)
print('=== Multi-Agent 실험 ===')
print('=' * 60)
graph = build_multi_agent_graph()
multi_results = []

for idx, problem in enumerate(sample_problems, 1):
    print(f"\n[문제 {idx}/{len(sample_problems)}]")
    print(f"ID: {problem.id}")
    print(f"문제: {problem.problem}")
    print(f"정답: {problem.answer}")
    
    start = time.time()
    try:
        # Initialize state properly for multi-agent graph
        initial_state = {
            'problem': problem.problem,
            'solution_candidate': '',
            'reasoning_steps': [],
            'verdict': '',
            'justification': '',
            'attempt': 0,
            'messages': [HumanMessage(content=problem.problem)]
        }
        
        result = graph.invoke(initial_state)
        elapsed = time.time() - start
        
        # Extract final solution from state
        solution = result.get('solution_candidate', '')
        verdict = result.get('verdict', 'UNKNOWN')
        justification = result.get('justification', '')
        
        response = f"Solution: {solution}\nVerdict: {verdict}\nJustification: {justification}"
        
        multi_results.append({
            'id': problem.id,
            'problem': problem.problem,
            'expected_answer': problem.answer,
            'solution': solution,
            'verdict': verdict,
            'justification': justification,
            'response': response,
            'time_ms': round(elapsed * 1000),
            'time_sec': round(elapsed, 2)
        })
        
        print(f"솔루션: {solution[:200]}{'...' if len(solution) > 200 else ''}")
        print(f"검증 결과: {verdict}")
        print(f"소요 시간: {elapsed:.2f}초 ({round(elapsed * 1000)}ms)")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        multi_results.append({
            'id': problem.id,
            'problem': problem.problem,
            'expected_answer': problem.answer,
            'solution': '',
            'verdict': 'ERROR',
            'justification': str(e),
            'response': f"ERROR: {str(e)}",
            'time_ms': 0,
            'time_sec': 0
        })
    print('-' * 60)

# 결과 요약
print('\n' + '=' * 60)
print('=== 결과 요약 ===')
print('=' * 60)

react_times = [r['time_ms'] for r in react_results if r['time_ms'] > 0]
multi_times = [r['time_ms'] for r in multi_results if r['time_ms'] > 0]

react_avg = sum(react_times) / len(react_times) if react_times else 0
multi_avg = sum(multi_times) / len(multi_times) if multi_times else 0

print(f"\nReAct 에이전트:")
print(f"  - 평균 응답 시간: {react_avg:.0f}ms ({react_avg/1000:.2f}초)")
print(f"  - 성공한 문제: {len(react_times)}/{len(sample_problems)}")

print(f"\nMulti-Agent 파이프라인:")
print(f"  - 평균 응답 시간: {multi_avg:.0f}ms ({multi_avg/1000:.2f}초)")
print(f"  - 성공한 문제: {len(multi_times)}/{len(sample_problems)}")

# Save results to JSON
results_data = {
    'react_results': react_results,
    'multi_agent_results': multi_results,
    'summary': {
        'react_avg_time_ms': react_avg,
        'multi_agent_avg_time_ms': multi_avg,
        'total_problems': len(sample_problems)
    }
}

with open('benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print(f"\n결과가 'benchmark_results.json' 파일에 저장되었습니다.")
print('=' * 60)
