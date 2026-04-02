"""Dataset loading utilities for math benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class MathProblem:
    id: str
    problem: str
    answer: str
    difficulty: str = "unknown"
    tags: list[str] = field(default_factory=list)


@dataclass
class MathDataset:
    problems: list[MathProblem]

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self) -> Iterator[MathProblem]:
        return iter(self.problems)

    def filter_by_tag(self, tag: str) -> "MathDataset":
        return MathDataset([p for p in self.problems if tag in p.tags])


def load_dataset(path: str | Path) -> MathDataset:
    """Load a JSONL dataset from disk.

    Each line must be a JSON object with at minimum "id", "problem", and "answer".

    Args:
        path: Path to a .jsonl file.

    Returns:
        MathDataset instance.
    """
    problems = []
    path = Path(path)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            problem = MathProblem(
                id=data["id"],
                problem=data["problem"],
                answer=data["answer"],
                difficulty=data.get("difficulty", "unknown"),
                tags=data.get("tags", [])
            )
            problems.append(problem)
    
    return MathDataset(problems)


def get_middle_school_benchmark() -> MathDataset:
    """Return a benchmark dataset of 10 middle school level math problems.
    
    Covers topics like:
    - Quadratic equations (이차방정식)
    - Linear functions (일차함수)
    - Discriminant (판별식)
    - Systems of equations
    - Simplification
    
    Returns:
        MathDataset with 10 problems.
    """
    problems = [
        MathProblem(
            id="ms_001",
            problem="이차방정식 x^2 - 5x + 6 = 0을 풀어라.",
            answer="[2, 3]",
            difficulty="middle_school",
            tags=["quadratic", "factoring"]
        ),
        MathProblem(
            id="ms_002",
            problem="이차방정식 x^2 + 4x + 4 = 0의 해를 구하시오.",
            answer="[-2]",
            difficulty="middle_school",
            tags=["quadratic", "perfect_square"]
        ),
        MathProblem(
            id="ms_003",
            problem="이차방정식 2x^2 - 8 = 0을 풀어라.",
            answer="[-2, 2]",
            difficulty="middle_school",
            tags=["quadratic", "difference_of_squares"]
        ),
        MathProblem(
            id="ms_004",
            problem="이차방정식 x^2 - 6x + 9 = 0의 판별식을 구하시오.",
            answer="0",
            difficulty="middle_school",
            tags=["quadratic", "discriminant"]
        ),
        MathProblem(
            id="ms_005",
            problem="일차함수 y = 2x + 3에서 x = 5일 때 y의 값을 구하시오.",
            answer="13",
            difficulty="middle_school",
            tags=["linear_function", "substitution"]
        ),
        MathProblem(
            id="ms_006",
            problem="연립방정식 x + y = 7, x - y = 3을 풀어라.",
            answer="x=5, y=2",
            difficulty="middle_school",
            tags=["system_of_equations", "linear"]
        ),
        MathProblem(
            id="ms_007",
            problem="식 (x + 3)^2을 전개하시오.",
            answer="x**2 + 6*x + 9",
            difficulty="middle_school",
            tags=["expansion", "algebra"]
        ),
        MathProblem(
            id="ms_008",
            problem="이차방정식 x^2 - 2x - 3 = 0을 풀어라.",
            answer="[-1, 3]",
            difficulty="middle_school",
            tags=["quadratic", "factoring"]
        ),
        MathProblem(
            id="ms_009",
            problem="식 x^2 - 9를 인수분해하시오.",
            answer="(x - 3)*(x + 3)",
            difficulty="middle_school",
            tags=["factoring", "difference_of_squares"]
        ),
        MathProblem(
            id="ms_010",
            problem="이차방정식 x^2 + x - 6 = 0의 두 근의 합을 구하시오.",
            answer="-1",
            difficulty="middle_school",
            tags=["quadratic", "vieta_formulas"]
        ),
    ]
    
    return MathDataset(problems)
