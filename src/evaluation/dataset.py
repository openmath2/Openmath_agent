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
    # TODO: implement
    raise NotImplementedError
