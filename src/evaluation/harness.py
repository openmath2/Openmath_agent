"""Evaluation harness: runs an agent over a dataset and aggregates metrics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .dataset import MathDataset, MathProblem
from .evaluators import BaseEvaluator


@dataclass
class EvalResult:
    problem_id: str
    prediction: str
    reference: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    results: list[EvalResult]

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "n": len(self.results),
            "results": [
                {"id": r.problem_id, "score": r.score, "prediction": r.prediction}
                for r in self.results
            ],
        }


class EvaluationHarness:
    """Run an agent over a dataset and compute evaluation metrics."""

    def __init__(self, agent, evaluator: BaseEvaluator):
        self.agent = agent
        self.evaluator = evaluator

    async def run_async(self, dataset: MathDataset) -> EvalSummary:
        """Evaluate the agent asynchronously over every problem.

        Args:
            dataset: MathDataset to evaluate on.

        Returns:
            EvalSummary with per-problem results and aggregate accuracy.
        """
        # TODO: implement async evaluation loop
        raise NotImplementedError

    def run(self, dataset: MathDataset) -> EvalSummary:
        """Synchronous wrapper around run_async."""
        return asyncio.run(self.run_async(dataset))
