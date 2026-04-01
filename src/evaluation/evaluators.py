"""Evaluators for scoring agent-generated math solutions."""

from abc import ABC, abstractmethod
import sympy as sp


class BaseEvaluator(ABC):
    @abstractmethod
    def score(self, prediction: str, reference: str) -> float:
        """Return a score in [0, 1]."""
        ...


class ExactMatchEvaluator(BaseEvaluator):
    """Returns 1.0 if prediction exactly matches the reference string."""

    def score(self, prediction: str, reference: str) -> float:
        # TODO: normalise whitespace / case before comparing
        return float(prediction.strip() == reference.strip())


class SymPyEquivalenceEvaluator(BaseEvaluator):
    """Returns 1.0 if prediction is symbolically equivalent to reference."""

    def score(self, prediction: str, reference: str) -> float:
        """Parse both expressions with SymPy and check equality.

        Returns:
            1.0 if symbolically equal, 0.0 otherwise, -1.0 if parse fails.
        """
        # TODO: implement using sp.simplify(pred - ref) == 0
        raise NotImplementedError
