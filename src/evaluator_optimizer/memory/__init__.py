"""
Memory package.

Provides L1/L2 memory, persistence, retrieval, and runtime
MemoryContext functionality.

Memory subsystem for the evaluator-optimizer.
"""

from evaluator_optimizer.memory.cache import MemoryCache
from evaluator_optimizer.memory.context import MemoryContext
from evaluator_optimizer.memory.memory_builder import MemoryBuilder
from evaluator_optimizer.memory.memory_service import MemoryService
from evaluator_optimizer.memory.models import MemoryItem
from evaluator_optimizer.memory.repository import (
    MemoryQuery,
    MemoryRepository,
    PickleMemoryRepository,
)
from evaluator_optimizer.memory.context_enricher import MemoryContextEnricher
from evaluator_optimizer.memory.memory_reuse import (
    MemoryReuseDecision,
    MemoryReusePolicy,
)
from evaluator_optimizer.memory.temporal_validity import (
    TemporalValidityChecker,
    TemporalValidityResult,
)

__all__ = [
    "MemoryBuilder",
    "MemoryCache",
    "MemoryContext",
    "MemoryItem",
    "MemoryQuery",
    "MemoryRepository",
    "MemoryService",
    "PickleMemoryRepository",
    "MemoryContextEnricher",
    "MemoryReuseDecision",
    "MemoryReusePolicy",
    "TemporalValidityChecker",
    "TemporalValidityResult",
]
