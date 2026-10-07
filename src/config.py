"""Central configuration for the Evaluator-Optimizer package.

Only application-level and runtime-tunable constants belong here. Domain
mappings and enum values remain close to the code that owns their semantics.
Environment variables may override selected runtime settings.
"""

from __future__ import annotations

import os
from pathlib import Path

# Application
PROJECT_NAME = "Autonomous_Investment_Research_Agent"
PACKAGE_VERSION = "0.1.0"
DEFAULT_ENCODING = "utf-8"

# Evaluation
DEFAULT_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-5-mini")
MIN_SCORE = 0.0
MAX_SCORE = 1.0
DEFAULT_PASS_THRESHOLD = 0.80
DEFAULT_SECONDARY_THRESHOLD = 0.75

METRIC_PASS_THRESHOLDS = {
    "relevance": DEFAULT_PASS_THRESHOLD,
    "groundedness": DEFAULT_PASS_THRESHOLD,
    "accuracy": DEFAULT_PASS_THRESHOLD,
    "clarity": DEFAULT_PASS_THRESHOLD,
    "citation_correctness": DEFAULT_PASS_THRESHOLD,
    "completeness": DEFAULT_SECONDARY_THRESHOLD,
    "source_quality": DEFAULT_SECONDARY_THRESHOLD,
    "temporal_validity": DEFAULT_SECONDARY_THRESHOLD,
}

# Severity mapping thresholds. Scores at or above a threshold receive the
# corresponding severity in the deterministic evaluator path.
SEVERITY_NONE_THRESHOLD = 0.80
SEVERITY_LOW_THRESHOLD = 0.65
SEVERITY_MEDIUM_THRESHOLD = 0.50
SEVERITY_HIGH_THRESHOLD = 0.30

# Evaluator-Optimizer workflow
DEFAULT_MAX_OPTIMIZATION_ITERATIONS = 2
DEFAULT_EXPOSE_EVALUATION_TRACE = False

# Memory
DEFAULT_MEMORY_CACHE_SIZE = 20
DEFAULT_MEMORY_MAX_RESULTS = 5
DEFAULT_MEMORY_MAX_AGE_DAYS = 365
DEFAULT_MEMORY_STORE_PATH = Path("data/memory/memory_store.pkl")
MEMORY_ID_HEX_LENGTH = 10
RESEARCH_TASK_ID_HEX_LENGTH = 8

# Logging
LOGGER_NAME = "evaluator_optimizer"
LOG_LEVEL = os.getenv("EVALUATOR_OPTIMIZER_LOG_LEVEL", "INFO").upper()
LOG_DIRECTORY = Path(os.getenv("EVALUATOR_OPTIMIZER_LOG_DIRECTORY", "logs"))
LOG_FILE_NAME = os.getenv(
    "EVALUATOR_OPTIMIZER_LOG_FILE",
    "evaluator_optimizer.log",
)
LOG_FILE = LOG_DIRECTORY / LOG_FILE_NAME
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"

# Logging event messages
LOG_WORKFLOW_STARTED = "Workflow started"
LOG_WORKFLOW_COMPLETED = "Workflow completed"
LOG_WORKFLOW_FAILED = "Workflow failed"
LOG_EVALUATION_STARTED = "Evaluation started"
LOG_EVALUATION_COMPLETED = "Evaluation completed"
LOG_EVALUATION_FAILED = "Evaluation failed"
LOG_REFLECTION_STARTED = "Reflection started"
LOG_REFLECTION_COMPLETED = "Reflection completed"
LOG_REFLECTION_FAILED = "Reflection failed"
LOG_OPTIMIZATION_STARTED = "Optimization started"
LOG_OPTIMIZATION_ITERATION = "Optimization iteration"
LOG_OPTIMIZATION_COMPLETED = "Optimization completed"
LOG_OPTIMIZATION_FAILED = "Optimization failed"
LOG_MEMORY_RETRIEVAL_STARTED = "Memory retrieval started"
LOG_MEMORY_RETRIEVAL_COMPLETED = "Memory retrieval completed"
LOG_MEMORY_PERSISTENCE_STARTED = "Memory persistence started"
LOG_MEMORY_PERSISTENCE_COMPLETED = "Memory persistence completed"
LOG_MEMORY_FAILED = "Memory operation failed"
