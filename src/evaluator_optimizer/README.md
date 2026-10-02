# AAI-520 Evaluator-Optimizer

Evaluator-Optimizer implements the quality and learning layer of the financial
research agent.

## Responsibilities

1. Evaluate generated research.
2. Self-reflect on detected quality issues.
3. Optimize the analysis using evaluation feedback.
4. Request additional research through Cohort 2 when required.
5. Reuse prior research through a two-level memory system.
6. Generate the final response and optional evaluation trace.

## Dependency boundary

Evaluator-Optimizer depends directly on Cohort 2 through `ResearchPackage`.

Evaluator-Optimizer does not directly depend on Cohort 1. If additional evidence
is required, Evaluator-Optimizer creates `ResearchTask` objects and returns them to
the Cohort 2 research layer.

## Package structure

```text
src/evaluator_optimizer/
├── domain/
│   ├── enums.py
│   └── models.py
├── evaluation/
│   ├── evaluator.py
│   ├── evaluation_service.py
│   ├── accuracy.py
│   ├── clarity.py
│   ├── groundedness.py
│   ├── relevance.py
│   ├── temporal_validity.py
│   ├── metrics.py
│   ├── models.py
│   ├── rubrics.py
│   └── severity.py
├── reflection/
│   ├── models.py
│   └── reflection_service.py
├── optimization/
│   ├── models.py
│   ├── optimizer.py
│   └── optimization_controller.py
├── memory/
│   ├── cache.py
│   ├── context.py
│   ├── context_enricher.py
│   ├── memory_builder.py
│   ├── memory_reuse.py
│   ├── memory_service.py
│   ├── models.py
│   ├── repository.py
│   ├── serializer.py
│   └── temporal_validity.py
└── orchestration/
    ├── evaluation_trace.py
    └── memory_aware_orchestrator.py
```

The implementation intentionally keeps domain objects independent of
LangGraph and any specific LLM provider.

## Centralized configuration and logging

Runtime-tunable constants are maintained in `src/evaluator_optimizer/config.py`.
This includes evaluator defaults and thresholds, optimization iteration limits,
memory defaults, memory storage location, and logging settings. Environment
variables can override the evaluator model, log level, and log directory/file.

Application logging is configured through `src/evaluator_optimizer/logging_config.py`.
The workflow uses component loggers under the `evaluator_optimizer` logger hierarchy
and records lifecycle events for evaluation, reflection, optimization, memory
retrieval/persistence, and final response generation.

A workflow execution receives a short `run_id` so related log entries can be traced
across stages. Logs are written to `logs/evaluator_optimizer.log` and also emitted
to the console. Full prompts, generated answers, retrieved evidence, and other
potentially sensitive research payloads are intentionally not logged by default.

Example:

```text
2026-09-30 00:10:01+0000 | INFO | evaluator_optimizer.optimization.optimization_controller |
Workflow started | run_id=7f32a91d3e21

2026-09-30 00:10:02+0000 | INFO | evaluator_optimizer.evaluation.evaluation_service |
Evaluation completed | iteration=1 | score=0.8600 | passed=True

2026-09-30 00:10:02+0000 | INFO | evaluator_optimizer.optimization.optimization_controller |
Workflow completed | run_id=7f32a91d3e21 | iterations=1 | passed=True
```

To run the complete test suite:

```bash
pytest -q
```
