# Project run workflow

This document describes how to run the data collection, research workflow, and evaluation handoff from a clean project environment.

## 1. Architecture

The runnable path is:

```text
Market/news collection
        |
        v
Repository-root data/processed/
        |
        v
NotebookResearchProvider
        |
        v
Planner -> Router -> Evidence context -> Hugging Face answer generation
        |
        v
ResearchState and query/answer/context handoff
        |
        v
Cohort 3 EvaluationService -> Reflection -> Optimization -> Memory
```

The existing evaluation, reflection, optimization, and memory implementation is under `src/evaluator_optimizer/`. The workflow notebooks do not duplicate that implementation; they produce the inputs it consumes.

## 2. Environment Setup

Use Python 3.11 or newer. From the repository root:

```bash
uv sync
```

Create a local environment file from the template:

```bash
cp .env.example .env
```

Fill `.env` locally. Never commit or share it:

```env
NEWS_API_KEY=your-newsapi-key
KAGGLE_API_TOKEN=your-kaggle-token
HF_TOKEN=your-huggingface-token
HF_MODEL=HuggingFaceH4/zephyr-7b-beta
STOCK_TICKER=AAPL
DATA_PERIOD=1y
COMPANY_NAME=Apple Inc.
```

`NEWS_API_KEY` is used by the market/news notebook. Yahoo Finance does not require a key. `KAGGLE_API_TOKEN` is used when Kaggle requires authentication. `HF_TOKEN` is used by the hosted Hugging Face inference client. `HF_MODEL` is configurable because provider availability varies by account and region.

If Hugging Face is unavailable or no token is configured, the workflow uses the deterministic evidence-grounded fallback. This keeps the data and handoff stages runnable without a model request.

## 3. Notebook Execution Order

Run the notebooks in this order.

### 3.1 Market and news collection

Open [01_market_news_collection.ipynb](../notebooks/01_market_news_collection.ipynb) and run every code cell from the beginning.

Responsibilities:

- Read `STOCK_TICKER` and `DATA_PERIOD`.
- Retrieve historical OHLCV data through Yahoo Finance.
- Retrieve English-language news through NewsAPI.
- Save reusable outputs under the repository root:

```text
data/processed/market_AAPL.csv
data/processed/news_AAPL.json
```

The notebook resolves the repository root before writing, so files are not saved under `notebooks/data/`.

### 3.2 Financial text preparation

Open [02_financial_text_preparation.ipynb](../notebooks/02_financial_text_preparation.ipynb) and run every code cell from the beginning.

Responsibilities:

- Install or verify `kagglehub`, `pandas`, and `scikit-learn` in the active kernel.
- Download the Financial Sentiment Analysis dataset.
- Validate `Sentence` and `Sentiment` fields.
- Clean text and encode labels.
- Create BoW and TF-IDF matrices.
- Save cleaned text under:

```text
data/processed/financial_sentiment_clean.csv
```

Restart the notebook kernel after package installation if VS Code reports that newly installed packages are unavailable.

### 3.3 Research workflow and handoff

Open [03_research_workflow_handoff.ipynb](../notebooks/03_research_workflow_handoff.ipynb) and run every code cell from the beginning.

Responsibilities:

- Load `market_<TICKER>.csv` and `news_<TICKER>.json` from `data/processed/`.
- Adapt those DataFrames through `NotebookResearchProvider`.
- Build the research graph.
- Generate an answer through Hugging Face when a usable provider is configured.
- Fall back to a deterministic evidence-grounded answer when Hugging Face cannot serve the selected model.
- Validate the downstream handoff contract.

## 4. Implementation Responsibilities

### Data acquisition and adaptation

`NotebookResearchProvider` in [src/tools/notebook_adapter.py](../src/tools/notebook_adapter.py) converts notebook outputs into the provider protocol:

```python
financial(ticker, query) -> dict[str, object]
news(ticker, query) -> list[EvidenceRecord]
filings(ticker, query) -> list[EvidenceRecord]
```

Financial summaries include ticker, company, observation count, latest close, period return, and latest volume when available.

News rows are converted into `EvidenceRecord` objects containing source, title, content, publication time, ticker, URL, and document ID.

### Shared state

`ResearchState` in [src/agent/state.py](../src/agent/state.py) stores:

- User query and ticker
- Structured research plan
- Completed tasks
- Financial data
- News and filing records
- Provenance-rich evidence
- Processed evidence
- Draft report
- Errors and metadata

`EvidenceRecord` preserves source traceability through fields such as `source`, `title`, `url`, `document_id`, and `chunk_id`.

### Planning and routing

`Planner` in [src/planning/planner.py](../src/planning/planner.py) creates tasks for financial, market, news, SEC, risk, report, and evaluation dimensions.

`RouterService` in [src/routing/router.py](../src/routing/router.py):

- Calls the provider for planned data dimensions.
- Deduplicates evidence.
- Builds the text context used by generation and evaluation.
- Exposes `route(query, context) -> str`.

`AgentGraph` in [src/agent/graph.py](../src/agent/graph.py) runs planning, retrieval, preprocessing, routing, and answer generation in order.

### Hugging Face generation

`HuggingFaceAnswerGenerator` in [src/routing/model.py](../src/routing/model.py) uses `huggingface_hub.InferenceClient.chat_completion()` with a grounding instruction. It receives only the research query and constructed evidence context.

The model is injected into `AgentGraph`, so the orchestration layer remains testable without network calls. Tests use a fake chat client.

## 5. Cohort 3 Handoff Contract

The final notebook creates these values:

```python
evaluation_query = research_state.user_query
evaluation_context = research_state.metadata["context"]
evaluation_answer = research_graph.route(
    evaluation_query,
    evaluation_context,
)

handoff = {
    "query": evaluation_query,
    "answer": evaluation_answer,
    "context": evaluation_context,
}
```

The contract is aligned with the existing Cohort 3 interfaces:

```python
ResearchRouterService.route(query: str, context: str) -> str
OptimizationController.run(
    query=query,
    answer=answer,
    context=context,
)
```

Therefore:

- `query` is the original research request.
- `answer` is the initial generated or deterministic response.
- `context` is the same evidence context supplied to the answer generator.

The shared context ensures that Cohort 3 evaluates the answer against the evidence available during generation.

## 6. Direct Integration Example

A production caller can pass the graph directly to the existing memory-aware service:

```python
from evaluator_optimizer.orchestration.memory_aware_orchestrator import (
    MemoryAwareResearchService,
)

service = MemoryAwareResearchService(
    router_service=research_graph,
    memory_service=memory_service,
    memory_builder=memory_builder,
    optimization_controller=optimization_controller,
)

trace, memory_context = service.run(
    exchange_symbol=stock_ticker,
    query=research_state.user_query,
    context=research_state.metadata["context"],
    topics=["financial performance", "market", "news", "risk"],
)
```

Cohort 3 then evaluates the initial response, reflects on failed metrics, applies bounded optimization, records an `EvaluationTrace`, and persists reusable memory.

## 7. Validation

From the repository root:

```bash
uv run --with pytest pytest -q
uv run --with ruff ruff check \
  src/agent src/planning src/tools src/chains src/routing \
  tests/test_research_workflow.py tests/test_cohort2_agent.py
```

The notebook handoff cell must print:

```text
Validated handoff fields: ['answer', 'context', 'query']
The output is compatible with ResearchRouterService and EvaluationService.
```

Do not place API keys in notebooks, source files, logs, notebook outputs, or commits.
