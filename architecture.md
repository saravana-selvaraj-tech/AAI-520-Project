# AAI-520 Autonomous Investment Research Agent — Architecture

## 1. Architecture Overview

The AAI-520 project implements an **Autonomous Investment Research Agent** using a modular agentic architecture.

The system transforms a high-level research question into a sequence of specialized research activities and orchestrates those activities through a stateful workflow.

The core architecture follows:

```text
User Request
     │
     ▼
┌───────────────┐
│    Planner    │
│ Decomposition │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│    Router     │
│ Tool Selection│
└───────┬───────┘
        │
 ┌──────┼──────────────┐
 │      │              │
 ▼      ▼              ▼
Financial News       SEC /
Tools    Tools       Filings
 │      │              │
 └──────┼──────────────┘
        ▼
┌──────────────────────┐
│ Retrieval & Context  │
│ Construction         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    Prompt Chain      │
│ Process → Classify → │
│ Extract → Summarize  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Report Generator   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Evaluator–Optimizer  │
└──────────┬───────────┘
           │
      ┌────┴────┐
      │         │
     Pass     Improve
      │         │
      │         └──────────┐
      ▼                    │
┌───────────────┐          │
│    Memory     │◄─────────┘
└───────┬───────┘
        ▼
┌──────────────────────┐
│ Final Research Report│
└──────────────────────┘
```

---

# 2. Architectural Principles

The architecture follows several principles.

### Modularity

Each major capability is implemented as an independent component.

### Separation of Concerns

Data acquisition, retrieval, reasoning, evaluation, and presentation are separated.

### Tool Independence

The agent should not directly depend on a specific data provider where avoidable.

### Evidence-Based Generation

Generated conclusions should be grounded in retrieved information.

### Iterative Improvement

The first generated report is not necessarily treated as the final answer. An evaluator determines whether additional research or revision is required.

### Extensibility

New companies, tools, data sources, retrieval strategies, and evaluation methods should be addable without redesigning the entire system.

---

# 3. Major Components

## 3.1 User Interface

The user submits a natural-language research request.

Example:

```text
Analyze Microsoft and provide an investment research report
covering financial performance, recent news, market trends,
and major risks.
```

The request is passed to the Planner.

---

# 4. Planner

## Responsibility

The Planner converts a high-level research question into executable subtasks.

### Example

Input:

```text
Analyze NVDA.
```

Possible plan:

```text
1. Retrieve historical financial data.
2. Analyze recent financial performance.
3. Retrieve recent company-related news.
4. Analyze news sentiment and key events.
5. Retrieve relevant SEC/company filings.
6. Compare current market behavior with historical behavior.
7. Identify key risks and uncertainties.
8. Generate an evidence-supported research report.
9. Evaluate the report.
10. Perform additional research if required.
```

The Planner should produce structured tasks rather than directly executing the research.

---

# 5. Router

The Router determines which tool or research capability should handle each task.

```text
                 ┌──────────────┐
                 │ Research Task│
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Router    │
                 └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
      Financial        News          SEC
        Tool           Tool          Tool
```

### Example routing

| Task                        | Route               |
| --------------------------- | ------------------- |
| Historical stock prices     | Financial Tool      |
| Revenue / earnings          | Financial Tool      |
| Recent news                 | News Tool           |
| News sentiment              | NLP / News Pipeline |
| Annual filing               | SEC Tool            |
| Relevant document retrieval | Retriever           |
| Report quality              | Evaluator           |

The Router should remain independent of the underlying implementation of individual tools.

---

# 6. Tool Layer

The tool layer provides the agent with external capabilities.

## 6.1 Financial Data Tool

Initial implementation:

```text
yfinance
```

Potential information:

* Historical prices
* Open / high / low / close
* Trading volume
* Returns
* Financial statements
* Earnings information
* Company metadata

Example interface:

```python
get_stock_history(ticker, period)
get_company_financials(ticker)
get_market_summary(ticker)
```

---

## 6.2 News Retrieval Tool

Responsible for retrieving relevant financial and company news.

Potential workflow:

```text
Query
  ↓
Retrieve articles
  ↓
Normalize text
  ↓
Remove duplicates
  ↓
Extract metadata
  ↓
Pass to NLP pipeline
```

Important metadata includes:

* Title
* Publication date
* Source
* URL/reference
* Company/ticker
* Article content or excerpt

---

## 6.3 SEC / Regulatory Tool

The SEC component provides access to public company filings.

Potential documents include:

* 10-K
* 10-Q
* 8-K
* Company filings
* Other relevant disclosures

The system should retain source metadata so that generated claims can be traced back to evidence.

---

# 7. Retrieval Layer

The Retrieval layer connects the agent to relevant textual evidence.

```text
Documents
    │
    ▼
Preprocessing
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Index
    │
    ▼
Similarity Search
    │
    ▼
Relevant Context
```

The initial project can use a financial-news dataset such as **Financial PhraseBank** for controlled NLP experiments and public financial/filing information for research retrieval.

---

# 8. Chunking Strategy

Long documents should be divided into manageable chunks before indexing.

Conceptually:

```text
Original Document
       │
       ▼
┌───────────────────┐
│ Chunk 1           │
├───────────────────┤
│ Chunk 2           │
├───────────────────┤
│ Chunk 3           │
├───────────────────┤
│ ...               │
└───────────────────┘
```

An overlap can be introduced between consecutive chunks when contextual continuity is important.

The chunking configuration should be treated as an experimental parameter rather than a fixed assumption.

---

# 9. Prompt Chaining

Prompt chaining decomposes text processing into smaller, controlled operations.

```text
Retrieved Article
       │
       ▼
Preprocessing
       │
       ▼
Classification
       │
       ▼
Information Extraction
       │
       ▼
Summarization
       │
       ▼
Structured Evidence
```

### Example

For a financial article:

```text
Article
  ↓
Clean text
  ↓
Identify company
  ↓
Classify sentiment
  ↓
Extract financial events
  ↓
Extract risks/opportunities
  ↓
Generate concise summary
```

This approach provides more control than asking a single LLM prompt to perform every operation simultaneously.

---

# 10. Agent State

The LangGraph workflow should maintain a shared state object.

A conceptual state could contain:

```python
state = {
    "user_query": ...,
    "ticker": ...,
    "research_plan": ...,
    "completed_tasks": ...,
    "retrieved_documents": ...,
    "financial_data": ...,
    "news_data": ...,
    "filings": ...,
    "evidence": ...,
    "draft_report": ...,
    "evaluation": ...,
    "revision_count": ...,
    "memory": ...
}
```

The state allows individual nodes to contribute information without requiring tightly coupled implementations.

---

# 11. LangGraph Workflow

The orchestration layer can be represented as a graph.

```text
START
  │
  ▼
Planner
  │
  ▼
Router
  │
  ├──────────────► Financial Research
  │
  ├──────────────► News Research
  │
  └──────────────► SEC Research
                    │
                    ▼
               Retrieval
                    │
                    ▼
              Prompt Chains
                    │
                    ▼
             Report Generator
                    │
                    ▼
                Evaluator
                    │
            ┌───────┴────────┐
            │                │
          PASS            NEEDS WORK
            │                │
            ▼                ▼
          Memory        Additional Research
            │                │
            │                └──────► Retrieval
            ▼
           END
```

This workflow allows the system to move beyond a simple linear chatbot pipeline.

---

# 12. Evaluator–Optimizer

The evaluator examines the generated research report.

## Evaluation Dimensions

### Relevance

Does the report answer the original research question?

### Evidence Coverage

Are important claims supported by retrieved evidence?

### Factual Consistency

Does the report remain consistent with the available source material?

### Completeness

Were the requested research dimensions addressed?

### Source Quality

Were appropriate sources used?

### Structure

Is the report organized and understandable?

---

## Evaluation Flow

```text
Generated Report
       │
       ▼
    Evaluator
       │
       ├── Quality acceptable ──► Final Report
       │
       └── Quality insufficient
                    │
                    ▼
             Identify Gaps
                    │
                    ▼
             Additional Research
                    │
                    ▼
              Revised Report
                    │
                    └──────► Evaluator
```

A maximum iteration count should be established to prevent uncontrolled loops.

---

# 13. Memory

Memory allows the system to retain useful information across research activities.

Potential memory content:

* Previously researched companies
* Previously retrieved evidence
* Research preferences
* Prior findings
* Previous evaluation feedback
* Reusable summaries

Memory should distinguish between:

```text
Short-term workflow state
        +
Longer-term reusable research context
```

The project should avoid treating every generated response as permanent knowledge. Memory entries should have appropriate metadata and provenance.

---

# 14. Data Flow

The end-to-end data flow is:

```text
User
 │
 ▼
Research Question
 │
 ▼
Planner
 │
 ▼
Task List
 │
 ▼
Router
 │
 ├──────────────┐
 ▼              ▼
Financial      Textual
Data           Sources
 │              │
 ▼              ▼
Processing     Retrieval
 │              │
 └──────┬───────┘
        ▼
   Evidence Store
        │
        ▼
  Prompt Chaining
        │
        ▼
 Report Generation
        │
        ▼
    Evaluation
        │
        ▼
 Optimization / Reflection
        │
        ▼
     Memory
        │
        ▼
 Final Research Report
```

---

# 15. Repository-to-Architecture Mapping

| Repository Component                     | Architectural Responsibility      |
| ---------------------------------------- | --------------------------------- |
| `src/agent/`                             | Agent orchestration and state     |
| `src/planning/`                          | Task decomposition                |
| `src/routing/`                           | Dynamic routing                   |
| `src/tools/`                             | External tool interfaces          |
| `src/retrieval/`                         | Retrieval and evidence            |
| `src/chains/`                            | Prompt chaining                   |
| `src/evaluation_optimizer/`              | Evaluation and optimization       |
| `src/evaluation_optimizer/memory`        | Persistent/reusable context       |
| `src/evaluation_optimizer/reflection`    | Reflect and improve reponse       |
| `tests/`                                 | Unit and integration testing      |
| `notebooks/`                             | Experiments and demonstrations    |
| `reports/`                               | Generated/sample research reports |

---

# 16. Cohort Ownership

The architecture is intentionally divided into three major workstreams.

## Cohort 1 — Data & Retrieval

### Components

```text
src/tools/
src/retrieval/
src/visualization/
data/
```

### Responsibilities

* Financial data ingestion
* News dataset preparation
* SEC data preparation
* Text preprocessing
* Chunking
* Retrieval
* Data quality checks
* Exploratory analysis
* Visualization

### Integration Contract

Cohort 1 provides standardized outputs to the agent layer.

Example:

```python
{
    "source": "...",
    "title": "...",
    "date": "...",
    "content": "...",
    "ticker": "...",
    "metadata": {...}
}
```

---

# 17. Cohort 2 — Agent & Orchestration

### Components

```text
src/agent/
src/planning/
src/routing/
src/chains/
```

### Responsibilities

* Agent state
* Planner
* Router
* LangGraph workflow
* Tool invocation
* Prompt chains
* Error handling
* Workflow integration

### Integration Contract

Cohort 2 consumes standardized data/tool interfaces and produces structured research context for the evaluation layer.

---

# 18. Cohort 3 — Evaluation & Intelligence

### Components

```text
src/evaluation/
src/memory/
reports/
tests/
```

### Responsibilities

* Report generation
* Evaluation criteria
* Evaluator implementation
* Reflection loop
* Memory
* Quality metrics
* End-to-end testing
* Final validation

---

# 19. Cross-Cohort Integration

The three cohorts should integrate through clearly defined interfaces.

```text
             ┌──────────────────────┐
             │      Cohort 1        │
             │ Data & Retrieval     │
             └──────────┬───────────┘
                        │
                  Data Contracts
                        │
                        ▼
             ┌──────────────────────┐
             │      Cohort 2        │
             │ Agent & Orchestration│
             └──────────┬───────────┘
                        │
                 Research Output
                        │
                        ▼
             ┌──────────────────────┐
             │      Cohort 3        │
             │ Evaluation & Memory │
             └──────────────────────┘
```

Each cohort should contribute:

* Source code
* Tests
* Documentation
* Notebook/experiment evidence
* Git commits
* Integration support
* Final presentation material

---

# 20. Error Handling

The agent should explicitly handle common failure conditions.

### Missing Financial Data

```text
Tool Failure
    ↓
Retry / Alternative Query
    ↓
Return structured error
    ↓
Continue remaining research
```

### No Relevant News

The system should report that no sufficiently relevant information was retrieved rather than fabricate evidence.

### Retrieval Failure

The system should record the failure and avoid generating unsupported claims.

### LLM Failure

The orchestration layer should support controlled retries and terminate gracefully after a configured retry limit.

---

# 21. Source Provenance

Every important retrieved item should retain provenance.

Recommended metadata:

```python
{
    "source_type": "SEC",
    "source_name": "...",
    "source_url": "...",
    "publication_date": "...",
    "retrieval_date": "...",
    "ticker": "NVDA",
    "document_id": "...",
    "chunk_id": "..."
}
```

This enables the final report to connect generated findings to their supporting evidence.

---

# 22. Security and Reliability Considerations

The project is academic and should avoid treating generated output as authoritative financial advice.

Important safeguards include:

* Do not fabricate missing data.
* Preserve source metadata.
* Distinguish retrieved facts from generated interpretation.
* Identify uncertainty.
* Avoid unsupported financial claims.
* Validate numerical calculations.
* Limit autonomous iteration.
* Log tool calls and major workflow decisions where practical.

---

# 23. Testing Strategy

Testing should occur at three levels.

## Unit Testing

Examples:

```text
Planner
Router
Financial tools
Chunking
Retrieval
Evaluator
```

## Integration Testing

Validate:

```text
Planner → Router → Tools → Retrieval → Report
```

## End-to-End Testing

Example:

```text
User Query
    ↓
Complete Agent Workflow
    ↓
Final Research Report
```

The end-to-end test should verify that the system can complete the complete research cycle without manual intervention beyond the initial request.

---

# 24. Observability

The system should log major workflow events.

Example:

```text
[PLANNER] Research plan generated
[ROUTER] Financial task → FinancialTool
[ROUTER] News task → NewsTool
[RETRIEVER] 8 relevant documents retrieved
[CHAIN] Sentiment analysis completed
[REPORT] Initial report generated
[EVALUATOR] Evidence coverage below threshold
[OPTIMIZER] Additional SEC research requested
[REPORT] Report revised
[EVALUATOR] Final evaluation completed
```

This makes the agent's behavior easier to demonstrate and debug.

---

# 25. Future Extensions

The initial implementation can be extended with:

* Additional financial data providers
* More sophisticated vector databases
* Hybrid keyword + semantic retrieval
* Reranking models
* Multi-agent collaboration
* Portfolio-level analysis
* Macro-economic indicators
* More advanced financial NLP
* Automated experiment tracking
* Human-in-the-loop review
* Improved long-term memory
* Additional evaluation metrics

These extensions are outside the minimum three-week implementation unless time permits.

---

# 26. Target End State

The completed system should demonstrate the following agentic loop:

```text
        ┌─────────────────────────────┐
        │       Research Request      │
        └──────────────┬──────────────┘
                       ▼
                ┌─────────────┐
                │    PLAN     │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │    ROUTE    │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   RESEARCH  │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  SYNTHESIZE │
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   EVALUATE  │
                └──────┬──────┘
                       │
              ┌────────┴────────┐
              │                 │
           Improve             Pass
              │                 │
              ▼                 ▼
          Research          ┌─────────┐
              │             │ MEMORY  │
              └────────────►└────┬────┘
                                  ▼
                           FINAL REPORT
```

The key objective is not merely to create a chatbot that generates financial text, but to demonstrate a **structured autonomous research workflow** combining planning, tool use, retrieval, reasoning, evaluation, optimization, and memory.
