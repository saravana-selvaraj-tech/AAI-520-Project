# AAI-520 Autonomous Investment Research Agent

An academic project for designing and implementing an **Autonomous Investment Research Agent** capable of planning financial research, dynamically selecting information sources and tools, synthesizing evidence, evaluating its own output, and retaining useful research context.

The project demonstrates key agentic AI concepts including:

* **Task Planning & Decomposition**
* **Dynamic Routing**
* **Tool Use**
* **Prompt Chaining**
* **Retrieval-Augmented Research**
* **Evaluator–Optimizer / Reflection**
* **Memory**
* **Evidence-based Report Generation**

---

## 1. Project Objective

Traditional financial research often requires analysts to manually gather information from multiple sources, interpret financial data, review recent news, and synthesize the findings into an investment report.

This project explores how an **agentic AI system** can automate and orchestrate these activities.

The proposed system accepts a research request such as the company or its stock symbol. Generate the concerned companies financial report. 
And then generates questions related to concerned companies financial data like the one given belowL

> "What is the latest earnings reported by Microsoft"

The agent then:

1. Understands and decomposes the research request.
2. Determines which information and tools are required.
3. Routes subtasks to appropriate research capabilities.
4. Retrieves financial, market, news, and regulatory information.
5. Processes and summarizes retrieved information.
6. Generates an evidence-based research report.
7. Take follow questions question from the user and generates answer from collected data
8. Evaluates the quality and completeness of the answer.
9. Iteratively improves the response when required.
10. Stores useful research context for subsequent tasks.

---

## 2. Key Learning Objectives

The project is designed to demonstrate practical implementation of:

| Concept              | Demonstration                                |
| -------------------- | -------------------------------------------- |
| Agentic AI           | Autonomous research workflow                 |
| Planning             | Breaking a research request into subtasks    |
| Routing              | Selecting appropriate research tools         |
| Prompt Chaining      | Sequential information-processing pipeline   |
| RAG / Retrieval      | Retrieving relevant financial/news evidence  |
| Tool Calling         | Financial and information retrieval tools    |
| Evaluation           | Checking generated reports/response                  |
| Reflection           | Improving weak or incomplete responses       |
| Memory               | Retaining useful research context            |
| Visualization        | Financial and research-related charts        |
| Software Engineering | Modular Python implementation                |
| Git/GitHub           | Collaborative version-controlled development |

---

## 3. High-Level Architecture

```text
                    ┌─────────────────────────┐
                    │     User Research       │
                    │        Question         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        Planner          │
                    │ Task Decomposition      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │         Router          │
                    │ Select Research Tools   │
                    └──────┬──────┬──────┬────┘
                           │      │      │
              ┌────────────┘      │      └─────────────┐
              ▼                   ▼                    ▼
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │ Financial    │    │ News / Text  │    │ SEC / Public │
      │ Data Tools   │    │ Retrieval    │    │ Filings      │
      └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │     Prompt Chaining     │
                    │ Retrieve → Process →    │
                    │ Classify → Extract →    │
                    │ Summarize               │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Research Report       │
                    │       Generator         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Evaluator–Optimizer      │
                    │ Quality / Evidence /     │
                    │ Completeness Checks      │
                    └────────────┬────────────┘
                                 │
                         ┌───────┴────────┐
                         │                │
                       Pass            Improve
                         │                │
                         ▼                │
                    ┌───────────┐         │
                    │  Memory   │◄────────┘
                    └─────┬─────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Final Research   │
                 │     Output       │
                 └──────────────────┘
```

A more detailed technical description is available in [`architecture.md`](architecture.md).

---

## 4. Initial Scope

The first implementation will focus on a limited set of publicly traded companies to keep the academic project manageable.

### Initial companies

* Apple — `AAPL`
* Microsoft — `MSFT`
* NVIDIA — `NVDA`
* Amazon — `AMZN`
* Alphabet — `GOOGL`

The architecture should remain extensible so that additional companies can be added without changing the core agent workflow.

---

## 5. Research Dimensions

The agent should be capable of combining multiple research dimensions.

### Financial Performance

Examples:

* Revenue
* Earnings
* Profitability
* EPS
* Growth
* Margins
* Historical stock performance

### Market Information

Examples:

* Current/historical price data
* Trading volume
* Returns
* Volatility
* Moving averages
* Comparative performance

### News and Sentiment

Examples:

* Recent financial news
* Company-related news
* Market sentiment
* Positive/negative/neutral classification
* Key events extracted from articles

### Regulatory / Company Filings

Examples:

* SEC filings
* Annual reports
* Quarterly reports
* Company disclosures

---

## 6. Technology Stack

The initial implementation is planned around the following stack.

| Component                   | Technology           |
| --------------------------- | -------------------- |
| Language                    | Python 3.11          |
| Agent orchestration         | LangGraph 1.2.12     |
| LLM / application framework | LangChain 1.4.2      |
| Market data                 | yfinance 1.7.0       |
| Regulatory information      | SEC EDGAR            |
| Data processing             | Pandas               |
| Numerical processing        | NumPy                |
| ML utilities                | scikit-learn         |
| Visualization               | Matplotlib / Seaborn |
| Development                 | Jupyter / VS Code    |
| Version control             | Git / GitHub         |
| Documentation               | Markdown             |

> Exact package versions should ultimately be captured in `requirements.txt` or `pyproject.toml` after the environment is validated.

---

## 7. Dataset Strategy

The project uses a combination of structured financial information and textual research data.

### Primary / Dynamic Sources

* Yahoo Finance through `yfinance`
* SEC EDGAR
* Publicly available financial information

### Research / NLP Dataset

The initial project can use:

* **Financial PhraseBank**
* A static financial-news dataset for reproducible experiments

### Optional Data Source

* FRED macroeconomic data

The architecture separates **data acquisition** from **agent reasoning**, allowing additional sources to be introduced later.

---

## 8. Repository Structure

```text
AAI-520-Project/
│
├── README.md
├── architecture.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_financial_analysis.ipynb
│   ├── 03_news_analysis.ipynb
│   ├── 04_rag_experiments.ipynb
│   └── 05_agent_evaluation.ipynb
│
├── src/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── nodes.py
│   │
│   ├── planning/
│   │   └── planner.py
│   │
│   ├── routing/
│   │   └── router.py
│   │
│   ├── tools/
│   │   ├── financial_tools.py
│   │   ├── news_tools.py
│   │   └── sec_tools.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── chunking.py
│   │   └── embeddings.py
│   │
│   ├── chains/
│   │   ├── preprocessing.py
│   │   ├── classification.py
│   │   ├── extraction.py
│   │   └── summarization.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── metrics.py
│   │
│   ├── memory/
│   │   └── memory.py
│   │
│   └── visualization/
│       └── charts.py
│
├── tests/
│   ├── test_planner.py
│   ├── test_router.py
│   ├── test_retrieval.py
│   ├── test_tools.py
│   └── test_evaluator.py
│
├── reports/
│   └── sample_report.md
│
└── docs/
    └── project_notes.md
```

---

## 9. Three-Cohort Development Model

The project is divided into three parallel but interconnected workstreams.

### Cohort 1 — Data & Retrieval

Primary responsibility:

* Financial data acquisition
* News/text datasets
* SEC information
* Data preprocessing
* Chunking
* Retrieval
* Exploratory analysis
* Data visualizations

### Cohort 2 — Agent & Orchestration

Primary responsibility:

* Agent state
* Planner
* Router
* Tool integration
* Prompt chains
* LangGraph workflow
* Agent execution flow

### Cohort 3 — Evaluation & Intelligence

Primary responsibility:

* Report generation
* Evaluator
* Reflection / optimization
* Memory
* Evaluation metrics
* End-to-end validation
* Final research quality assessment

The work is intentionally designed so that all three cohorts contribute to both implementation and documentation.

---

## 10. Three-Week Execution Plan

### Week 1 — Foundation

**Goal:** Establish data, architecture, interfaces, and baseline components.

| Cohort   | Key Deliverables                                          |
| -------- | --------------------------------------------------------- |
| Cohort 1 | Data sources, datasets, preprocessing pipeline            |
| Cohort 2 | Planner, state model, routing design                      |
| Cohort 3 | Evaluation criteria, report schema, baseline LLM workflow |

**Common milestone:** All teams agree on interfaces and data contracts.

---

### Week 2 — Integration

**Goal:** Connect individual components into an agentic workflow.

| Cohort   | Key Deliverables                             |
| -------- | -------------------------------------------- |
| Cohort 1 | Retrieval and financial/news tools           |
| Cohort 2 | LangGraph orchestration and prompt chains    |
| Cohort 3 | Evaluator, reflection loop, memory prototype |

**Common milestone:** End-to-end research question can execute through the agent.

---

### Week 3 — Validation & Finalization

**Goal:** Validate, evaluate, document, and demonstrate the complete system.

| Cohort   | Key Deliverables                               |
| -------- | ---------------------------------------------- |
| Cohort 1 | Data quality validation and visualizations     |
| Cohort 2 | Integration testing and workflow optimization  |
| Cohort 3 | Evaluation results, final report, presentation |

**Common milestone:** Reproducible end-to-end demonstration.

---

## 11. Expected Final Output

For a research request such as:

```text
Analyze NVDA based on recent financial performance,
market behavior, relevant news, and company filings.
```

the system should produce a structured research report containing:

1. Research question
2. Research plan
3. Sources consulted
4. Financial performance
5. Market analysis
6. News analysis
7. Key events
8. Supporting evidence
9. Risks / uncertainties
10. Summary of findings
11. Evaluation results
12. Source references

The system is intended as an **academic research assistant**, not as a personalized financial-advice system.

---

## 12. Evaluation Strategy

The final system will be evaluated across several dimensions.

### Retrieval Quality

* Relevance of retrieved documents
* Source coverage
* Retrieval completeness

### Answer Quality

* Factual consistency
* Evidence support
* Completeness
* Relevance
* Clarity

### Agent Quality

* Correct task decomposition
* Appropriate routing
* Correct tool selection
* Successful workflow execution

### Self-Evaluation

The evaluator should identify:

* Unsupported claims
* Missing evidence
* Incomplete research dimensions
* Contradictory information
* Poor-quality sources

The optimizer can then request additional research or revise the generated report.

---

## 13. Development Guidelines

### Python

Follow **PEP 8** coding conventions.

### Git

Use meaningful commits such as:

```text
feat: add financial data retrieval tool
feat: implement research planner
feat: add news retrieval pipeline
test: add router unit tests
fix: handle missing financial data
docs: update agent architecture
```

### Branching

Recommended branch structure:

```text
main
│
├── cohort-1-data
├── cohort-2-agent
└── cohort-3-evaluation
```

Integration should happen through pull requests.

---

## 14. Definition of Done

The project is considered complete when:

* [ ] Data sources are integrated
* [ ] Research request can be decomposed
* [ ] Agent can dynamically route subtasks
* [ ] Financial data can be retrieved
* [ ] News/text can be retrieved
* [ ] Relevant evidence can be retrieved
* [ ] Prompt chains process retrieved information
* [ ] Research report can be generated
* [ ] Evaluator checks generated output
* [ ] Reflection/optimization loop works
* [ ] Memory mechanism is demonstrated
* [ ] Visualizations are included
* [ ] Unit/integration tests are implemented
* [ ] Notebook demonstrates experiments
* [ ] README and architecture documentation are complete
* [ ] Final end-to-end demonstration works

---

## 15. Academic Disclaimer

This repository is developed for academic purposes as part of the AAI-520 project.

The generated research reports are intended to demonstrate **agentic AI, retrieval, reasoning, orchestration, and evaluation techniques**. They should not be interpreted as personalized investment advice or recommendations to buy, sell, or hold securities.

