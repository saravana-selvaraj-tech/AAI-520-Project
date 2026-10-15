# AAI-520: Autonomous Investment Research Agent

The **Autonomous Investment Research Agent** is an end-to-end academic project that explores how agentic AI can gather financial information, research a publicly traded company, produce an evidence-grounded report, and improve the response through evaluation, reflection, and optimization.

The project brings together data collection, financial-text preparation, research planning and routing, answer generation, evaluation, and memory in one cohesive workflow. The notebooks provide a reproducible path from source data to a final research response and its evaluation results.

> **Disclaimer:** This project is intended for academic and experimental use. Its output is not financial advice or a recommendation to buy, sell, or hold a security. Verify material claims against reliable primary sources before relying on them.

## Project workflow at a glance

1. Collect market history and company-related news.
2. Prepare a financial-sentiment text dataset for separate text-analysis experiments.
3. Collect SEC EDGAR filing evidence for the selected ticker.
4. Run the integrated investment research agent, which loads the market, news, and filing artifacts, plans research, processes evidence, generates an initial answer, evaluates it, reflects on issues, and optimizes within configured limits.
5. Inspect the final response and evaluation/memory trace to understand both the result and the quality checks applied.

**Integration boundary:** the integrated notebook currently consumes the market CSV, news JSON, and SEC filings JSON. Notebook 02 writes a cleaned sentiment CSV and creates BoW/TF-IDF matrices in memory, but the current integrated notebook does not load those artifacts. Its prompt-chain preprocessing is performed separately by `process_evidence()` inside `AgentGraph.run()`.

The agent is designed to make the research process more traceable: a generated answer should be considered alongside the evidence available to it, evaluation findings, and any optimization steps—not in isolation.

## Technology and required libraries

Use the project's dependency definitions as the source of truth. The repository uses `pyproject.toml` and `uv.lock` to describe and lock its Python environment.

The main library groups used by the workflow include:

| Purpose | Libraries / tools |
|---|---|
| Python environment and dependencies | Python 3.11+, `uv` |
| Notebook execution | Jupyter / IPython |
| Market data collection | `yfinance`, `pandas`, `numpy` |
| News collection | `requests` and the configured NewsAPI integration |
| Financial text and data preparation | `pandas`, `numpy`, `scikit-learn` and the dataset tooling used by the notebook |
| Agent workflow and orchestration | The agent graph, planning, routing, and service components implemented in `src/` |
| Local language-model inference | `transformers`, `torch`, Hugging Face model files |
| Environment configuration | `python-dotenv` |
| Testing and code quality | `pytest`; Ruff if included in the development environment |

Install the full declared project environment rather than installing these libraries individually. This ensures that transitive dependencies and compatible versions are resolved together.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/saravana-selvaraj-tech/AAI-520-Project.git
cd AAI-520-Project
```

### 2. Install the project dependencies

Install [`uv`](https://docs.astral.sh/uv/) if it is not already available, then run:

```bash
uv sync
```

This creates or updates the project's managed Python environment using `pyproject.toml` and `uv.lock`.

If you use Jupyter through an IDE, select the Python interpreter/environment created for this project as the notebook kernel.

### 3. Create and configure `.env`

Copy the checked-in `.env.example` template to a local `.env` file in the repository root. Do not create `.env` from scratch: the template documents the configuration names expected by the current implementation.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Open `.env` and replace the placeholders with the credentials and values required for the data sources and model configuration you intend to use.

#### API keys and configuration values

The collection and research workflow may use the following settings, depending on the active code path:

| Variable | Purpose | When it is required |
|---|---|---|
| `NEWS_API_KEY` | Authenticates requests to NewsAPI for news collection | Required to collect news through NewsAPI |
| `KAGGLE_API_TOKEN` | Authenticates Kaggle dataset access where the financial-text notebook uses Kaggle | Required only if that dataset is downloaded through an authenticated Kaggle workflow |
| `HF_TOKEN` | Hugging Face access token | Usually optional for a public model; needed if the selected model or resource requires authentication |
| `HF_MODEL` | Hugging Face model identifier used for answer generation | Set to the intended generation model; the example configuration has used `Qwen/Qwen2.5-0.5B-Instruct` |
| `GROQ_API_KEY` | Authenticates evaluator-model requests through Groq | Required when Groq is selected as the evaluator provider |
| `OPENAI_API_KEY` | Authenticates evaluator-model requests through OpenAI | Required when OpenAI is selected as the evaluator provider |
| `EVALUATOR_MODEL` | Model identifier used by the LLM-as-a-Judge evaluator | Defaults to `gpt-5.4-mini`; override it when selecting another supported model, including the demonstrated Groq model `openai/gpt-oss-120b` |
| `STOCK_TICKER` | Company ticker to research | Set to the ticker you want to analyze; for example, `AAPL` |
| `DATA_PERIOD` | Historical market-data period, where consumed by the collection code | Set to a period supported by the configured data provider; for example, `1y` |
| `COMPANY_NAME` | Company name associated with the ticker, where consumed by the workflow | Set to the matching company name; for example, `Apple Inc.` |
| `SEC_USER_AGENT` | Descriptive user-agent string sent with SEC EDGAR requests | Set an application name and a real contact address before collecting filings |
| `SEC_MAX_FILINGS` | Maximum number of filing records requested by the collection adapter | Optional; defaults to `8` in the filings notebook |

**Important:** This table describes the variables known from the project workflow; it is not a substitute for the current `.env.example`. Before running the notebooks, compare your `.env` with `.env.example` and set every value marked as required by that template and the relevant notebook. Do not add variable names to `.env` merely because they appear in this README if they are not used by the checked-out code.

Keep real credentials only in your local `.env`. Never commit API keys, access tokens, passwords, or other secrets to GitHub. The `.env.example` file should contain placeholders, not real credentials.

## Centralized configuration and logging

### Configuration

The project uses environment-based configuration so that credentials and run-specific settings can be changed without editing application logic or notebook source code.

- `.env.example` documents the expected environment variables and safe example values.
- `.env` holds local values for the current machine and should remain untracked.
- Configuration is consumed by the relevant application services and notebook workflow.
- Model selection, ticker, data period, and other supported run settings should be changed through the project's configuration mechanism rather than duplicated as hard-coded values across modules.

For evaluator configuration, provide only the key for the selected provider: `GROQ_API_KEY` for Groq or `OPENAI_API_KEY` for OpenAI. The Hugging Face evaluator path may require `HF_TOKEN` depending on the selected model and access requirements. `EVALUATOR_MODEL` defaults to `gpt-5.4-mini`; set it explicitly when using a different model, such as `openai/gpt-oss-120b` for the Groq demonstration.

When changing a setting, confirm that the relevant code path actually reads it. A variable being present in `.env` does not automatically mean every notebook or service uses it.

### Logging

Logging supports tracing the research and evaluation workflow, diagnosing failures, and understanding how a run progressed. It is intended to make key workflow events observable without relying exclusively on notebook cell output.

When reviewing logs, look for:

- which workflow stage started or completed;
- data-provider, model-loading, or configuration failures;
- routing, research, evaluation, reflection, and optimization activity;
- iteration or termination decisions when emitted by the active workflow;
- exception details needed to diagnose a failed run.

Log levels determine how much detail is emitted. Use the configured log-level setting if the implementation exposes one. Avoid logging secrets, API tokens, or other sensitive values. The exact log destination, filename, format, and supported level names are determined by the checked-out logging/configuration implementation; consult the source configuration before assuming that logs are written to a particular file.

## Notebook execution order

The repository contains three preparation/collection notebooks plus the integrated research notebook. Run the relevant notebooks from the repository root, execute cells sequentially, and inspect outputs before proceeding. **The financial-text preparation notebook is a complementary text-analysis workflow; the current integrated research notebook does not directly load its cleaned CSV or in-memory vectorizers.**

### 1. Market and news collection

[`notebooks/01_market_news_collection.ipynb`](https://github.com/saravana-selvaraj-tech/AAI-520-Project/blob/main/notebooks/01_market_news_collection.ipynb)

**Purpose:** Collect market-history data and company/news information for the configured ticker and period.

**Expected handoff:** The current integration notebook looks for `data/processed/market_<TICKER>.csv` and `data/processed/news_<TICKER>.json`.

Inspect the collection status, record counts, dates, warnings, and the files actually written. A successful cell does not guarantee that every provider returned complete data.

### 2. Financial text preparation

[`notebooks/02_financial_text_preparation.ipynb`](https://github.com/saravana-selvaraj-tech/AAI-520-Project/blob/main/notebooks/02_financial_text_preparation.ipynb)

**Purpose:** Prepare the `sbhatti/financial-sentiment-analysis` dataset for reproducible financial-text experiments. The notebook downloads the dataset with `kagglehub`, validates the `Sentence` and `Sentiment` columns, removes rows with missing values and duplicate records, normalizes the three sentiment labels, and cleans the text.

**Processing and outputs:**

- Writes cleaned records to `data/processed/financial_sentiment_clean.csv`, including `clean_text` and `label_encoded` (`negative = -1`, `neutral = 0`, `positive = 1`).
- Uses a stratified 80/20 train/test split with `random_state=42`.
- Builds unigram/bigram CountVectorizer (Bag-of-Words) and TF-IDF representations, each capped at 5,000 features. The fitted vectorizers and matrices remain in notebook memory; the current notebook does not persist them.
- Installs `kagglehub`, `pandas`, and `scikit-learn` into the active kernel environment through `uv`, so `uv` must be available.

Inspect the dataset load count, cleaned record count, output path, and train/test feature shapes. Kaggle access or download problems should be resolved before treating the preparation run as successful.

**Integration boundary:** The current `AAI520_Investment_Research_Agent.ipynb` does **not** read `financial_sentiment_clean.csv` or consume the notebook's BoW/TF-IDF matrices. It uses `AgentGraph`'s `process_evidence()` path to preprocess the market/news/filing evidence retrieved for the specific research question. Therefore, this notebook currently supplies a separate prepared text dataset for analysis/experimentation, rather than a direct input to the integrated research run.

### 3. SEC EDGAR filing collection

[`notebooks/03_filings_collection.ipynb`](https://github.com/saravana-selvaraj-tech/AAI-520-Project/blob/main/notebooks/03_filings_collection.ipynb)

**Purpose:** Collect company filing evidence from SEC EDGAR for the configured ticker. The notebook uses `NotebookResearchProvider.filings()` to resolve the ticker to a CIK, read the SEC submissions index, and retrieve primary documents for recent annual, quarterly, current, and selected foreign-issuer filing types. The query can be adjusted to restrict form types such as `10-K` or `10-Q`.

**Saved artifact:** `data/processed/filings_<TICKER>.json`. Each serialized `EvidenceRecord` preserves filing/source metadata such as the SEC source, title, filing date, ticker, canonical URL, accession/document identifier, and a bounded plain-text excerpt. The JSON also records the ticker, collection timestamp, and source label.

**Expected handoff:** The integration notebook loads the `filings` array from this JSON, validates each row as an `EvidenceRecord`, and passes the resulting list to `NotebookResearchProvider` alongside market and news data.

Set `SEC_USER_AGENT` to a descriptive application name plus a real contact address, use reasonable request rates, and inspect source URLs before relying on filing content. Check the ticker/CIK mapping, filing counts, output path, and empty-result/provider warnings. The integrated research path requires this artifact in the current notebook version.

### 4. Integrated investment research agent

[`notebooks/AAI520_Investment_Research_Agent.ipynb`](https://github.com/saravana-selvaraj-tech/AAI-520-Project/blob/main/notebooks/AAI520_Investment_Research_Agent.ipynb)

**Purpose:** Run the research graph and the evaluator–optimizer workflow. The notebook loads the market, news, and SEC filing artifacts; creates a `NotebookResearchProvider`; invokes `AgentGraph`; inspects the research plan and processed evidence; then passes the initial answer, context, and the graph-owned `RouterService` into the evaluation/optimization workflow.

**What to inspect after execution:**

- the query and company/ticker used for the run;
- retrieved evidence and `processed_evidence` summaries;
- the initial generated report and final response;
- evaluation scores/rationales, identified issues, and optimization actions;
- iteration count, termination details, and memory IDs/trace information where emitted.

The current integration notebook's explicit input files are the market, news, and filing artifacts above. It does not directly consume the cleaned financial sentiment CSV from notebook 02. If that dataset is intended to influence the live research answer, a code change is needed to load it and deliberately integrate its records/features into the provider or evidence-processing path; documentation alone does not create that connection.

## How to read the final notebook result

The final notebook should be read as both a **research report** and a **record of how the report was evaluated and improved**. Use the following sequence when reviewing its output.

### 1. Read the final answer in context

Start with the research question, selected company/ticker, and the final generated response. Check whether the response addresses the requested question and whether its conclusions are supported by the retrieved context. Treat missing, stale, or conflicting source data as a limitation, not as proof that a claim is true.

### 2. Check the evidence

Review the retrieved context, cited records, source descriptions, and relevant dates when they are displayed. Ask:

- Does the evidence relate directly to the question?
- Does it support the material claims in the answer?
- Is the evidence sufficiently current for the claim being made?
- Are important risks, uncertainties, or gaps acknowledged?

A fluent answer can still be unsupported. If the notebook does not expose a source for a claim, do not assume that claim has been verified.

### 3. Interpret evaluation dimensions

The evaluation output may contain dimensions such as:

| Dimension | How to interpret it |
|---|---|
| Retrieval relevance | Whether the retrieved material is useful for the question |
| Answer relevance | Whether the answer responds to the question asked |
| Groundedness | Whether claims are supported by the supplied evidence |
| Answerability handling | Whether the answer appropriately handles missing or insufficient evidence |
| Evaluator comments / issues | The explanation of weaknesses, missing information, or required corrections |

Use the scale and criteria printed by the notebook or defined in the evaluator implementation. If a rubric uses a small ordinal scale (for example, 0–2), interpret the values according to that rubric; do not compare them directly with scores from a different scale.

### 4. Compare initial and optimized answers

Where both are shown, compare the initial response with the final response and examine the evaluation trace:

- Which issues were identified?
- Were they categorized and prioritized appropriately?
- Did optimization add relevant evidence, correct a weakness, or improve the handling of uncertainty?
- Did the relevant evaluation scores improve, remain unchanged, or decline?
- How many optimization/re-evaluation iterations occurred, and why did the workflow stop?

A higher score is useful only when the scoring rubric is meaningful and the improvement is supported by the output. A longer answer or a more confident tone is not, by itself, evidence of improvement.

### 5. Draw a balanced conclusion

A strong run should produce a relevant, evidence-supported answer and a trace that makes weaknesses and revisions understandable. If the final response remains weak on groundedness, relevance, or answerability, report that limitation rather than treating completion of the workflow as proof of correctness.

For an academic comparison, keep the research query, ticker, source context, and evaluation settings consistent between baseline and optimized runs. Record the initial and final scores, issues found, optimization steps, iteration count, and any visible changes in evidence coverage or answer quality.

## Troubleshooting

- **Missing API key:** Recheck `.env`, variable spelling, and whether the notebook expects the key in the current environment. Restart the kernel after environment changes if needed.
- **Empty news or market results:** Check the ticker, requested period, provider availability, rate limits, and credentials.
- **Dataset download/authentication error:** Check the data source and required access token; do not assume that a failed download produced a valid dataset.
- **Hugging Face model loading/download problem:** Verify the model identifier, available disk space, dependencies, and access requirements. The first model load may take time.
- **Slow CPU generation:** Small local models can still be slow on CPU. Reduce unnecessary generation length or repeated model initialization where supported by the implementation.
- **Notebook works only when run out of order:** Restart the kernel and run notebooks/cells sequentially; verify that expected intermediate artifacts were created.
- **Unexpected evaluator score:** Inspect the scoring rubric, evaluator comments, supplied context, and issue details before concluding that the answer improved or regressed.

## Tests

Run the test suite from the repository root:

```bash
uv run pytest -q
```

If Ruff is available in the project's development dependencies:

```bash
uv run ruff check .
```

Unit tests can validate individual components without exercising every external service or downloading the local model. For that reason, test success and a successful end-to-end notebook run are separate checks.

## Repository documentation

- [Architecture](architecture.md)
- [Workflow execution guide](RUN_WORKFLOW.md)
- [Project dependencies and tooling](pyproject.toml)
- [Locked dependencies](uv.lock)
- [Tests](tests/)

## License

See [`LICENSE`](LICENSE) for the repository's license terms, if present.
