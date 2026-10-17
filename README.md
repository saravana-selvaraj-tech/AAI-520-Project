# AAI-520: Autonomous Investment Research Agent

The **Autonomous Investment Research Agent** is an academic research
workflow that collects market, news, and SEC filing evidence; plans
research tasks; processes evidence; generates an initial report;
evaluates the report; reflects on weaknesses; performs bounded
optimization; and persists selected research memory for later runs.

> **Disclaimer:** This prototype is for academic experimentation, not
> financial advice or a buy/sell/hold recommendation. Verify material
> claims against primary sources and current data.

## Project workflow at a glance

1.  Collect market history and company-related news.
2.  Optionally prepare the separate financial-sentiment text dataset for
    text-analysis experiments.
3.  Collect SEC EDGAR filing evidence.
4.  Run the integrated research notebook: planning → provider/specialist
    routing → evidence processing → initial answer → evaluation →
    reflection/optimization/re-evaluation → memory persistence.
5.  Inspect the notebook's rubric evidence tables, evaluation trace,
    memory trace, score visualization, and saved question reports.

**Integration boundary:** The integrated notebook reads
`data/processed/market_<TICKER>.csv`, `news_<TICKER>.json`, and
`filings_<TICKER>.json`. Notebook 02 saves
`financial_sentiment_clean.csv` and creates BoW/TF-IDF matrices for a
separate experiment; the current integrated notebook does not load those
artifacts. Its live prompt-chain processing is
`chains.preprocessing.process_evidence()` inside `AgentGraph.run()`.

## Technology and dependencies

Use `pyproject.toml` and `uv.lock` as the source of truth for dependency
versions. The workflow uses Python, Jupyter/IPython, pandas, requests,
yfinance, python-dotenv, scikit-learn, KaggleHub, PyTorch/Transformers
for local Qwen generation, and the configured evaluator SDK (`openai` or
`groq`). SEC retrieval uses the SEC EDGAR public endpoints through
`NotebookResearchProvider`.

## Setup

### 1. Install the environment

``` bash
uv sync
```

Use the project's Python environment as the notebook kernel. Notebook 02
also checks that `uv` is available in the active environment because it
installs its notebook-specific dependencies.

### 2. Configure `.env`

Copy `.env.example` to `.env` in the project root and populate only the
values needed for the run. Keep `.env` untracked and never commit
secrets.

  -----------------------------------------------------------------------
  Variable                            Purpose / current behavior
  ----------------------------------- -----------------------------------
  `STOCK_TICKER`                      Selected ticker, written by
                                      notebook 01 and read by later
                                      notebooks.

  `COMPANY_NAME`                      Company name resolved from Yahoo
                                      Finance by notebook 01.

  `DATA_PERIOD`                       Yahoo Finance history period;
                                      notebook 01 defaults to `1y`.

  `NEWS_API_KEY`                      Required by notebook 01 to query
                                      NewsAPI.

  `SEC_USER_AGENT`                    Descriptive SEC request User-Agent
                                      including a real contact address.

  `SEC_MAX_FILINGS`                   Maximum filing count requested by
                                      notebook 03; defaults to `8` there.

  `HF_MODEL`                          Local answer-generation model;
                                      defaults to
                                      `Qwen/Qwen2.5-0.5B-Instruct`.

  `OPENAI_API_KEY`                    Selects the OpenAI evaluator path
                                      in the integration notebook when
                                      present.

  `GROQ_API_KEY`                      Selects the Groq evaluator path
                                      when the OpenAI key is absent.

  `EVALUATOR_MODEL`                   Evaluator model override.
                                      Integration defaults:
                                      `gpt-5.4-mini` for OpenAI and
                                      `openai/gpt-oss-120b` for Groq.

  `HF_TOKEN`                          Relevant to gated/private Hugging
                                      Face resources and the source's
                                      Hugging Face evaluator option; the
                                      current integration notebook's
                                      active evaluator selection is
                                      OpenAI/Groq.

  `RESEARCH_MAX_EVIDENCE_RECORDS`     Caps evidence records retained by
                                      the router; default `12`.

  `RESEARCH_MAX_EVIDENCE_CHARS`       Caps characters retained per
                                      evidence record; default `8000`.

  `RESEARCH_MAX_NEWS_RECORDS`         Caps news records used by the
                                      provider; default `12`.

  `RESEARCH_MAX_SEC_FILINGS`          Caps filing records used by the
                                      router; default `4`.

  `RESEARCH_MAX_RISK_FINDINGS`        Caps heuristic risk findings;
                                      default `20`.

  `ANSWER_MAX_NEW_TOKENS`             Use this as the intended
                                      generation-budget setting only if
                                      the calling code wires it through;
                                      `QwenAnswerGenerator` itself
                                      defaults to `512` and accepts a
                                      constructor argument.

  `ANSWER_MAX_INPUT_CHARS`            Use this as the intended
                                      input-limit setting only if the
                                      calling code wires it through;
                                      `QwenAnswerGenerator` itself
                                      defaults to `6000` and accepts a
                                      constructor argument.
  -----------------------------------------------------------------------

The integration notebook currently chooses OpenAI if `OPENAI_API_KEY`
exists, otherwise Groq if `GROQ_API_KEY` exists; if neither exists, it
stops with an environment error. The Hugging Face evaluator option
exists in the evaluator class, but it is not selected by the active
notebook branch as currently written. Do not assume a variable is
effective unless the executed code reads it.

## Notebook execution order

Run cells sequentially and inspect record counts, provider errors, and
saved artifacts. The integration notebook should be run from a clean
kernel for a final demonstration.

### 1. Market and news collection --- `notebooks/01_market_news_collection.ipynb`

Prompts for a ticker, validates ticker syntax, verifies that Yahoo
Finance returns market history, resolves the company name, and writes
`STOCK_TICKER` and `COMPANY_NAME` into `.env`. It saves:

-   `data/processed/market_<TICKER>.csv`
-   `data/processed/news_<TICKER>.json`

Evidence to capture: selected ticker/company, requested period, market
rows, news rows, successful output paths, and any provider warnings.
`NEWS_API_KEY` is required.

### 2. Financial text preparation --- `notebooks/02_financial_text_preparation.ipynb`

Downloads `sbhatti/financial-sentiment-analysis` with KaggleHub;
validates `Sentence` and `Sentiment`; removes missing/duplicate rows;
cleans text; encodes negative/neutral/positive as `-1/0/1`; and saves
`data/processed/financial_sentiment_clean.csv`. It creates a stratified
80/20 train/test split with `random_state=42`, then creates
CountVectorizer BoW and TF-IDF unigram/bigram matrices capped at 5,000
features.

Evidence to capture: loaded and cleaned row counts, label validity,
output path, train/test dimensions for both vectorizers. These are
text-preparation artifacts, **not** live inputs to the integrated agent
in the current version.

### 3. SEC EDGAR filings --- `notebooks/03_filings_collection.ipynb`

Uses `NotebookResearchProvider.filings()` to resolve ticker/CIK and
retrieve recent annual, quarterly, current, and selected foreign-issuer
filing forms. Saves `data/processed/filings_<TICKER>.json` with ticker,
collection timestamp, source, and serialized evidence records. Records
retain filing/source metadata, dates, identifiers, URLs, and bounded
text excerpts.

Evidence to capture: returned filing count, titles, filing dates,
identifiers, URLs, content lengths, output path, and empty-result
warnings. Set a valid `SEC_USER_AGENT` with a real contact address.

### 4. Integrated research --- `notebooks/AAI520_Investment_Research_Agent.ipynb`

Loads market/news/filing artifacts into `NotebookResearchProvider`,
executes `AgentGraph`, displays the plan and route trace, exposes the
prompt-chain stage outputs, generates an initial report, hands that
report and its context to
`MemoryAwareResearchService.run_with_initial_answer()`, and executes up
to five optional follow-up questions through the memory-aware question
workflow.

The archive may also contain
`AAI520_Investment_Research_Agent_rubric_aligned_v2.ipynb`, an alternate
rubric-focused notebook. Treat `AAI520_Investment_Research_Agent.ipynb`
as the canonical run unless the team deliberately chooses the alternate;
running both is not an additional workflow stage.

## Rubric evidence: what to capture from the notebooks

The integration notebook includes dedicated rubric-evidence sections.
The following evidence is **runtime evidence**: it should be captured
from a successful execution, not inferred from the existence of source
classes or stale saved cell outputs.

  --------------------------------------------------------------------------------------------------------
  Rubric criterion        Notebook evidence to    Where / key fields
                          capture                 
  ----------------------- ----------------------- --------------------------------------------------------
  Prompt chaining: Ingest Counts for market,      Section 7, `ingest_counts`; stage table row `1. Ingest`
                          news, and SEC filing    
                          records                 

  Prompt chaining:        Normalized text preview Section 7, `processed_evidence`;
  Preprocess              and record output       `stage_outputs.preprocess`
                          counts                  

  Prompt chaining:        Sentiment labels and    Section 7, `stage_outputs.classify`; `sentiment_counts`
  Classify                label counts            

  Prompt chaining:        Candidate               Section 7, `extracted_facts`, `extracted_preview`
  Extract                 sentences/facts with    
                          source attribution      

  Prompt chaining:        Extractive summary per  Section 7, `stage_outputs.summarize`; final
  Summarize               record and initial      synthesis/report length
                          synthesis output        

  Research planning       Planned task            Section 6, `plan_df`; task types include
                          descriptions, task      financial/market, news, SEC, risk, report, evaluation
                          types and statuses      when selected by planner logic

  Specialist routing      Selected specialist,    Section 5, `routing_trace`; `generation_trace` for
                          provider method,        synthesis route/model/status
                          status, result/evidence 
                          counts and errors       

  Dynamic tool/data use   Actual `financial`,     Section 5 routing trace; provider method and status;
                          `news`, and/or          review `Executing task type` log output
                          `filings` provider      
                          calls, local risk       
                          analysis, context       
                          construction, result    
                          counts                  

  Evaluator--Optimizer    Initial and final       Sections 12--13; `trace_rows`; initial/final evaluation
                          scores, pass flags,     tables
                          rationales, issues,     
                          optimization actions,   
                          iterations              

  Self-reflection         Identified issues and   Section 13; `identified_issues`, `optimization_actions`
                          recommended             
                          optimization actions    
                          derived from evaluation 

  Re-evaluation /         Initial-vs-final metric Section 17 visualization; traceability table in Section
  improvement             comparison and plotted  16
                          scores for each         
                          question                

  Learning across runs    Retrieved/applied       Sections 14--15; memory trace and cross-run persistence
                          memory IDs; persisted   check
                          memory file and         
                          successful reload       
                          through a fresh         
                          repository object       

  Final user-facing       One Markdown report per Section 15;
  output                  processed question,     `reports/research_report_<TICKER>_q<N>_<timestamp>.md`
                          containing question and 
                          final response          
  --------------------------------------------------------------------------------------------------------

### The five evaluator metrics

The active integration notebook configures five evaluators. Each
`EvaluationResult` has a normalized score in the range **0.0--1.0**, a
`passed` flag, a rationale, and zero or more structured issues. Inspect
both the initial and final evaluation for every question.

  -----------------------------------------------------------------------
  Metric                  What the evaluator      Notebook evidence
                          checks                  
  ----------------------- ----------------------- -----------------------
  `relevance`             Does the response       Initial/final
                          directly answer the     evaluation tables and
                          requested question and  rationale in Section 13
                          stay on scope?          

  `groundedness`          Are important claims    Initial/final score,
                          supported by the        pass flag, rationale
                          supplied context,       and issues in Section
                          without invented        13
                          unsupported claims?     

  `accuracy`              Are facts, numbers,     Initial/final score and
                          calculations and        rationale in Section 13
                          conclusions consistent  
                          with the supplied       
                          evidence?               

  `temporal_validity`     Are the evidence dates  Initial/final score and
                          and reporting periods   rationale in Section
                          appropriate for the     13; compare evidence
                          question, especially    publication dates
                          when it asks for        
                          latest/current data?    

  `clarity`               Is the answer           Initial/final score and
                          organized, precise,     rationale in Section 13
                          readable and            
                          unambiguous?            
  -----------------------------------------------------------------------

A score is an LLM-as-a-Judge rubric signal, not a probability of
correctness. Do not claim that optimization improved quality unless the
initial/final scores, rationales, issues, and answer differences support
that conclusion. If the optimizer stops without changing the answer,
record that fact rather than implying an improvement.

### What counts as evidence for each agentic function

-   **Plans:** show the actual `plan_df` rows and their task statuses.
    The current `Planner` is deterministic and keyword-based; do not
    describe it as an unconstrained LLM planner.
-   **Uses tools dynamically:** show the emitted `routing_trace` with
    specialist/provider method and results. Merely printing public
    method names or constructing a provider is not enough.
-   **Self-reflects:** show evaluator rationales, identified issues, and
    optimization actions linked to the revised answer and re-evaluation.
-   **Learns across runs:** show retrieved/applied memory IDs and the
    disk reload check. A memory file existing proves persistence, not
    that a separate process reused a previous memory. For strong
    cross-run evidence, restart the kernel and run a related question,
    then compare retrieved IDs with IDs persisted by the previous run.

### Prompt-chain interpretation and limitations

The integration code runs `preprocess_text`, `classify_text`,
`extract_facts`, and `summarize_text` over a bounded number of evidence
records. Classification is a simple keyword-count sentiment heuristic;
extraction returns **candidate sentences**, not independently verified
facts; summarization is extractive/truncation-based. These stages make
transformations inspectable, but they are not a substitute for source
validation or financial analysis APIs. Ingestion is the
market/news/filings artifact handoff; final synthesis is generated after
evidence/context construction.

## Retrieval and Recall@k status

The current uploaded `src/` and integration notebook do not show a
dedicated top-*k* retrieval-ranking stage or a measured Recall@k
calculation with relevance ground truth. Do **not** claim a measured
Recall@k result or an implemented Recall@k selection strategy from this
snapshot. If Recall@k is a graded requirement, implement and test
retrieval ranking, define relevant-item ground truth, and expose the
selected top-*k* evidence and calculation in the notebook.

## Saved artifacts

-   `data/processed/market_<TICKER>.csv` --- market history.
-   `data/processed/news_<TICKER>.json` --- news articles.
-   `data/processed/financial_sentiment_clean.csv` --- cleaned financial
    sentiment dataset (separate workflow).
-   `data/processed/filings_<TICKER>.json` --- SEC filing records.
-   `data/memory/memory_store.pkl` --- disk-backed memory repository.
-   `reports/research_report_<TICKER>_q<N>_<timestamp>.md` --- one
    Markdown report per processed question, including the question and
    final response.

The saved question reports do not include the full evaluator trace by
default; use the integration notebook's evaluation/traceability tables
to preserve that evidence.

## Tests and troubleshooting

Run the project's test suite from the repository root with the project's
configured environment (for example, `pytest` if tests are present in
the checkout). For the final academic evidence bundle, record the exact
command and pass/skip/fail summary.

-   **No market/news/filing records:** verify ticker, API key, SEC
    User-Agent, network access, date period, and artifact paths.
-   **Evaluator not configured:** set `OPENAI_API_KEY` or `GROQ_API_KEY`
    and an appropriate `EVALUATOR_MODEL` if overriding defaults.
-   **Local model memory pressure:** use the bounded default generator
    configuration and avoid creating a second model instance in the
    notebook; smaller context/token limits reduce activation/cache
    pressure but do not reduce the model's loaded-weight footprint.
-   **No memory IDs retrieved:** the first run may have no matching
    prior memory; rerun a related question after memory persistence and
    inspect the retrieved IDs.
-   **No evidence for a criterion:** report it as not observed, not as
    successful merely because the code contains a corresponding class.

## Repository documentation

-   `README.md` --- setup, notebook sequence, rubric evidence and output
    interpretation.
-   `architecture.md` --- runtime components, control/data flow and
    boundaries.
-   `AAI520_Draft_Implementation_Evidence_Report.md` --- rubric
    crosswalk and evidence protocol for the academic submission.
