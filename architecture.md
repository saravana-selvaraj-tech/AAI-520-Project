# System Architecture --- AAI-520 Autonomous Investment Research Agent

## 1. Purpose and implementation boundary

The project integrates market/news collection, SEC filing evidence,
deterministic planning and routing, bounded prompt-chain preprocessing,
answer generation, LLM-as-a-Judge evaluation, reflection/optimization,
and pickle-backed memory. This document describes the implementation in
the supplied `src/` and notebook snapshot. It distinguishes executed
behavior from aspirational capabilities and from evidence that must be
collected at runtime.

## 2. High-level architecture

``` mermaid
flowchart TD
    A["01 Market + News Collection"] --> M[("market CSV + news JSON")]
    B["03 SEC Filing Collection"] --> F[("filings JSON")]
    C["02 Financial Text Preparation"] --> X[("clean sentiment CSV + in-memory BoW/TF-IDF")]
    M --> D["Integration Notebook"]
    F --> D
    D --> G["AgentGraph"]
    G --> P["Planner: deterministic keyword-based tasks"]
    P --> R["RouterService"]
    R --> T["Provider calls: financial / news / filings"]
    R --> Q["Local risk analysis + context construction"]
    T --> E["EvidenceRecord collection"]
    Q --> E
    E --> H["Preprocess → Classify → Extract → Summarize"]
    H --> I["Context + answer generation"]
    I --> J["Initial research answer"]
    J --> K["MemoryAwareResearchService"]
    K --> L["Five-metric EvaluationService"]
    L --> N["Reflection + prioritized actions"]
    N --> O["Bounded optimization + re-evaluation"]
    O --> Z["Final response + trace"]
    K <--> W[("Pickle memory store")]
    Z --> Y["Notebook tables, plots, per-question Markdown reports"]
```

**Important data boundary:** Notebook 02 is a complementary
financial-sentiment preparation experiment. Its cleaned CSV and
in-memory BoW/TF-IDF matrices are not loaded by the current integration
notebook. The integrated workflow consumes market CSV, news JSON, and
SEC filings JSON.

## 3. Notebook responsibilities and execution sequence

### 3.1 `notebooks/01_market_news_collection.ipynb`

-   Prompts for and validates a ticker; validates it against Yahoo
    Finance market history.
-   Resolves company name and writes `STOCK_TICKER` and `COMPANY_NAME`
    to project `.env`.
-   Collects market history and NewsAPI articles.
-   Writes `data/processed/market_<TICKER>.csv` and
    `data/processed/news_<TICKER>.json`.
-   Evidence: ticker/company, period, record counts, output paths and
    provider errors.

### 3.2 `notebooks/02_financial_text_preparation.ipynb`

-   Downloads `sbhatti/financial-sentiment-analysis` using KaggleHub.
-   Validates the `Sentence`/`Sentiment` schema, removes
    missing/duplicate records, normalizes labels and cleans text.
-   Writes `data/processed/financial_sentiment_clean.csv` with integer
    labels `-1/0/1`.
-   Uses a stratified 80/20 split (`random_state=42`) and builds BoW and
    TF-IDF unigram/bigram matrices capped at 5,000 features.
-   Evidence: loaded/cleaned row counts, labels, saved path and
    train/test matrix dimensions.

### 3.3 `notebooks/03_filings_collection.ipynb`

-   Uses `NotebookResearchProvider.filings()` to resolve a ticker to a
    CIK and retrieve recent supported SEC filing forms.
-   Writes `data/processed/filings_<TICKER>.json`, preserving
    source/title/date/ticker/URL/document metadata and bounded text
    excerpts.
-   Evidence: count, titles, dates, identifiers, URLs, content lengths,
    artifact path and empty-result warnings.
-   A descriptive `SEC_USER_AGENT` with a real contact address should be
    configured.

### 3.4 `notebooks/AAI520_Investment_Research_Agent.ipynb`

The canonical integrated notebook loads the three data artifacts,
constructs `NotebookResearchProvider`, executes `AgentGraph`, displays
the plan and actual routing trace, exposes processed-evidence stage
outputs, and hands the initial answer/context to
`MemoryAwareResearchService.run_with_initial_answer()`. It then allows
up to five follow-up questions and displays per-question
evaluation/memory traces, score comparisons, and saved final reports.

`AAI520_Investment_Research_Agent_rubric_aligned_v2.ipynb`, when present
in the repository, is an alternate rubric-focused integration
notebook---not a fifth stage in the workflow. Select one canonical
integration notebook for the final run.

## 4. Source component responsibilities

### 4.1 State and evidence contracts

`agent.state.EvidenceRecord` preserves source, title, content,
publication date, ticker, URL, document ID and chunk ID. `ResearchState`
carries the user query, ticker, planned tasks, evidence collections,
processed evidence, draft report, errors, and metadata traces.

### 4.2 Planning

`planning.planner.Planner` builds a list of `ResearchTask` objects using
deterministic keyword checks. Task types include `financial`, `market`,
`news`, `sec`, `risk`, `report`, and `evaluation`. It is an explicit
research plan, but it is not an unconstrained LLM-generated plan. The
notebook displays task type, description and status.

### 4.3 Routing and tools

`routing.router.RouterService.execute()` dispatches task types to
concrete capabilities and writes `routing_trace` entries. Fields include
task type, description, specialist, provider method, status, result
count, evidence count, result summary and error when present.

-   Financial/market tasks call provider method `financial`.
-   News tasks call provider method `news` and append returned
    `EvidenceRecord`s.
-   SEC tasks call provider method `filings` and append returned
    records.
-   Risk tasks run local keyword-based risk sentence extraction over
    evidence collected so far.
-   Report tasks build the evidence context.
-   Evaluation tasks are marked `deferred` in the research plan because
    actual evaluation happens after an answer exists.

`AgentGraph.run()` adds a separate `generation_trace` for the synthesis
specialist, route, answer generator, status and answer length. The
notebook should show this runtime trace; component names or public
callable names alone are not proof that a tool ran.

### 4.4 Prompt-chain processing

`chains.preprocessing.process_evidence()` applies four deterministic
transformations to a bounded number of evidence records:

1.  `preprocess_text` --- normalize whitespace.
2.  `classify_text` --- assign positive/negative/neutral by counting a
    small set of keywords.
3.  `extract_facts` --- select candidate sentences containing financial
    keywords or numbers.
4.  `summarize_text` --- create an extractive, length-bounded summary.

The derived evidence keeps previews to control memory usage. The
notebook displays ingestion counts, stage output counts/examples,
classification-label counts, candidate facts and source labels. The
sentiment classifier is heuristic; extracted sentences are candidates,
not independently verified facts.

### 4.5 Answer generation

The graph builds a context from available evidence and calls
`RouterService.route()`. If the configured `QwenAnswerGenerator` is
active, it uses local Hugging Face Transformers inference; its current
defaults are `Qwen/Qwen2.5-0.5B-Instruct`, `max_new_tokens=512`, and
`max_input_chars=6000`. If no generator is configured, the router
returns an evidence-context fallback. The notebook records the active
generator/fallback in `generation_trace`.

The supplied source does not expose a dedicated retrieval-ranking
component or a measured Recall@k computation with relevance ground
truth. The current evidence flow is provider retrieval followed by
record caps, deduplication, preprocessing and context construction. Do
not describe this snapshot as implementing a verified Recall@k strategy
unless that capability is added and evidenced separately.

### 4.6 Evaluation, reflection and optimization

`MemoryAwareResearchService.run_with_initial_answer()` accepts the
initial graph answer and its evidence context without regenerating that
initial answer. The same graph-owned `RouterService` is passed to
`AnswerOptimizer`, allowing suggestion-driven revisions to use the
shared routing/generation path.

`EvaluationService` runs five metric evaluators:

  -----------------------------------------------------------------------
  Metric                              Evaluation intent
  ----------------------------------- -----------------------------------
  `relevance`                         Does the answer address the
                                      question?

  `groundedness`                      Are claims supported by supplied
                                      context?

  `accuracy`                          Are facts/numbers consistent with
                                      evidence?

  `temporal_validity`                 Are dates/reporting periods
                                      appropriate to the question?

  `clarity`                           Is the answer clear, precise and
                                      organized?
  -----------------------------------------------------------------------

Each `EvaluationResult` contains a score from 0.0 to 1.0, a pass flag,
rationale and structured issues. `ReflectionService` maps evaluation
findings to recommended changes. `OptimizationController` performs a
bounded evaluate → reflect → optimize → re-evaluate loop (default
maximum: two optimization iterations). The notebook displays
initial/final evaluations, issues, actions, iteration count and a score
visualization for each question.

The integration notebook actively selects OpenAI when `OPENAI_API_KEY`
is present, otherwise Groq when `GROQ_API_KEY` is present. Its defaults
are `gpt-5.4-mini` for OpenAI and `openai/gpt-oss-120b` for Groq. The
evaluator class also contains a Hugging Face path, but the notebook's
active provider-selection branch is OpenAI/Groq. Verify
credentials/model support in the actual environment.

### 4.7 Memory and cross-run evidence

The memory layer includes `MemoryService`, `MemoryBuilder`,
`MemoryCache`, and `PickleMemoryRepository`, normally persisted at
`data/memory/memory_store.pkl`. The trace can record retrieved/applied
memory IDs. The notebook opens a fresh repository object to check
deserialization and reports store existence, size, timestamp and item
counts.

A file existing or being reloadable proves persistence, not by itself
that a later process retrieved and used earlier memory. To substantiate
cross-run learning, restart the kernel, submit a related question, and
compare retrieved IDs to IDs persisted by the earlier run. Treat
historical memory as context that must be checked against current
evidence.

## 5. Rubric evidence map in the integration notebook

  ----------------------------------------------------------------------------
  Rubric item             Observable notebook     Evidence fields/section
                          artifact                
  ----------------------- ----------------------- ----------------------------
  Prompt chaining ---     Market/news/filing      Section 7 `ingest_counts`
  ingest                  record counts           

  Prompt chaining ---     Normalized text and     Section 7
  preprocess              record counts           `stage_outputs.preprocess`

  Prompt chaining ---     Sentiment labels and    Section 7
  classify                distribution            `stage_outputs.classify`,
                                                  `sentiment_counts`

  Prompt chaining ---     Candidate facts with    Section 7
  extract                 source attribution      `extracted_preview`

  Prompt chaining ---     Per-record summaries    Section 7
  summarize               and final synthesis     `stage_outputs.summarize`,
                          length                  generation output

  Planning                Task list and status    Section 6 `plan_df`

  Routing/specialists     Per-task specialist and Section 5 `routing_trace`,
                          provider execution      `generation_trace`

  Dynamic tools/data      Provider method,        Section 5 routing trace and
                          result/evidence counts, execution logs
                          errors                  

  Evaluator--Optimizer    Initial/final score     Sections 12--13 `trace_rows`
                          tables, issues, actions 
                          and iterations          

  Self-reflection         Evaluation rationales,  Section 13
                          identified issues and   
                          recommended actions     

  Re-evaluation           Initial/final numeric   Sections 16--17
                          score plot and          
                          traceability table      

  Memory/learning         Retrieved/applied IDs   Sections 14--15
                          and disk reload check   

  Final output            Separate Markdown       Section 15, `reports/`
                          report per question     
  ----------------------------------------------------------------------------

Evidence should be considered demonstrated only when the current
execution actually emits the corresponding output. Empty data, absent
traces, missing metrics, failed provider calls or empty memory retrieval
must be reported as such.

## 6. Data flow and output artifacts

-   `data/processed/market_<TICKER>.csv`
-   `data/processed/news_<TICKER>.json`
-   `data/processed/financial_sentiment_clean.csv` (separate preparation
    workflow)
-   `data/processed/filings_<TICKER>.json`
-   `data/memory/memory_store.pkl`
-   `reports/research_report_<TICKER>_q<N>_<timestamp>.md`

The saved Markdown question reports contain the question and final
response plus run metadata; the complete evaluation and memory trace is
displayed in the notebook rather than embedded in these reports by
default.

## 7. Configuration and reproducibility

Key variables used by the notebooks/source include `STOCK_TICKER`,
`COMPANY_NAME`, `DATA_PERIOD`, `NEWS_API_KEY`, `SEC_USER_AGENT`,
`SEC_MAX_FILINGS`, `HF_MODEL`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
`EVALUATOR_MODEL`, and the `RESEARCH_MAX_*` evidence caps. Keep secrets
in local `.env` only. See `README.md` for the variable table.

For a reproducible run, start with a clean kernel, run notebook 01,
notebook 03, then the canonical integration notebook. Notebook 02 is
separate and can be run for the sentiment dataset experiment. Capture
the ticker, source counts, actual routing trace, prompt-chain stage
outputs, initial/final metric results, issue/action trace, iteration
count, memory IDs, persistence check, and report paths.

## 8. Interpretation and limitations

-   The planner is deterministic and keyword-based; task selection
    depends on query wording.
-   Classification and risk analysis are heuristic; fact extraction
    produces candidate sentences.
-   Evaluator scores are fallible rubric judgments, not probabilities of
    factual correctness.
-   Memory persistence is not equivalent to proven cross-process reuse;
    verify it by restarting and rerunning.
-   The current source snapshot does not implement a dedicated Recall@k
    calculation or evidence-ranking evaluation.
-   Notebook 02 is not connected to the integrated agent's live evidence
    path.
-   External API availability, dataset access, local model memory, and
    model/provider compatibility affect execution.
-   A passing unit test suite does not replace a successful end-to-end
    notebook run.

Update this architecture alongside changes to `src/`, notebook entry
points, evaluation metrics, configuration, or memory behavior. Document
actual runtime traces rather than intended behavior alone.
