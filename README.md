# Deep Research with Ollama

Local-first deep research pipeline built around Ollama, paper/web search, BibTeX resolution, and LaTeX report generation.

It asks clarifying questions, rewrites the topic into a tighter research brief, searches across multiple backends, reads large sources in chunks, resolves citations, keeps a persistent research constitution, and emits report artifacts you can inspect or edit.

## Current Status

The system is in a solid prototype state:

- Structured outputs are schema-driven per stage.
- Retrieval is now a hybrid of LLM-guided strategy plus static ranking and coverage heuristics.
- Large papers are chunked and summarized incrementally.
- The system keeps persistent research memory in `constitution.json` and `constitution.bib`.
- Broad or ambiguous topics are much better than before, but still weaker than narrow technical topics.

One important detail: the "agents" are currently role-based sequential stages, not parallel workers. The stages behave like specialized agents, but they execute in one pipeline.

## System Diagram

```mermaid
flowchart LR
    U["User / CLI"] --> C["Clarifier\nask_clarifying_questions"]
    C --> P["Planner\nbuild_plan"]
    P --> RS["Retrieval Strategy\nLLM schema + heuristic fallback"]
    RS --> R["Retriever\nmulti-round search"]
    R --> SEL["Static Selector\ncoverage + diversity + penalties"]
    SEL --> F["Fetcher\nHTML / PDF / arXiv"]
    F --> SUM["Reader\nchunk summaries + merge"]
    SUM --> CIT["Citation Resolver\nCrossref / arXiv / web BibTeX"]
    CIT --> W["Writer\nLaTeX sections + findings"]
    W --> CONS["Constitution Store\nnotes / findings / citations"]
    CONS --> OUT["Artifacts\nreport.tex\nreferences.bib\nretrieval.json\nrun.json"]

    subgraph Search Backends
        AX["arXiv"]
        SS["Semantic Scholar"]
        CR["Crossref"]
        GC["Google CSE"]
        SP["SerpAPI"]
        DDG["DuckDuckGo HTML"]
    end

    R --> AX
    R --> SS
    R --> CR
    R --> GC
    R --> SP
    R --> DDG
```

## Workflow Diagram

```mermaid
flowchart TD
    A["Start: deep-research run <topic>"] --> B["Load constitution + research program"]
    B --> C{"Clarify?"}
    C -->|yes| D["Generate clarifying questions"]
    C -->|no| E["Use provided answers"]
    D --> E
    E --> F["Build research plan\nqueries + rewritten question + must-cover"]
    F --> G["Build retrieval strategy\nanchor phrases + concept groups + generic terms"]
    G --> H["Search round 1..N"]
    H --> I["Deduplicate + promote paper landing pages"]
    I --> J["Score results\npapers, overlap, query hits, coverage, penalties"]
    J --> K["Expand queries from top results"]
    K --> L{"More rounds?"}
    L -->|yes| H
    L -->|no| M["Select diverse final sources"]
    M --> N["Fetch documents"]
    N --> O{"Large source?"}
    O -->|yes| P["Chunk text / PDF"]
    O -->|no| Q["Single chunk"]
    P --> R["Summarize chunks"]
    Q --> R
    R --> S["Resolve BibTeX"]
    S --> T["Synthesize report"]
    T --> U["Update constitution"]
    U --> V["Write report.tex / references.bib / retrieval.json / run.json"]
```

## Retrieval Loop

```mermaid
flowchart TD
    A["Seed queries from topic + plan"] --> B["Search all enabled backends"]
    B --> C["Normalize records"]
    C --> D["Promote web results into paper records\nDOI / arXiv / publisher title lookup"]
    D --> E["Deduplicate across DOI / arXiv / title"]
    E --> F["Static score"]
    F --> F1["Signals:\npaper kind\nbackend quality\nquery hits\nanchor overlap\nconcept-group coverage\ncitation count"]
    F --> F2["Penalties:\ngeneric repo/blog pages\nthin Crossref records\nweak web-only overlap"]
    F1 --> G["Ranked pool"]
    F2 --> G
    G --> H["Generate expansion queries from top-ranked evidence"]
    H --> I["Repeat until budget exhausted"]
    I --> J["Greedy final selection\ncoverage + backend diversity + paper/web mix"]
```

## What It Does

- Clarifies the task before research starts.
- Rewrites the request into a sharper research brief.
- Generates topic-specific retrieval strategy with schema-constrained LLM output and heuristic fallback.
- Searches `arXiv`, `Semantic Scholar`, `Crossref`, `Google CSE`, `SerpAPI`, or `DuckDuckGo HTML`.
- Promotes likely paper landing pages into canonical paper records when possible.
- Reads large PDFs and long documents in chunks.
- Resolves DOI-backed BibTeX via Crossref when possible.
- Falls back to arXiv or generated web BibTeX when needed.
- Stores findings, notes, and citations in a persistent constitution.
- Outputs LaTeX paragraphs plus a `.bib` file.

## Main Components

- [`cli.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/cli.py)
  Entry point and CLI commands.
- [`pipeline.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/pipeline.py)
  End-to-end orchestration: clarify, plan, retrieve, select, summarize, synthesize, write outputs.
- [`tools.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/tools.py)
  Search backends, dedupe, promotion, document fetching, chunking.
- [`ollama.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/ollama.py)
  Ollama client with schema-backed structured outputs.
- [`schemas.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/schemas.py)
  JSON Schemas for each LLM stage.
- [`citations.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/citations.py)
  BibTeX and cite-key resolution.
- [`constitution.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/constitution.py)
  Persistent findings, notes, and citation store.
- [`prompts.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/prompts.py)
  Role prompts for clarifier, planner, retrieval strategy, reader, and writer.
- [`program.py`](/Users/kunpai/Documents/Playground/deep_research_ollama/src/deep_research_ollama/program.py)
  Run-level editable `research_program.md` support.

## Artifacts

Each run writes:

- `report.tex`
  Final LaTeX report.
- `references.bib`
  BibTeX entries for selected citations.
- `retrieval.json`
  Search rounds, issued queries, ranking trace, and selected sources.
- `run.json`
  Full run snapshot including plan, selected sources, documents, citations, and synthesis payload.
- `constitution.json`
  Persistent memory of notes, findings, and citations.
- `constitution.bib`
  Persistent BibTeX memory.
- `research_program.md`
  Editable run-level instruction surface.

## Setup

1. Create a virtual environment and install the package.

```bash
cd /Users/kunpai/Documents/Playground/deep_research_ollama
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Make sure Ollama is running locally and the model exists.

3. Optional environment variables:

```bash
export OLLAMA_MODEL=gemma4:e4b
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...
export SERPAPI_API_KEY=...
export SEMANTIC_SCHOLAR_API_KEY=...
```

If neither Google CSE nor SerpAPI is configured, the pipeline falls back to DuckDuckGo HTML for web search.

## Usage

Interactive run:

```bash
deep-research run "retrieval-augmented generation for scientific assistants" \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science
```

Non-interactive run:

```bash
deep-research run "retrieval-augmented generation for scientific assistants" \
  --no-clarify \
  --answer objective="compare RAG architectures for literature assistants" \
  --answer audience="ML engineers" \
  --answer constraints="prefer surveys, benchmarks, and production systems" \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science
```

Show or modify the persistent constitution:

```bash
deep-research show-constitution \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science

deep-research delete-citation smith2024rag \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science

deep-research delete-finding finding-2 \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science
```

Initialize the editable research program file:

```bash
deep-research init-program \
  --output-dir /Users/kunpai/Documents/Playground/deep_research_ollama/output/rag_science
```

Launch the local GUI:

```bash
deep-research gui \
  --host 127.0.0.1 \
  --port 8765 \
  --output-root /Users/kunpai/Documents/Playground/deep_research_ollama/output \
  --open-browser
```

The GUI launches the existing pipeline in a background subprocess, polls `run.json` and `constitution.json` for checkpointed progress, and lets you inspect `retrieval.json`, `report.tex`, `references.bib`, `constitution.json`, and `gui_run.log` from one page.

## Structured Outputs

The pipeline uses schema-driven outputs for the LLM stages:

- clarifying questions
- research planning
- retrieval strategy generation
- source-note summaries
- final synthesis

The Ollama client sends the schema through the `format` field and validates the returned object before the pipeline accepts it.

## Design Choices

- Retrieval is not purely LLM-ranked.
  Static scoring and greedy coverage selection are used so the search behavior is inspectable and less model-fragile.
- Search is multi-backend by default.
  This helps recover niche papers that may appear in Crossref or publisher pages but not arXiv or Semantic Scholar.
- Source memory is persistent.
  The constitution lets the system accumulate robust citation state across runs instead of rebuilding everything from scratch.
- Chunked reading is mandatory for large sources.
  This keeps local models usable on long PDFs and large HTML pages.

## Known Limitations

- Role-based "agents" are sequential, not parallel workers.
- Broad topics can still admit weak blog/tutorial/web sources.
- Crossref can surface noisy matches on ambiguous topics.
- Google Scholar is not integrated directly.
- The final writer is only as good as the local model; when it fails, the system falls back to source-note compilation.

## Suggested Next Improvements

- Parallel worker agents for retrieval, reading, and verification.
- Better venue-aware ranking for broad topics.
- More aggressive canonicalization of publisher landing pages.
- Domain-specific reranking passes for bio, materials, and policy topics.
- Better evaluation harnesses using saved `retrieval.json` traces.
