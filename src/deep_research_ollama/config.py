from __future__ import annotations

import os
from dataclasses import dataclass

from deep_research_ollama.providers import default_api_base, provider_metadata

@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    llm_api_base: str = "http://127.0.0.1:11434"
    llm_api_key: str | None = None
    request_timeout_seconds: int = 90
    max_questions: int = 4
    max_queries: int = 6
    max_web_results_per_query: int = 5
    max_paper_results_per_query: int = 4
    max_selected_sources: int = 8
    max_search_rounds: int = 3
    max_total_queries: int = 14
    max_query_batch_size: int = 4
    max_expansion_queries_per_round: int = 4
    max_ranked_results_for_expansion: int = 8
    min_papers: int = 2
    min_web_sources: int = 1
    max_sources_per_backend: int = 3
    max_critic_results: int = 0
    max_summary_model_calls: int = 18
    chunk_chars: int = 5500
    chunk_overlap_chars: int = 500
    max_chunks_per_source: int = 6
    max_source_chars: int = 45000
    user_agent: str = "deep-research-ollama/0.1"
    program_filename: str = "research_program.md"
    constitution_filename: str = "constitution.json"
    constitution_bib_filename: str = "constitution.bib"
    references_filename: str = "references.bib"
    report_filename: str = "report.tex"
    run_filename: str = "run.json"
    retrieval_filename: str = "retrieval.json"
    compile_latex: bool = True
    latex_timeout_seconds: int = 180
    google_api_key: str | None = None
    google_cse_id: str | None = None
    serpapi_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    enable_google_scholar: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        compile_latex_env = os.getenv("COMPILE_LATEX")
        llm_provider = os.getenv("LLM_PROVIDER", cls.llm_provider).strip().lower() or cls.llm_provider
        llm_model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OLLAMA_MODEL")
            or cls.llm_model
        )
        llm_api_base = (
            os.getenv("LLM_API_BASE")
            or os.getenv("OLLAMA_BASE_URL")
            or default_api_base(llm_provider)
        )
        llm_api_key = (
            os.getenv("LLM_API_KEY")
            or _provider_api_key_from_env(llm_provider)
        )
        return cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            request_timeout_seconds=int(
                os.getenv("REQUEST_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))
            ),
            max_questions=int(os.getenv("MAX_QUESTIONS", str(cls.max_questions))),
            max_queries=int(os.getenv("MAX_QUERIES", str(cls.max_queries))),
            max_web_results_per_query=int(
                os.getenv(
                    "MAX_WEB_RESULTS_PER_QUERY", str(cls.max_web_results_per_query)
                )
            ),
            max_paper_results_per_query=int(
                os.getenv(
                    "MAX_PAPER_RESULTS_PER_QUERY", str(cls.max_paper_results_per_query)
                )
            ),
            max_selected_sources=int(
                os.getenv("MAX_SELECTED_SOURCES", str(cls.max_selected_sources))
            ),
            max_search_rounds=int(
                os.getenv("MAX_SEARCH_ROUNDS", str(cls.max_search_rounds))
            ),
            max_total_queries=int(
                os.getenv("MAX_TOTAL_QUERIES", str(cls.max_total_queries))
            ),
            max_query_batch_size=int(
                os.getenv("MAX_QUERY_BATCH_SIZE", str(cls.max_query_batch_size))
            ),
            max_expansion_queries_per_round=int(
                os.getenv(
                    "MAX_EXPANSION_QUERIES_PER_ROUND",
                    str(cls.max_expansion_queries_per_round),
                )
            ),
            max_ranked_results_for_expansion=int(
                os.getenv(
                    "MAX_RANKED_RESULTS_FOR_EXPANSION",
                    str(cls.max_ranked_results_for_expansion),
                )
            ),
            min_papers=int(os.getenv("MIN_PAPERS", str(cls.min_papers))),
            min_web_sources=int(
                os.getenv("MIN_WEB_SOURCES", str(cls.min_web_sources))
            ),
            max_sources_per_backend=int(
                os.getenv(
                    "MAX_SOURCES_PER_BACKEND", str(cls.max_sources_per_backend)
                )
            ),
            max_critic_results=int(
                os.getenv("MAX_CRITIC_RESULTS", str(cls.max_critic_results))
            ),
            max_summary_model_calls=int(
                os.getenv(
                    "MAX_SUMMARY_MODEL_CALLS", str(cls.max_summary_model_calls)
                )
            ),
            chunk_chars=int(os.getenv("CHUNK_CHARS", str(cls.chunk_chars))),
            chunk_overlap_chars=int(
                os.getenv("CHUNK_OVERLAP_CHARS", str(cls.chunk_overlap_chars))
            ),
            max_chunks_per_source=int(
                os.getenv("MAX_CHUNKS_PER_SOURCE", str(cls.max_chunks_per_source))
            ),
            max_source_chars=int(
                os.getenv("MAX_SOURCE_CHARS", str(cls.max_source_chars))
            ),
            compile_latex=(
                cls.compile_latex
                if compile_latex_env is None
                else compile_latex_env.strip().lower() not in {"0", "false", "no", "off"}
            ),
            latex_timeout_seconds=int(
                os.getenv("LATEX_TIMEOUT_SECONDS", str(cls.latex_timeout_seconds))
            ),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            enable_google_scholar=os.getenv("ENABLE_GOOGLE_SCHOLAR", "1").strip().lower()
            not in {"0", "false", "no", "off"},
        )

    @property
    def ollama_base_url(self) -> str:
        return self.llm_api_base

    @property
    def ollama_model(self) -> str:
        return self.llm_model

    def resolved_model(self) -> str:
        model = self.llm_model.strip()
        provider = self.llm_provider.strip().lower() or "ollama"
        if not model:
            return ""
        prefix = f"{provider}/"
        if model.startswith(prefix):
            return model
        return f"{provider}/{model}"

    def model_display_name(self) -> str:
        return self.resolved_model() or "-"


def _provider_api_key_from_env(provider: str) -> str | None:
    for env_name in provider_metadata(provider).get("api_key_envs", []):
        value = os.getenv(str(env_name))
        if value:
            return value
    return None
