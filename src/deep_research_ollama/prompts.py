from __future__ import annotations

import json

from deep_research_ollama.schemas import (
    clarifier_schema,
    planner_schema,
    relevance_critic_schema,
    retrieval_strategy_schema,
    schema_text,
    selector_schema,
    source_note_schema,
    writer_schema,
)


def _system_prompt(program: str, role_prompt: str) -> str:
    return (
        "You are part of a local deep-research system.\n"
        "Follow the research program below as the operating constitution for this run.\n\n"
        f"{program}\n\n"
        f"{role_prompt}"
    )


def clarifier_prompt(topic: str, max_questions: int, program: str) -> tuple[str, str]:
    schema = clarifier_schema(max_questions)
    system = _system_prompt(
        program,
        (
            "You are ClarifierAgent. "
            "Ask focused clarifying questions before research begins. "
            "Target scope, audience, timeframe, comparison targets, and evidence standard. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}. "
            f"Ask at most {max_questions} questions."
        ),
    )
    user = f"Topic: {topic}"
    return system, user


def planner_prompt(
    topic: str,
    answers: dict[str, str],
    constitution_snapshot: dict,
    max_queries: int,
    program: str,
) -> tuple[str, str]:
    schema = planner_schema(max_queries)
    system = _system_prompt(
        program,
        (
            "You are PlannerAgent. "
            "Rewrite the request into a precise research brief, then derive search queries, "
            "adjacent topics, focus areas, and source requirements. "
            "Prefer source requirements that improve evidence quality, not sheer volume. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}. "
            f"Return at most {max_queries} queries."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Clarifications: {json.dumps(answers, ensure_ascii=True)}\n"
        f"Constitution snapshot: {json.dumps(constitution_snapshot, ensure_ascii=True)}"
    )
    return system, user


def selector_prompt(
    topic: str,
    rewritten_question: str,
    answers: dict[str, str],
    source_requirements: list[str],
    serialized_results: list[dict],
    max_sources: int,
    program: str,
) -> tuple[str, str]:
    candidate_ids = [str(item.get("result_id", "")).strip() for item in serialized_results]
    schema = selector_schema(max_sources, candidate_ids)
    system = _system_prompt(
        program,
        (
            "You are ScoutAgent. "
            "Select the most relevant and diverse sources for deep research. "
            "Prefer primary sources, surveys, strong academic papers, and a small number of high-signal web references. "
            "Avoid picking near-duplicates when a more diverse set exists. "
            "Return valid JSON that matches this schema exactly. "
            "Use only candidate result ids from the schema enum for selected_ids. "
            f"{schema_text(schema)}. "
            f"Select at most {max_sources} source ids."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Clarifications: {json.dumps(answers, ensure_ascii=True)}\n"
        f"Source requirements: {json.dumps(source_requirements, ensure_ascii=True)}\n"
        f"Candidate results: {json.dumps(serialized_results, ensure_ascii=True)}"
    )
    return system, user


def retrieval_strategy_prompt(
    topic: str,
    rewritten_question: str,
    answers: dict[str, str],
    related_topics: list[str],
    focus_areas: list[str],
    must_cover: list[str],
    program: str,
) -> tuple[str, str]:
    schema = retrieval_strategy_schema()
    system = _system_prompt(
        program,
        (
            "You are RetrievalStrategistAgent. "
            "Analyze the topic and produce a reusable retrieval strategy. "
            "Identify high-specificity anchor phrases, orthogonal concept groups, search facets "
            "that can be combined into stronger queries, and generic terms that should be downweighted. "
            "Anchor phrases should be specific enough to retrieve substantive papers, not broad boilerplate. "
            "Generic terms should include broad evaluation or tooling words that often create noisy matches. "
            "Concept groups should represent different dimensions of the topic, such as domain, method, "
            "artifact, benchmark, mechanism, or application, depending on what the topic actually contains. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Clarifications: {json.dumps(answers, ensure_ascii=True)}\n"
        f"Related topics: {json.dumps(related_topics, ensure_ascii=True)}\n"
        f"Focus areas: {json.dumps(focus_areas, ensure_ascii=True)}\n"
        f"Must cover: {json.dumps(must_cover, ensure_ascii=True)}"
    )
    return system, user


def relevance_critic_prompt(
    topic: str,
    rewritten_question: str,
    answers: dict[str, str],
    candidates: list[dict],
    program: str,
) -> tuple[str, str]:
    candidate_ids = [str(item.get("result_id", "")).strip() for item in candidates]
    schema = relevance_critic_schema(candidate_ids)
    system = _system_prompt(
        program,
        (
            "You are CriticAgent. "
            "Evaluate whether each candidate reference is substantively relevant to the query and research goal. "
            "Use the provided query, title, and abstract or snippet. "
            "Return relevant=true only when the candidate clearly helps answer the question, not when it merely shares broad words. "
            "Return relevant=false for generic, tangential, or domain-mismatched matches. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Clarifications: {json.dumps(answers, ensure_ascii=True)}\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=True)}"
    )
    return system, user


def chunk_summary_prompt(
    topic: str,
    rewritten_question: str,
    document_title: str,
    document_url: str,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    program: str,
) -> tuple[str, str]:
    schema = source_note_schema()
    system = _system_prompt(
        program,
        (
            "You are ReaderAgent. Summarize one source chunk for later synthesis. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}. "
            "Keep claims factual and source-grounded. "
            "Evidence snippets must be short source-derived phrases, not broad paraphrases."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Document title: {document_title}\n"
        f"Document url: {document_url}\n"
        f"Chunk: {chunk_index}/{chunk_count}\n"
        f"Text:\n{chunk_text}"
    )
    return system, user


def merge_source_prompt(
    topic: str,
    rewritten_question: str,
    document_title: str,
    document_url: str,
    chunk_summaries: list[dict],
    program: str,
) -> tuple[str, str]:
    schema = source_note_schema()
    system = _system_prompt(
        program,
        (
            "You are ReaderAgent. Merge chunk summaries into one compact research note. "
            "Return valid JSON that matches this schema exactly: "
            f"{schema_text(schema)}."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Document title: {document_title}\n"
        f"Document url: {document_url}\n"
        f"Chunk summaries: {json.dumps(chunk_summaries, ensure_ascii=True)}"
    )
    return system, user


def writer_prompt(
    topic: str,
    rewritten_question: str,
    must_cover: list[str],
    source_requirements: list[str],
    answers: dict[str, str],
    constitution_snapshot: dict,
    source_notes: list[dict],
    citations: list[dict],
    program: str,
) -> tuple[str, str]:
    citation_keys = [str(item.get("cite_key", "")).strip() for item in citations]
    schema = writer_schema(citation_keys)
    system = _system_prompt(
        program,
        (
            "You are WriterAgent for a research assistant. "
            "Produce a structured synthesis using only the provided evidence. "
            "Do not invent citation keys. "
            "If evidence is weak or contradictory, say so in notes instead of overstating certainty. "
            "Return valid JSON that matches this schema exactly. "
            "Use only provided citation keys from the schema enum for paragraph and finding citation_keys. "
            "Use delete_citation_keys only for stale constitution citations that should be removed; "
            "do not list current-run citation keys there. "
            f"{schema_text(schema)}. "
            "Every paragraph must cite one or more of the provided citation keys."
        ),
    )
    user = (
        f"Topic: {topic}\n"
        f"Rewritten question: {rewritten_question}\n"
        f"Must cover: {json.dumps(must_cover, ensure_ascii=True)}\n"
        f"Source requirements: {json.dumps(source_requirements, ensure_ascii=True)}\n"
        f"Clarifications: {json.dumps(answers, ensure_ascii=True)}\n"
        f"Constitution snapshot: {json.dumps(constitution_snapshot, ensure_ascii=True)}\n"
        f"Source notes: {json.dumps(source_notes, ensure_ascii=True)}\n"
        f"Citations: {json.dumps(citations, ensure_ascii=True)}"
    )
    return system, user
