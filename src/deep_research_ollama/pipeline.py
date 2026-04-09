from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from deep_research_ollama.citations import CitationResolver
from deep_research_ollama.config import Settings
from deep_research_ollama.constitution import ConstitutionStore
from deep_research_ollama.models import (
    CitationRecord,
    CollaborationSession,
    CollaborationTurn,
    Finding,
    ReportSection,
    ResearchPlan,
    RetrievalConceptGroup,
    RetrievalStrategy,
    SearchResult,
    SourceDocument,
    SourceNote,
    SynthesisResult,
)
from deep_research_ollama.ollama import OllamaClient, OllamaError
from deep_research_ollama.program import load_research_program
from deep_research_ollama.prompts import (
    chunk_summary_prompt,
    clarifier_prompt,
    collaboration_coordinator_prompt,
    collaboration_worker_prompt,
    merge_source_prompt,
    planner_prompt,
    relevance_critic_prompt,
    retrieval_strategy_prompt,
    writer_prompt,
)
from deep_research_ollama.schemas import (
    clarifier_schema,
    collaboration_session_schema,
    collaboration_turn_schema,
    planner_schema,
    relevance_critic_schema,
    retrieval_strategy_schema,
    source_note_schema,
    writer_schema,
)
from deep_research_ollama.tools import SearchToolkit


class ResearchPipeline:
    def __init__(self, settings: Settings, output_dir: Path) -> None:
        self.settings = settings
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.program = load_research_program(self.output_dir, settings)
        self.ollama = OllamaClient(settings)
        self.search = SearchToolkit(settings)
        self.citations = CitationResolver(settings)
        self.constitution = ConstitutionStore(
            output_dir / settings.constitution_filename,
            output_dir / settings.constitution_bib_filename,
        )
        self._retrieval_strategy_cache: dict[tuple[Any, ...], RetrievalStrategy] = {}
        self._relevance_critic_cache: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}

    def run(
        self,
        topic: str,
        *,
        interactive: bool = True,
        initial_answers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        constitution = self.constitution.load(topic)
        checkpoint = self._load_resume_checkpoint(topic)
        initial_answers = initial_answers or {}

        if checkpoint:
            self.constitution.mark_resumed(str(checkpoint.get("status", "")))
            answers = self._merge_resume_answers(checkpoint.get("answers", {}), initial_answers)
            plan = self._coerce_plan_from_payload(checkpoint.get("plan"))
            retrieval = dict(checkpoint.get("retrieval", {}))
            selected = self._coerce_search_results(checkpoint.get("selected_sources", []))
            documents = self._coerce_source_documents(checkpoint.get("documents", []))
            citation_records = self._coerce_citation_records(
                checkpoint.get("citations", []),
                constitution=constitution,
                documents=documents,
            )
            note_by_source = self._coerce_source_notes(
                checkpoint.get("source_notes", []),
                constitution=constitution,
                documents=documents,
            )
            collaboration = self._coerce_collaboration_session(checkpoint.get("collaboration"))
        else:
            answers = self.ask_clarifying_questions(
                topic, prefilled_answers=initial_answers, interactive=interactive
            )
            plan = None
            retrieval = {}
            selected = []
            documents = []
            citation_records = []
            note_by_source = {}
            collaboration = CollaborationSession()

        if plan is None:
            plan = self.build_plan(topic, answers)

        if not selected:
            results, retrieval = self.retrieve_results(topic, plan, answers)
            selected = self.select_results(topic, plan, answers, results)
            retrieval["selected_sources_pre_budget"] = [item.to_dict() for item in selected]
            self._set_selected_source_debug(topic, plan, answers, retrieval, selected)
            self._write_checkpoint(
                topic=topic,
                status="selected",
                answers=answers,
                plan=plan,
                retrieval=retrieval,
                selected=selected,
            )
        elif "selected_sources" not in retrieval:
            self._set_selected_source_debug(topic, plan, answers, retrieval, selected)
        retrieval.setdefault("selected_sources_pre_budget", [item.to_dict() for item in selected])

        if not documents:
            documents = [self.search.fetch_document(result) for result in selected]

        if documents and (
            "budget" not in retrieval or self._budget_override_requires_reapply(retrieval)
        ):
            if self._budget_override_requires_reapply(retrieval):
                candidate_results = self._coerce_search_results(
                    retrieval.get("selected_sources_pre_budget", [])
                )
                if candidate_results:
                    selected = candidate_results
                    documents = [self.search.fetch_document(result) for result in selected]
            selected, documents, budget = self._apply_summary_budget(selected, documents)
            retrieval["budget"] = budget
            self._set_selected_source_debug(topic, plan, answers, retrieval, selected)
            allowed_source_ids = {document.source_id for document in documents}
            citation_records = [
                citation
                for citation in citation_records
                if citation.source_id in allowed_source_ids
            ]
            note_by_source = {
                source_id: note
                for source_id, note in note_by_source.items()
                if source_id in allowed_source_ids
            }
            self._write_checkpoint(
                topic=topic,
                status="budgeted",
                answers=answers,
                plan=plan,
                retrieval=retrieval,
                selected=selected,
                documents=documents,
                citations=citation_records,
                note_by_source=note_by_source,
            )

        existing_citation_source_ids = {citation.source_id for citation in citation_records}
        missing_citation_documents = [
            document
            for document in documents
            if document.source_id not in existing_citation_source_ids
        ]

        if missing_citation_documents:
            citation_records.extend(self.resolve_citations(missing_citation_documents, constitution))
            self.constitution.checkpoint_sources(
                [],
                citation_records,
                checkpoint_stage="cited",
            )
            self._write_checkpoint(
                topic=topic,
                status="cited",
                answers=answers,
                plan=plan,
                retrieval=retrieval,
                selected=selected,
                documents=documents,
                citations=citation_records,
            )
        else:
            self.constitution.checkpoint_sources(
                [],
                citation_records,
                checkpoint_stage="cited",
            )

        if len(note_by_source) < len(documents):
            note_by_source = self.summarize_documents(
                plan,
                documents,
                citation_records,
                existing_notes=note_by_source,
                on_note=lambda note, notes: self._checkpoint_note_progress(
                    topic=topic,
                    answers=answers,
                    plan=plan,
                    retrieval=retrieval,
                    selected=selected,
                    documents=documents,
                    citations=citation_records,
                    note=note,
                    note_by_source=notes,
                ),
            )
        if not collaboration.turns:
            collaboration = self.collaborate(
                topic,
                plan,
                answers,
                note_by_source,
                citation_records,
            )
            self._write_checkpoint(
                topic=topic,
                status="collaborated",
                answers=answers,
                plan=plan,
                retrieval=retrieval,
                selected=selected,
                documents=documents,
                citations=citation_records,
                note_by_source=note_by_source,
                collaboration=collaboration,
                progress={
                    "collaboration_turns": len(collaboration.turns),
                    "consensus_claims": len(collaboration.consensus_claims),
                },
            )
        synthesis = self.synthesize(
            topic,
            plan,
            answers,
            note_by_source,
            citation_records,
            collaboration,
        )
        self.constitution.apply_run(list(note_by_source.values()), citation_records, synthesis)
        paths = self.write_outputs(
            topic=topic,
            answers=answers,
            plan=plan,
            selected=selected,
            documents=documents,
            citations=citation_records,
            source_notes=list(note_by_source.values()),
            synthesis=synthesis,
            retrieval=retrieval,
            collaboration=collaboration,
        )
        return {
            "provider": self.settings.llm_provider,
            "model": self.settings.model_display_name(),
            "answers": answers,
            "plan": plan.to_dict(),
            "selected_sources": [result.to_dict() for result in selected],
            "retrieval": retrieval,
            "collaboration": collaboration.to_dict(),
            "paths": {name: str(path) for name, path in paths.items()},
        }

    def _load_resume_checkpoint(self, topic: str) -> dict[str, Any] | None:
        run_path = self.output_dir / self.settings.run_filename
        if not run_path.exists():
            return None
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(payload.get("topic", "")).strip() != topic.strip():
            return None
        if str(payload.get("status", "")).strip() == "completed":
            return None
        return payload

    def _budget_override_requires_reapply(self, retrieval: dict[str, Any]) -> bool:
        budget = retrieval.get("budget", {})
        if not isinstance(budget, dict):
            return False
        configured = max(1, int(self.settings.max_summary_model_calls))
        recorded = budget.get("max_summary_model_calls")
        try:
            recorded_value = max(1, int(recorded))
        except (TypeError, ValueError):
            return False
        return recorded_value != configured

    @staticmethod
    def _merge_resume_answers(
        checkpoint_answers: dict[str, Any],
        override_answers: dict[str, str],
    ) -> dict[str, str]:
        if not isinstance(checkpoint_answers, dict):
            checkpoint_answers = {}
        merged = {
            str(key).strip(): str(value).strip()
            for key, value in checkpoint_answers.items()
            if str(key).strip() and str(value).strip()
        }
        for key, value in override_answers.items():
            if key and value:
                merged[str(key).strip()] = str(value).strip()
        return merged

    def _coerce_plan_from_payload(self, payload: Any) -> ResearchPlan | None:
        if not isinstance(payload, dict):
            return None
        try:
            return ResearchPlan(
                queries=[str(item).strip() for item in payload.get("queries", []) if str(item).strip()],
                related_topics=[
                    str(item).strip()
                    for item in payload.get("related_topics", [])
                    if str(item).strip()
                ],
                focus_areas=[
                    str(item).strip()
                    for item in payload.get("focus_areas", [])
                    if str(item).strip()
                ],
                rewritten_question=str(payload.get("rewritten_question", "")).strip(),
                must_cover=[
                    str(item).strip() for item in payload.get("must_cover", []) if str(item).strip()
                ],
                source_requirements=[
                    str(item).strip()
                    for item in payload.get("source_requirements", [])
                    if str(item).strip()
                ],
            )
        except (AttributeError, TypeError):
            return None

    def _coerce_search_results(self, items: list[dict[str, Any]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in items:
            payload = self._strip_record_meta(item)
            try:
                results.append(SearchResult(**payload))
            except TypeError:
                continue
        return results

    def _coerce_source_documents(self, items: list[dict[str, Any]]) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for item in items:
            payload = self._strip_record_meta(item)
            try:
                documents.append(SourceDocument(**payload))
            except TypeError:
                continue
        return documents

    def _coerce_citation_records(
        self,
        items: list[dict[str, Any]],
        *,
        constitution: dict[str, Any],
        documents: list[SourceDocument],
    ) -> list[CitationRecord]:
        citations: list[CitationRecord] = []
        for item in items:
            payload = self._strip_record_meta(item)
            try:
                citations.append(CitationRecord(**payload))
            except TypeError:
                continue
        if citations:
            return citations
        if not documents:
            return []

        document_ids = {document.source_id for document in documents}
        for item in constitution.get("citations", {}).values():
            payload = self._strip_record_meta(item)
            if document_ids and payload.get("source_id") not in document_ids:
                continue
            try:
                citations.append(CitationRecord(**payload))
            except TypeError:
                continue
        return citations

    def _coerce_source_notes(
        self,
        items: list[dict[str, Any]],
        *,
        constitution: dict[str, Any],
        documents: list[SourceDocument],
    ) -> dict[str, SourceNote]:
        notes: dict[str, SourceNote] = {}
        for item in items:
            payload = self._strip_record_meta(item)
            try:
                note = SourceNote(**payload)
            except TypeError:
                continue
            notes[note.source_id] = note
        if notes:
            return notes
        if not documents:
            return {}

        document_ids = {document.source_id for document in documents}
        for item in constitution.get("source_notes", {}).values():
            payload = self._strip_record_meta(item)
            if document_ids and payload.get("source_id") not in document_ids:
                continue
            try:
                note = SourceNote(**payload)
            except TypeError:
                continue
            notes[note.source_id] = note
        return notes

    def _coerce_collaboration_session(self, payload: Any) -> CollaborationSession:
        if not isinstance(payload, dict):
            return CollaborationSession()

        turns: list[CollaborationTurn] = []
        for item in payload.get("turns", []):
            if not isinstance(item, dict):
                continue
            turns.append(
                CollaborationTurn(
                    role=str(item.get("role", "")).strip(),
                    summary=str(item.get("summary", "")).strip(),
                    claims=self._coerce_collaboration_claims(item.get("claims", [])),
                    criticisms=[
                        str(entry).strip()
                        for entry in item.get("criticisms", [])
                        if str(entry).strip()
                    ],
                    open_questions=[
                        str(entry).strip()
                        for entry in item.get("open_questions", [])
                        if str(entry).strip()
                    ],
                    messages_to_next=[
                        str(entry).strip()
                        for entry in item.get("messages_to_next", [])
                        if str(entry).strip()
                    ],
                )
            )

        return CollaborationSession(
            turns=turns,
            consensus_claims=self._coerce_collaboration_claims(
                payload.get("consensus_claims", [])
            ),
            disputed_claims=[
                str(entry).strip()
                for entry in payload.get("disputed_claims", [])
                if str(entry).strip()
            ],
            open_questions=[
                str(entry).strip()
                for entry in payload.get("open_questions", [])
                if str(entry).strip()
            ],
            coordinator_notes=[
                str(entry).strip()
                for entry in payload.get("coordinator_notes", [])
                if str(entry).strip()
            ],
        )

    @staticmethod
    def _coerce_collaboration_claims(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        claims: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            citation_keys = [
                str(entry).strip()
                for entry in item.get("citation_keys", [])
                if str(entry).strip()
            ]
            status = str(item.get("status", "")).strip() or "tentative"
            claims.append(
                {
                    "claim": claim,
                    "citation_keys": citation_keys,
                    "status": status,
                }
            )
        return claims

    @staticmethod
    def _strip_record_meta(item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        payload.pop("_meta", None)
        return payload

    def _set_selected_source_debug(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        retrieval: dict[str, Any],
        selected: list[SearchResult],
    ) -> None:
        retrieval["selected_sources"] = [
            self._debug_result_record(topic, plan, answers, result) for result in selected
        ]

    def ask_clarifying_questions(
        self,
        topic: str,
        *,
        prefilled_answers: dict[str, str],
        interactive: bool,
    ) -> dict[str, str]:
        answers = dict(prefilled_answers)
        if not interactive:
            return answers

        try:
            system, user = clarifier_prompt(topic, self.settings.max_questions, self.program)
            payload = self.ollama.chat_json(
                system,
                user,
                schema=clarifier_schema(self.settings.max_questions),
            )
            questions = payload.get("questions", [])
        except OllamaError:
            questions = []

        if not questions:
            questions = [
                {"id": "objective", "question": "What exact question should the report answer?"},
                {"id": "audience", "question": "Who is the intended audience for the report?"},
                {"id": "constraints", "question": "Any constraints on time span, methods, or sources?"},
                {"id": "comparison_targets", "question": "Any methods, systems, or time periods to compare against?"},
            ]

        for question in questions[: self.settings.max_questions]:
            question_id = self._slug(question.get("id", "answer"))
            if answers.get(question_id):
                continue
            if not interactive:
                continue
            prompt = question.get("question", "Clarify the topic:")
            reply = input(f"{prompt}\n> ").strip()
            if reply:
                answers[question_id] = reply
        return answers

    def build_plan(self, topic: str, answers: dict[str, str]) -> ResearchPlan:
        system, user = planner_prompt(
            topic,
            answers,
            self.constitution.prompt_snapshot(),
            self.settings.max_queries,
            self.program,
        )
        try:
            payload = self.ollama.chat_json(
                system,
                user,
                schema=planner_schema(self.settings.max_queries),
            )
        except OllamaError:
            payload = {}

        related_topics = payload.get("related_topics", [])
        focus_areas = payload.get("focus_areas", [])
        must_cover = payload.get("must_cover", [])
        source_requirements = payload.get("source_requirements", [])
        rewritten_question = (
            str(payload.get("rewritten_question", "")).strip()
            or self._fallback_rewritten_question(topic, answers)
        )
        queries = [topic]
        queries.extend(
            self._build_static_queries(
                topic,
                answers,
                rewritten_question,
                related_topics,
                focus_areas,
                must_cover,
            )
        )
        queries.append(rewritten_question or topic)
        queries.extend(payload.get("queries", []))
        queries.extend(self._fallback_queries(topic))

        cleaned_queries = []
        seen = set()
        for query in queries:
            item = self._clean_query(query)
            if not item or item in seen:
                continue
            seen.add(item)
            cleaned_queries.append(item)
            if len(cleaned_queries) >= self.settings.max_queries:
                break

        if not source_requirements:
            source_requirements = [
                "Prefer peer-reviewed papers, strong surveys, and official technical documentation.",
                "Include at least one high-signal web source only when it adds context unavailable in papers.",
            ]

        return ResearchPlan(
            queries=cleaned_queries or [topic],
            related_topics=[str(item).strip() for item in related_topics if str(item).strip()],
            focus_areas=[str(item).strip() for item in focus_areas if str(item).strip()],
            rewritten_question=rewritten_question,
            must_cover=[str(item).strip() for item in must_cover if str(item).strip()],
            source_requirements=[
                str(item).strip() for item in source_requirements if str(item).strip()
            ],
        )

    def retrieve_results(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        strategy = self._get_retrieval_strategy(topic, plan, answers, allow_model=True)
        pending_queries = self._dedupe_preserve_order(plan.queries)
        issued_queries: list[str] = []
        all_results: list[SearchResult] = []
        rounds: list[dict[str, Any]] = []

        for round_index in range(self.settings.max_search_rounds):
            if not pending_queries or len(issued_queries) >= self.settings.max_total_queries:
                break

            batch: list[str] = []
            while (
                pending_queries
                and len(batch) < self.settings.max_query_batch_size
                and len(issued_queries) + len(batch) < self.settings.max_total_queries
            ):
                query = pending_queries.pop(0)
                if query in issued_queries or query in batch:
                    continue
                batch.append(query)

            if not batch:
                break

            batch_results: list[SearchResult] = []
            query_debug: list[dict[str, Any]] = []
            for query in batch:
                query_results = self.search.search([query])
                batch_results.extend(query_results)
                ranked_for_query = sorted(
                    query_results,
                    key=lambda item: self._score_result(topic, plan, answers, item),
                    reverse=True,
                )
                query_debug.append(
                    {
                        "query": query,
                        "result_count": len(query_results),
                        "top_results": [
                            self._debug_result_record(topic, plan, answers, item)
                            for item in ranked_for_query[:5]
                        ],
                    }
                )

            all_results = SearchToolkit._dedupe_results(all_results + batch_results)
            ranked = sorted(
                all_results,
                key=lambda item: self._score_result(topic, plan, answers, item),
                reverse=True,
            )
            issued_queries.extend(batch)
            expansion_queries = self._expand_queries_from_results(
                topic,
                plan,
                answers,
                ranked,
                issued_queries,
                pending_queries,
            )
            pending_queries.extend(expansion_queries)
            pending_queries = self._dedupe_preserve_order(pending_queries)

            rounds.append(
                {
                    "round": round_index + 1,
                    "issued_queries": batch,
                    "new_results": len(batch_results),
                    "total_unique_results": len(all_results),
                    "queries": query_debug,
                    "expansion_queries": expansion_queries,
                    "top_ranked": [
                        self._debug_result_record(topic, plan, answers, item)
                        for item in ranked[:8]
                    ],
                }
            )

        ranked_results = sorted(
            all_results,
            key=lambda item: self._score_result(topic, plan, answers, item),
            reverse=True,
        )
        retrieval = {
            "seed_queries": list(plan.queries),
            "strategy": strategy.to_dict(),
            "issued_queries": issued_queries,
            "rounds": rounds,
            "final_ranked": [
                self._debug_result_record(topic, plan, answers, item)
                for item in ranked_results[:20]
            ],
        }
        return ranked_results, retrieval

    def _get_retrieval_strategy(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        *,
        allow_model: bool = False,
    ) -> RetrievalStrategy:
        cache_key = (
            topic,
            tuple(sorted((str(key), str(value)) for key, value in answers.items())),
            plan.rewritten_question,
            tuple(plan.must_cover),
            tuple(plan.focus_areas),
            tuple(plan.related_topics),
        )
        if cache_key in self._retrieval_strategy_cache:
            return self._retrieval_strategy_cache[cache_key]

        strategy = self._heuristic_retrieval_strategy(topic, plan, answers)
        if allow_model:
            system, user = retrieval_strategy_prompt(
                topic,
                plan.rewritten_question,
                answers,
                plan.related_topics,
                plan.focus_areas,
                plan.must_cover,
                self.program,
            )
            try:
                payload = self.ollama.chat_json(
                    system,
                    user,
                    schema=retrieval_strategy_schema(),
                )
                strategy = self._coerce_retrieval_strategy(payload, topic, plan, answers)
            except OllamaError:
                pass

        self._retrieval_strategy_cache[cache_key] = strategy
        return strategy

    def _coerce_retrieval_strategy(
        self,
        payload: dict[str, Any],
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> RetrievalStrategy:
        fallback = self._heuristic_retrieval_strategy(topic, plan, answers)
        anchor_phrases = self._clean_phrase_list(
            payload.get("anchor_phrases", []),
            fallback.anchor_phrases,
            limit=10,
        )
        search_facets = self._clean_phrase_list(
            payload.get("search_facets", []),
            fallback.search_facets,
            limit=10,
        )
        generic_terms = self._clean_term_list(
            payload.get("generic_terms", []),
            fallback.generic_terms,
            limit=16,
        )

        concept_groups: list[RetrievalConceptGroup] = []
        for raw_group in payload.get("concept_groups", []):
            if not isinstance(raw_group, dict):
                continue
            label = str(raw_group.get("label", "")).strip() or f"group-{len(concept_groups) + 1}"
            phrases = self._clean_phrase_list(raw_group.get("phrases", []), [], limit=6)
            if not phrases:
                continue
            priority = raw_group.get("priority", 1)
            if not isinstance(priority, int):
                priority = 1
            priority = max(1, min(priority, 3))
            concept_groups.append(
                RetrievalConceptGroup(label=label, phrases=phrases, priority=priority)
            )

        if not concept_groups:
            concept_groups = fallback.concept_groups

        return RetrievalStrategy(
            anchor_phrases=anchor_phrases,
            search_facets=search_facets,
            generic_terms=generic_terms,
            concept_groups=concept_groups,
        )

    def _heuristic_retrieval_strategy(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> RetrievalStrategy:
        texts = [
            topic,
            plan.rewritten_question,
            answers.get("objective", ""),
            answers.get("constraints", ""),
        ]
        anchors = self._anchor_phrases_heuristic(topic, plan, answers)
        facets = self._facet_phrases_heuristic(topic, plan, answers)
        generic_terms = self._generic_terms_heuristic(topic, plan, answers)

        groups: list[RetrievalConceptGroup] = []
        group_specs = [
            ("must_cover", list(plan.must_cover), 3),
            ("focus", list(plan.focus_areas), 2),
            ("related", list(plan.related_topics), 2),
            ("topic", [topic], 2),
            ("constraints", [answers.get("constraints", ""), answers.get("objective", "")], 1),
        ]
        for label, sources, priority in group_specs:
            for index, source in enumerate(sources, start=1):
                if not str(source).strip():
                    continue
                segments = self._strategy_source_segments(str(source))
                phrases: list[str] = []
                for segment in segments:
                    phrases.extend(self._extract_phrases_from_text(segment, max_phrases=4))
                phrases = self._prune_strategy_phrases(
                    [
                        phrase
                        for phrase in self._dedupe_preserve_order(phrases)
                        if not self._phrase_is_generic(phrase, generic_terms)
                    ],
                    generic_terms,
                    limit=5,
                )
                if phrases:
                    groups.append(
                        RetrievalConceptGroup(
                            label=f"{label}-{index}",
                            phrases=phrases,
                            priority=priority,
                        )
                    )

        if not groups:
            groups = [
                RetrievalConceptGroup(
                    label="topic",
                    phrases=anchors[:5],
                    priority=3,
                )
            ]

        broad_terms = self._ordered_terms(" ".join(texts))
        generic_terms = self._dedupe_preserve_order(generic_terms + broad_terms[:6])[:16]

        return RetrievalStrategy(
            anchor_phrases=anchors,
            search_facets=facets,
            generic_terms=generic_terms,
            concept_groups=groups[:6],
        )

    @staticmethod
    def _strategy_source_segments(text: str) -> list[str]:
        cleaned = re.sub(
            r"\b(?:including|such as|for example|e\.g\.)\b",
            ",",
            text,
            flags=re.I,
        )
        cleaned = re.sub(r"\b(?:rather than|instead of)\b", " over ", cleaned, flags=re.I)
        cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.I)
        raw_segments = re.split(r"[;,]", cleaned)
        segments: list[str] = []
        for segment in raw_segments:
            item = segment.strip()
            item = re.sub(r"\bover\b.*$", "", item, flags=re.I).strip(" :-")
            item = re.sub(
                r"^(?:survey|assess|evaluate|prioritize|include|cover|focus on|focusing on|compare|summarize|explore)\b",
                "",
                item,
                flags=re.I,
            ).strip(" :-")
            lowered = item.lower()
            for marker in (
                "used for",
                "used in",
                "applied to",
                "applied in",
                "leveraged for",
                "leveraged in",
                "deployed for",
                "deployed in",
            ):
                if lowered.startswith("how ") and marker in lowered:
                    item = re.split(rf"\b{re.escape(marker)}\b", item, maxsplit=1, flags=re.I)[-1]
                    item = item.strip(" :-")
                    break
            if item:
                segments.append(item)
        return segments or ([text.strip()] if text.strip() else [])

    @staticmethod
    def _context_has_term(context: str, context_terms: set[str], marker: str) -> bool:
        normalized = re.sub(r"[-_]+", " ", marker.lower()).strip()
        if not normalized:
            return False
        if " " not in normalized:
            return normalized in context_terms
        normalized_context = re.sub(r"[-_]+", " ", context.lower())
        return bool(re.search(rf"\b{re.escape(normalized)}\b", normalized_context))

    @classmethod
    def _context_has_any(
        cls,
        context: str,
        context_terms: set[str],
        markers: list[str],
    ) -> bool:
        return any(cls._context_has_term(context, context_terms, marker) for marker in markers)

    @staticmethod
    def _prune_strategy_phrases(
        phrases: list[str],
        generic_terms: list[str],
        *,
        limit: int,
    ) -> list[str]:
        ordered = sorted(
            phrases,
            key=lambda item: (-len(item.split()), -len(item), item),
        )
        kept: list[str] = []
        generic = set(generic_terms)
        for phrase in ordered:
            terms = [term for term in phrase.split() if term]
            if not terms:
                continue
            if len(terms) == 1 and (len(terms[0]) <= 3 or terms[0] in generic):
                continue
            phrase_terms = set(terms)
            if any(phrase_terms <= set(existing.split()) for existing in kept):
                continue
            kept.append(phrase)
            if len(kept) >= limit:
                break
        return kept

    def select_results(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        ranked = sorted(
            results,
            key=lambda item: self._score_result(topic, plan, answers, item),
            reverse=True,
        )
        shortlisted = self._build_shortlist(ranked)
        shortlisted = self._apply_relevance_critic(topic, plan, answers, shortlisted)
        shortlisted = sorted(
            shortlisted,
            key=lambda item: self._score_result(topic, plan, answers, item),
            reverse=True,
        )
        return self._select_static_results(topic, plan, answers, shortlisted)

    def _apply_relevance_critic(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results or self.settings.max_critic_results <= 0:
            return results

        limit = min(len(results), self.settings.max_critic_results)
        candidates: list[dict[str, str]] = []
        for result in results[:limit]:
            query = next(
                (item.strip() for item in result.matched_queries if str(item).strip()),
                plan.rewritten_question or topic,
            )
            abstract = (result.abstract or "").strip()
            snippet = (result.snippet or "").strip()
            summary = abstract or snippet
            if not result.title.strip() or not summary:
                continue
            candidates.append(
                {
                    "result_id": result.result_id,
                    "query": query,
                    "title": result.title,
                    "abstract": summary,
                    "kind": result.kind,
                    "backend": result.backend,
                }
            )

        if not candidates:
            return results

        cache_key = (
            topic,
            plan.rewritten_question,
            tuple(sorted((str(key), str(value)) for key, value in answers.items())),
            tuple(
                (
                    candidate["result_id"],
                    candidate["query"],
                    candidate["title"],
                    candidate["abstract"],
                )
                for candidate in candidates
            ),
        )
        if cache_key in self._relevance_critic_cache:
            judgments_by_id = self._relevance_critic_cache[cache_key]
        else:
            system, user = relevance_critic_prompt(
                topic,
                plan.rewritten_question,
                answers,
                candidates,
                self.program,
            )
            try:
                payload = self.ollama.chat_json(
                    system,
                    user,
                    schema=relevance_critic_schema(
                        [candidate["result_id"] for candidate in candidates]
                    ),
                )
            except OllamaError:
                return results
            judgments_by_id = {}
            query_by_id = {candidate["result_id"]: candidate["query"] for candidate in candidates}
            for item in payload.get("judgments", []):
                if not isinstance(item, dict):
                    continue
                result_id = str(item.get("result_id", "")).strip()
                if not result_id:
                    continue
                judgments_by_id[result_id] = {
                    "relevant": bool(item.get("relevant", False)),
                    "reason": str(item.get("reason", "")).strip(),
                    "query": query_by_id.get(result_id, ""),
                }
            self._relevance_critic_cache[cache_key] = judgments_by_id

        for result in results:
            judgment = judgments_by_id.get(result.result_id)
            if not judgment:
                continue
            result.critic_relevant = bool(judgment.get("relevant", False))
            result.critic_reason = str(judgment.get("reason", "")).strip()
            result.critic_query = str(judgment.get("query", "")).strip()

        return results

    def resolve_citations(
        self, documents: list[SourceDocument], constitution: dict[str, Any]
    ) -> list[CitationRecord]:
        existing_keys = set(constitution.get("citations", {}).keys())
        citations: list[CitationRecord] = []
        for document in documents:
            citation = self.citations.resolve(document, existing_keys)
            existing_keys.add(citation.cite_key)
            citations.append(citation)
        return citations

    def summarize_documents(
        self,
        plan: ResearchPlan,
        documents: list[SourceDocument],
        citations: list[CitationRecord],
        existing_notes: dict[str, SourceNote] | None = None,
        on_note: Callable[[SourceNote, dict[str, SourceNote]], None] | None = None,
    ) -> dict[str, SourceNote]:
        citation_map = {citation.source_id: citation for citation in citations}
        notes: dict[str, SourceNote] = dict(existing_notes or {})
        for document in documents:
            if document.source_id in notes and notes[document.source_id].summary:
                continue
            chunks = document.text_chunks or ([document.abstract] if document.abstract else [])
            if not chunks and document.text:
                chunks = [document.text[: self.settings.chunk_chars]]
            chunk_summaries: list[dict[str, Any]] = []
            for index, chunk in enumerate(chunks, start=1):
                system, user = chunk_summary_prompt(
                    plan.rewritten_question or document.title,
                    plan.rewritten_question or document.title,
                    document.title,
                    document.url,
                    chunk,
                    index,
                    len(chunks),
                    self.program,
                )
                try:
                    payload = self.ollama.chat_json(
                        system,
                        user,
                        schema=source_note_schema(),
                    )
                except OllamaError:
                    payload = {
                        "summary": chunk[:500],
                        "claims": [],
                        "evidence_snippets": [chunk[:180]] if chunk else [],
                        "related_topics": [],
                    }
                chunk_summaries.append(payload)

            if len(chunk_summaries) == 1:
                merged = chunk_summaries[0]
            else:
                system, user = merge_source_prompt(
                    plan.rewritten_question or document.title,
                    plan.rewritten_question or document.title,
                    document.title,
                    document.url,
                    chunk_summaries,
                    self.program,
                )
                try:
                    merged = self.ollama.chat_json(
                        system,
                        user,
                        schema=source_note_schema(),
                    )
                except OllamaError:
                    merged = {
                        "summary": document.abstract or document.text[:600],
                        "claims": [],
                        "evidence_snippets": [document.abstract[:180]] if document.abstract else [],
                        "related_topics": [],
                    }

            citation_key = citation_map.get(document.source_id)
            notes[document.source_id] = SourceNote(
                source_id=document.source_id,
                title=document.title,
                url=document.url,
                citation_key=citation_key.cite_key if citation_key else "",
                summary=str(merged.get("summary", "")).strip(),
                claims=[
                    str(item).strip()
                    for item in merged.get("claims", [])
                    if str(item).strip()
                ],
                related_topics=[
                    str(item).strip()
                    for item in merged.get("related_topics", [])
                    if str(item).strip()
                ],
                evidence_snippets=[
                    str(item).strip()
                    for item in merged.get("evidence_snippets", [])
                    if str(item).strip()
                ],
            )
            if on_note is not None:
                on_note(notes[document.source_id], notes)
        return notes

    def collaborate(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        note_by_source: dict[str, SourceNote],
        citations: list[CitationRecord],
    ) -> CollaborationSession:
        if not note_by_source or not citations:
            return CollaborationSession()

        source_notes = [note.to_dict() for note in note_by_source.values()]
        citation_payloads = [citation.to_dict() for citation in citations]
        citation_keys = [citation.cite_key for citation in citations]
        turns: list[CollaborationTurn] = []

        roles = [
            (
                "EvidenceAgent",
                "Extract the strongest evidence-backed claims. Favor precise claims with direct citation support and call out any scope limits.",
            ),
            (
                "SkepticAgent",
                "Challenge overreach in prior turns. Downgrade or reject claims that are weakly grounded, ambiguous, or contradicted by the notes.",
            ),
            (
                "GapAgent",
                "Look for what the debate is missing: uncovered subtopics, missing comparisons, and unresolved methodological gaps.",
            ),
        ]

        for role_name, role_instruction in roles:
            system, user = collaboration_worker_prompt(
                role_name=role_name,
                role_instruction=role_instruction,
                topic=topic,
                rewritten_question=plan.rewritten_question,
                answers=answers,
                prior_turns=[turn.to_dict() for turn in turns],
                source_notes=source_notes,
                citations=citation_payloads,
                program=self.program,
            )
            try:
                payload = self.ollama.chat_json(
                    system,
                    user,
                    schema=collaboration_turn_schema(citation_keys),
                )
            except OllamaError:
                payload = {}
            turns.append(self._coerce_collaboration_turn(payload, role_name))

        system, user = collaboration_coordinator_prompt(
            topic=topic,
            rewritten_question=plan.rewritten_question,
            answers=answers,
            turns=[turn.to_dict() for turn in turns],
            source_notes=source_notes,
            citations=citation_payloads,
            program=self.program,
        )
        try:
            payload = self.ollama.chat_json(
                system,
                user,
                schema=collaboration_session_schema(citation_keys),
            )
        except OllamaError:
            payload = self._fallback_collaboration_payload(turns)
        session = self._coerce_collaboration_session(payload)
        if not session.turns:
            session.turns = turns
        elif len(session.turns) < len(turns):
            session.turns.extend(turns[len(session.turns) :])
        return session

    def _coerce_collaboration_turn(
        self,
        payload: dict[str, Any],
        role_name: str,
    ) -> CollaborationTurn:
        claims = self._coerce_collaboration_claims(payload.get("claims", []))
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            if claims:
                summary = claims[0]["claim"]
            else:
                summary = f"{role_name} produced no structured summary."
        return CollaborationTurn(
            role=role_name,
            summary=summary,
            claims=claims,
            criticisms=[
                str(item).strip()
                for item in payload.get("criticisms", [])
                if str(item).strip()
            ],
            open_questions=[
                str(item).strip()
                for item in payload.get("open_questions", [])
                if str(item).strip()
            ],
            messages_to_next=[
                str(item).strip()
                for item in payload.get("messages_to_next", [])
                if str(item).strip()
            ],
        )

    @staticmethod
    def _fallback_collaboration_payload(turns: list[CollaborationTurn]) -> dict[str, Any]:
        supported_claims: list[dict[str, Any]] = []
        disputed_claims: list[str] = []
        open_questions: list[str] = []
        coordinator_notes: list[str] = []
        for turn in turns:
            for claim in turn.claims:
                if claim.get("status") in {"supported", "tentative"} and claim not in supported_claims:
                    supported_claims.append(claim)
                if claim.get("status") in {"challenged", "rejected"}:
                    disputed_claims.append(str(claim.get("claim", "")).strip())
            open_questions.extend(turn.open_questions)
            if turn.criticisms:
                coordinator_notes.append(f"{turn.role}: " + "; ".join(turn.criticisms[:2]))
        return {
            "turns": [turn.to_dict() for turn in turns],
            "consensus_claims": supported_claims[:8],
            "disputed_claims": [item for item in disputed_claims if item][:8],
            "open_questions": [item for item in open_questions if item][:8],
            "coordinator_notes": coordinator_notes[:8],
        }

    def synthesize(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        note_by_source: dict[str, SourceNote],
        citations: list[CitationRecord],
        collaboration: CollaborationSession,
    ) -> SynthesisResult:
        system, user = writer_prompt(
            topic,
            plan.rewritten_question,
            plan.must_cover,
            plan.source_requirements,
            answers,
            self.constitution.prompt_snapshot(),
            collaboration.to_dict(),
            [note.to_dict() for note in note_by_source.values()],
            [citation.to_dict() for citation in citations],
            self.program,
        )
        try:
            payload = self.ollama.chat_json(
                system,
                user,
                schema=writer_schema([citation.cite_key for citation in citations]),
            )
        except OllamaError:
            payload = self._fallback_synthesis_payload(topic, note_by_source)

        synthesis = self._coerce_synthesis_payload(payload, topic)
        return self._validate_synthesis(synthesis, note_by_source, citations)

    def write_outputs(
        self,
        *,
        topic: str,
        answers: dict[str, str],
        plan: ResearchPlan,
        selected: list[SearchResult],
        documents: list[SourceDocument],
        citations: list[CitationRecord],
        source_notes: list[SourceNote],
        synthesis: SynthesisResult,
        retrieval: dict[str, Any],
        collaboration: CollaborationSession,
    ) -> dict[str, Path]:
        report_path = self.output_dir / self.settings.report_filename
        references_path = self.output_dir / self.settings.references_filename
        run_path = self.output_dir / self.settings.run_filename
        retrieval_path = self.output_dir / self.settings.retrieval_filename

        references_path.write_text(
            "\n\n".join(
                self._sanitize_bibtex_entry(citation.bibtex).strip()
                for citation in citations
                if citation.bibtex
            ).strip()
            + ("\n" if citations else ""),
            encoding="utf-8",
        )
        report_path.write_text(self._render_report(synthesis), encoding="utf-8")
        retrieval_path.write_text(
            json.dumps(retrieval, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        compile_result = self._compile_latex(report_path)
        compiled = compile_result.get("pdf")
        run_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "topic": topic,
                    "provider": self.settings.llm_provider,
                    "model": self.settings.model_display_name(),
                    "answers": answers,
                    "plan": plan.to_dict(),
                    "retrieval": retrieval,
                    "selected_sources": [item.to_dict() for item in selected],
                    "documents": [item.to_dict() for item in documents],
                    "citations": [item.to_dict() for item in citations],
                    "source_notes": [item.to_dict() for item in source_notes],
                    "collaboration": collaboration.to_dict(),
                    "synthesis": synthesis.to_dict(),
                    "compiled_pdf": str(compiled) if compiled else "",
                    "latex": {
                        "status": compile_result.get("status", ""),
                        "message": compile_result.get("message", ""),
                        "log_path": str(report_path.with_suffix(".log")),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "report": report_path,
            "references": references_path,
            **({"pdf": compiled} if compiled else {}),
            "program": self.output_dir / self.settings.program_filename,
            "constitution": self.output_dir / self.settings.constitution_filename,
            "constitution_bib": self.output_dir / self.settings.constitution_bib_filename,
            "run": run_path,
            "retrieval": retrieval_path,
        }

    def _write_checkpoint(
        self,
        *,
        topic: str,
        status: str,
        answers: dict[str, str],
        plan: ResearchPlan,
        retrieval: dict[str, Any] | None = None,
        selected: list[SearchResult] | None = None,
        documents: list[SourceDocument] | None = None,
        citations: list[CitationRecord] | None = None,
        note_by_source: dict[str, SourceNote] | None = None,
        collaboration: CollaborationSession | None = None,
        synthesis: SynthesisResult | None = None,
        progress: dict[str, Any] | None = None,
    ) -> None:
        run_path = self.output_dir / self.settings.run_filename
        retrieval_path = self.output_dir / self.settings.retrieval_filename
        references_path = self.output_dir / self.settings.references_filename

        if retrieval is not None:
            retrieval_path.write_text(
                json.dumps(retrieval, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        if citations is not None:
            references_path.write_text(
                "\n\n".join(
                    self._sanitize_bibtex_entry(citation.bibtex).strip()
                    for citation in citations
                    if citation.bibtex
                ).strip()
                + ("\n" if citations else ""),
                encoding="utf-8",
            )

        payload: dict[str, Any] = {
            "status": status,
            "topic": topic,
            "provider": self.settings.llm_provider,
            "model": self.settings.model_display_name(),
            "answers": answers,
            "plan": plan.to_dict(),
            "retrieval": retrieval or {},
            "selected_sources": [item.to_dict() for item in (selected or [])],
            "documents": [item.to_dict() for item in (documents or [])],
            "citations": [item.to_dict() for item in (citations or [])],
            "source_notes": [note.to_dict() for note in (note_by_source or {}).values()],
            "collaboration": collaboration.to_dict() if collaboration is not None else None,
            "synthesis": synthesis.to_dict() if synthesis is not None else None,
            "compiled_pdf": "",
        }
        if progress:
            payload["progress"] = progress
        run_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _checkpoint_note_progress(
        self,
        *,
        topic: str,
        answers: dict[str, str],
        plan: ResearchPlan,
        retrieval: dict[str, Any],
        selected: list[SearchResult],
        documents: list[SourceDocument],
        citations: list[CitationRecord],
        note: SourceNote,
        note_by_source: dict[str, SourceNote],
    ) -> None:
        self.constitution.checkpoint_sources(
            [note],
            [],
            checkpoint_stage="summarizing",
        )
        self._write_checkpoint(
            topic=topic,
            status="summarizing",
            answers=answers,
            plan=plan,
            retrieval=retrieval,
            selected=selected,
            documents=documents,
            citations=citations,
            note_by_source=note_by_source,
            progress={
                "summarized_sources": len(note_by_source),
                "total_sources": len(documents),
                "last_source_id": note.source_id,
            },
        )

    def _apply_summary_budget(
        self,
        selected: list[SearchResult],
        documents: list[SourceDocument],
    ) -> tuple[list[SearchResult], list[SourceDocument], dict[str, Any]]:
        paired: list[tuple[SearchResult, SourceDocument, list[str]]] = []
        estimated_before = 0
        for result, document in zip(selected, documents):
            chunks = document.text_chunks or ([document.abstract] if document.abstract else [])
            if not chunks and document.text:
                chunks = [document.text[: self.settings.chunk_chars]]
            paired.append((result, document, list(chunks)))
            estimated_before += len(chunks) + (1 if len(chunks) > 1 else 0)

        max_calls = max(1, self.settings.max_summary_model_calls)
        allocations: dict[str, int] = {}
        breadth_first: list[tuple[SearchResult, SourceDocument, list[str]]] = []
        remaining_calls = max_calls
        dropped_sources: list[dict[str, Any]] = []

        for result, document, chunks in paired:
            if not chunks:
                dropped_sources.append(
                    {
                        "source_id": document.source_id,
                        "title": document.title,
                        "reason": "no_extractable_text",
                    }
                )
                continue
            if remaining_calls < 1:
                dropped_sources.append(
                    {
                        "source_id": document.source_id,
                        "title": document.title,
                        "reason": "summary_call_budget",
                    }
                )
                continue
            allocations[document.source_id] = 1
            breadth_first.append((result, document, chunks))
            remaining_calls -= 1

        progress = True
        while remaining_calls > 0 and progress:
            progress = False
            for _, document, chunks in breadth_first:
                allocated = allocations.get(document.source_id, 0)
                if allocated >= len(chunks):
                    continue
                extra_cost = 2 if allocated == 1 else 1
                if remaining_calls < extra_cost:
                    continue
                allocations[document.source_id] = allocated + 1
                remaining_calls -= extra_cost
                progress = True
                if remaining_calls <= 0:
                    break

        budgeted_selected: list[SearchResult] = []
        budgeted_documents: list[SourceDocument] = []
        truncated_sources: list[dict[str, Any]] = []
        estimated_after = 0

        for result, document, chunks in paired:
            allocated = allocations.get(document.source_id, 0)
            if allocated <= 0:
                continue
            sampled_chunks = self._sample_chunks_limit(chunks, allocated)
            if len(sampled_chunks) < len(chunks):
                truncated_sources.append(
                    {
                        "source_id": document.source_id,
                        "title": document.title,
                        "available_chunks": len(chunks),
                        "used_chunks": len(sampled_chunks),
                    }
                )
            document.text_chunks = sampled_chunks
            budgeted_selected.append(result)
            budgeted_documents.append(document)
            estimated_after += len(sampled_chunks) + (1 if len(sampled_chunks) > 1 else 0)

        budget = {
            "max_summary_model_calls": max_calls,
            "estimated_summary_calls_before_budget": estimated_before,
            "estimated_summary_calls_after_budget": estimated_after,
            "selected_source_count_before_budget": len(selected),
            "selected_source_count_after_budget": len(budgeted_selected),
            "dropped_sources": dropped_sources,
            "truncated_sources": truncated_sources,
        }
        return budgeted_selected, budgeted_documents, budget

    @staticmethod
    def _sample_chunks_limit(chunks: list[str], limit: int) -> list[str]:
        if limit <= 0 or not chunks:
            return []
        if len(chunks) <= limit:
            return list(chunks)
        if limit == 1:
            return [chunks[0]]
        last_index = len(chunks) - 1
        sample_indices = sorted({round(i * last_index / (limit - 1)) for i in range(limit)})
        return [chunks[index] for index in sample_indices]

    def _compile_latex(self, report_path: Path) -> dict[str, Any]:
        if not self.settings.compile_latex:
            return {
                "pdf": None,
                "status": "skipped",
                "message": "LaTeX compilation disabled by settings.",
            }

        pdf_path = report_path.with_suffix(".pdf")
        latexmk = shutil.which("latexmk")
        latex_engine = self._preferred_latex_engine()
        if latexmk:
            latexmk_flag = "-pdf"
            if latex_engine and latex_engine.name == "lualatex":
                latexmk_flag = "-lualatex"
            elif latex_engine and latex_engine.name == "xelatex":
                latexmk_flag = "-xelatex"
            command = [
                latexmk,
                latexmk_flag,
                "-interaction=nonstopmode",
                "-halt-on-error",
                report_path.name,
            ]
        else:
            bibtex = shutil.which("bibtex")
            if not latex_engine or not bibtex:
                return {
                    "pdf": None,
                    "status": "skipped",
                    "message": "LaTeX tools not available. Install latexmk or lualatex/xelatex/pdflatex + bibtex.",
                }
            stem = report_path.stem
            commands = [
                [str(latex_engine), "-interaction=nonstopmode", "-halt-on-error", report_path.name],
                [bibtex, stem],
                [str(latex_engine), "-interaction=nonstopmode", "-halt-on-error", report_path.name],
                [str(latex_engine), "-interaction=nonstopmode", "-halt-on-error", report_path.name],
            ]
            for command in commands:
                try:
                    subprocess.run(
                        command,
                        cwd=self.output_dir,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=self.settings.latex_timeout_seconds,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    detail = self._latex_failure_detail(report_path)
                    if pdf_path.exists():
                        return {
                            "pdf": pdf_path,
                            "status": "partial",
                            "message": self._combine_failure_message(
                                f"LaTeX command failed after producing a PDF: {exc}",
                                detail,
                            ),
                        }
                    return {
                        "pdf": None,
                        "status": "failed",
                        "message": self._combine_failure_message(
                            f"LaTeX command failed: {exc}",
                            detail,
                        ),
                    }
            return {
                "pdf": pdf_path if pdf_path.exists() else None,
                "status": "succeeded" if pdf_path.exists() else "failed",
                "message": "" if pdf_path.exists() else "LaTeX commands completed without producing a PDF.",
            }

        try:
            subprocess.run(
                command,
                cwd=self.output_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.settings.latex_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = self._latex_failure_detail(report_path)
            if pdf_path.exists():
                return {
                    "pdf": pdf_path,
                    "status": "partial",
                    "message": self._combine_failure_message(
                        f"latexmk failed after producing a PDF: {exc}",
                        detail,
                    ),
                }
            return {
                "pdf": None,
                "status": "failed",
                "message": self._combine_failure_message(
                    f"latexmk failed: {exc}",
                    detail,
                ),
            }
        return {
            "pdf": pdf_path if pdf_path.exists() else None,
            "status": "succeeded" if pdf_path.exists() else "failed",
            "message": "" if pdf_path.exists() else "latexmk completed without producing a PDF.",
        }

    def _coerce_synthesis_payload(self, payload: dict[str, Any], topic: str) -> SynthesisResult:
        sections = []
        for section in payload.get("sections", []):
            sections.append(
                ReportSection(
                    heading=str(section.get("heading", "Section")).strip(),
                    paragraphs=[
                        {
                            "text": str(paragraph.get("text", "")).strip(),
                            "citation_keys": [
                                str(key).strip()
                                for key in paragraph.get("citation_keys", [])
                                if str(key).strip()
                            ],
                        }
                        for paragraph in section.get("paragraphs", [])
                        if str(paragraph.get("text", "")).strip()
                    ],
                )
            )

        findings = []
        for item in payload.get("findings", []):
            finding_id = str(item.get("finding_id", "")).strip() or f"finding-{len(findings)+1}"
            findings.append(
                Finding(
                    finding_id=finding_id,
                    claim=str(item.get("claim", "")).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                    citation_keys=[
                        str(key).strip()
                        for key in item.get("citation_keys", [])
                        if str(key).strip()
                    ],
                )
            )

        return SynthesisResult(
            title=str(payload.get("title", topic.title())).strip() or topic.title(),
            abstract=str(payload.get("abstract", "")).strip(),
            sections=sections,
            findings=findings,
            notes=[
                str(item).strip() for item in payload.get("notes", []) if str(item).strip()
            ],
            delete_citation_keys=[
                str(item).strip()
                for item in payload.get("delete_citation_keys", [])
                if str(item).strip()
            ],
            delete_finding_ids=[
                str(item).strip()
                for item in payload.get("delete_finding_ids", [])
                if str(item).strip()
            ],
        )

    def _validate_synthesis(
        self,
        synthesis: SynthesisResult,
        note_by_source: dict[str, SourceNote],
        citations: list[CitationRecord],
    ) -> SynthesisResult:
        valid_keys = {citation.cite_key for citation in citations}
        notes = list(synthesis.notes)
        source_notes = list(note_by_source.values())

        validated_sections: list[ReportSection] = []
        for section in synthesis.sections:
            paragraphs = []
            for paragraph in section.paragraphs:
                text = self._strip_inline_citation_markers(
                    str(paragraph.get("text", "")).strip(),
                    valid_keys,
                )
                if not text:
                    continue
                citation_keys = self._normalize_citation_keys(
                    paragraph.get("citation_keys", []), valid_keys
                )
                if not citation_keys:
                    citation_keys = self._infer_citation_keys(text, source_notes, valid_keys)
                if not citation_keys:
                    notes.append(
                        f"Dropped unsupported paragraph in section '{section.heading}' because it had no valid source grounding."
                    )
                    continue
                paragraphs.append({"text": text, "citation_keys": citation_keys})
            if paragraphs:
                validated_sections.append(ReportSection(section.heading, paragraphs))

        validated_findings: list[Finding] = []
        for finding in synthesis.findings:
            text = " ".join(part for part in [finding.claim, finding.evidence] if part).strip()
            citation_keys = self._normalize_citation_keys(finding.citation_keys, valid_keys)
            if not citation_keys:
                citation_keys = self._infer_citation_keys(text, source_notes, valid_keys)
            if not citation_keys:
                notes.append(
                    f"Dropped unsupported finding '{finding.finding_id}' because it had no valid citation support."
                )
                continue
            validated_findings.append(
                Finding(
                    finding_id=finding.finding_id,
                    claim=finding.claim,
                    evidence=finding.evidence,
                    citation_keys=citation_keys,
                )
            )

        if not validated_sections and source_notes:
            fallback_paragraphs = []
            for note in source_notes[: min(3, len(source_notes))]:
                if note.summary and note.citation_key in valid_keys:
                    fallback_paragraphs.append(
                        {"text": note.summary, "citation_keys": [note.citation_key]}
                    )
            if fallback_paragraphs:
                validated_sections.append(ReportSection("Findings", fallback_paragraphs))

        protected_citation_keys = set(valid_keys)
        filtered_delete_citation_keys = self._dedupe_preserve_order(
            [
                key
                for key in synthesis.delete_citation_keys
                if key and key not in protected_citation_keys
            ]
        )
        ignored_delete_citation_keys = self._dedupe_preserve_order(
            [
                key
                for key in synthesis.delete_citation_keys
                if key and key in protected_citation_keys
            ]
        )
        if ignored_delete_citation_keys:
            notes.append(
                "Ignored deletion request for current-run citation(s): "
                + ", ".join(ignored_delete_citation_keys)
            )

        protected_finding_ids = {finding.finding_id for finding in validated_findings}
        filtered_delete_finding_ids = self._dedupe_preserve_order(
            [
                finding_id
                for finding_id in synthesis.delete_finding_ids
                if finding_id and finding_id not in protected_finding_ids
            ]
        )
        ignored_delete_finding_ids = self._dedupe_preserve_order(
            [
                finding_id
                for finding_id in synthesis.delete_finding_ids
                if finding_id and finding_id in protected_finding_ids
            ]
        )
        if ignored_delete_finding_ids:
            notes.append(
                "Ignored deletion request for current-run finding(s): "
                + ", ".join(ignored_delete_finding_ids)
            )

        return SynthesisResult(
            title=synthesis.title,
            abstract=synthesis.abstract,
            sections=validated_sections,
            findings=validated_findings,
            notes=self._dedupe_preserve_order(notes),
            delete_citation_keys=filtered_delete_citation_keys,
            delete_finding_ids=filtered_delete_finding_ids,
        )

    def _ensure_source_diversity(
        self,
        ranked_results: list[SearchResult],
        preselected: list[SearchResult],
    ) -> list[SearchResult]:
        selected: list[SearchResult] = []
        selected_ids: set[str] = set()
        backend_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()

        def add(result: SearchResult, *, ignore_backend_limit: bool = False) -> bool:
            if result.result_id in selected_ids:
                return False
            if (
                not ignore_backend_limit
                and backend_counts[result.backend] >= self.settings.max_sources_per_backend
            ):
                return False
            domain = self._domain(result.url)
            if domain and domain_counts[domain] >= 2:
                return False
            selected.append(result)
            selected_ids.add(result.result_id)
            backend_counts[result.backend] += 1
            if domain:
                domain_counts[domain] += 1
            return True

        papers = [item for item in ranked_results if item.kind == "paper"]
        webs = [item for item in ranked_results if item.kind == "web"]
        paper_target = min(
            self.settings.min_papers,
            len(papers),
            self.settings.max_selected_sources,
        )
        web_target = min(
            self.settings.min_web_sources,
            len(webs),
            max(0, self.settings.max_selected_sources - paper_target),
        )

        prioritized_pool = self._dedupe_results_by_id(preselected + ranked_results)

        while sum(1 for item in selected if item.kind == "paper") < paper_target:
            if not any(add(item) for item in prioritized_pool if item.kind == "paper"):
                break
        while sum(1 for item in selected if item.kind == "web") < web_target:
            if not any(add(item) for item in prioritized_pool if item.kind == "web"):
                break

        for result in prioritized_pool:
            if len(selected) >= self.settings.max_selected_sources:
                break
            add(result)

        if len(selected) < self.settings.max_selected_sources:
            for result in prioritized_pool:
                if len(selected) >= self.settings.max_selected_sources:
                    break
                add(result, ignore_backend_limit=True)

        return selected[: self.settings.max_selected_sources]

    @staticmethod
    def _dedupe_results_by_id(results: list[SearchResult]) -> list[SearchResult]:
        seen: set[str] = set()
        ordered: list[SearchResult] = []
        for result in results:
            if result.result_id in seen:
                continue
            seen.add(result.result_id)
            ordered.append(result)
        return ordered

    def _build_shortlist(self, ranked_results: list[SearchResult]) -> list[SearchResult]:
        shortlist_target = min(
            len(ranked_results),
            max(self.settings.max_selected_sources * 8, len(ranked_results[: self.settings.max_queries * 3])),
        )
        shortlisted: list[SearchResult] = []
        seen: set[str] = set()

        for result in ranked_results:
            if len(shortlisted) >= shortlist_target:
                break
            if result.result_id in seen:
                continue
            shortlisted.append(result)
            seen.add(result.result_id)

        query_best: dict[str, SearchResult] = {}
        for result in ranked_results:
            for query in result.matched_queries:
                query_best.setdefault(query, result)

        for result in query_best.values():
            if result.result_id in seen:
                continue
            shortlisted.append(result)
            seen.add(result.result_id)

        return shortlisted

    def _select_static_results(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        ranked_results: list[SearchResult],
    ) -> list[SearchResult]:
        selected: list[SearchResult] = []
        selected_ids: set[str] = set()
        backend_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        covered_queries: set[str] = set()
        uncovered_terms = set(self._selection_terms(topic, plan, answers))
        paper_target = min(
            self.settings.min_papers,
            sum(1 for item in ranked_results if item.kind == "paper"),
            self.settings.max_selected_sources,
        )
        web_target = min(
            self.settings.min_web_sources,
            sum(1 for item in ranked_results if item.kind == "web"),
            max(0, self.settings.max_selected_sources - paper_target),
        )

        while len(selected) < self.settings.max_selected_sources:
            best_result: SearchResult | None = None
            best_score: tuple[int, int, int, int] | None = None

            for result in ranked_results:
                if result.result_id in selected_ids:
                    continue
                candidate_score = self._incremental_selection_score(
                    topic,
                    plan,
                    answers,
                    result,
                    uncovered_terms,
                    covered_queries,
                    backend_counts,
                    domain_counts,
                    paper_target,
                    web_target,
                    selected,
                )
                if best_score is None or candidate_score > best_score:
                    best_result = result
                    best_score = candidate_score

            if best_result is None or best_score is None or best_score[0] < -20:
                break

            selected.append(best_result)
            selected_ids.add(best_result.result_id)
            backend_counts[best_result.backend] += 1
            domain = self._domain(best_result.url)
            if domain:
                domain_counts[domain] += 1
            covered_queries.update(best_result.matched_queries)
            uncovered_terms.difference_update(self._result_terms(best_result))

        return self._ensure_source_diversity(ranked_results, selected)

    def _incremental_selection_score(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result: SearchResult,
        uncovered_terms: set[str],
        covered_queries: set[str],
        backend_counts: Counter[str],
        domain_counts: Counter[str],
        paper_target: int,
        web_target: int,
        selected: list[SearchResult],
    ) -> tuple[int, int, int, int]:
        base_score = self._score_result(topic, plan, answers, result)[0]
        result_terms = self._result_terms(result)
        term_gain = len(result_terms & uncovered_terms)
        query_gain = len(set(result.matched_queries) - covered_queries)
        kind_need_bonus = 0
        if result.kind == "paper" and sum(1 for item in selected if item.kind == "paper") < paper_target:
            kind_need_bonus += 8
        if result.kind == "web" and sum(1 for item in selected if item.kind == "web") < web_target:
            kind_need_bonus += 5

        penalty = 0
        if backend_counts[result.backend] >= self.settings.max_sources_per_backend:
            penalty += 8
        domain = self._domain(result.url)
        if domain and domain_counts[domain] >= 1:
            penalty += 4

        return (
            base_score + term_gain * 4 + query_gain * 3 + kind_need_bonus - penalty,
            term_gain,
            query_gain,
            kind_need_bonus - penalty,
        )

    def _expand_queries_from_results(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        ranked_results: list[SearchResult],
        issued_queries: list[str],
        pending_queries: list[str],
    ) -> list[str]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        groups = self._strategy_concept_groups(topic, plan, answers)
        anchors = self._anchor_phrases(topic, plan, answers)[:8]
        facets = self._facet_phrases(topic, plan, answers)[:6]
        result_phrases: list[str] = []
        for result in ranked_results[: self.settings.max_ranked_results_for_expansion]:
            result_phrases.extend(self._result_phrases(result))
        result_phrases = [
            phrase
            for phrase in self._dedupe_preserve_order(result_phrases)
            if not self._phrase_is_generic(phrase, strategy.generic_terms)
        ][:10]

        existing = set(issued_queries) | set(pending_queries)
        buckets: list[list[str]] = []

        result_facet = [
            f"{phrase} {facet}"
            for phrase in result_phrases
            for facet in facets
            if not self._phrases_overlap(phrase, facet)
        ]
        anchor_facet = [
            f"{anchor} {facet}"
            for anchor in anchors
            for facet in facets
            if not self._phrases_overlap(anchor, facet)
        ]
        anchor_result = [
            f"{anchor} {phrase}"
            for anchor in anchors
            for phrase in result_phrases
            if not self._phrases_overlap(anchor, phrase)
        ]
        group_bridge: list[str] = []
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                for left_phrase in left.phrases[:2]:
                    for right_phrase in right.phrases[:2]:
                        if self._phrases_overlap(left_phrase, right_phrase):
                            continue
                        group_bridge.append(f"{left_phrase} {right_phrase}")
        group_result = [
            f"{group.phrases[0]} {phrase}"
            for group in groups
            if group.phrases
            for phrase in result_phrases
            if not self._phrases_overlap(group.phrases[0], phrase)
        ]
        buckets.extend([group_bridge, result_facet, anchor_facet, anchor_result, group_result])

        ordered: list[str] = []
        seen = set()
        for bucket in buckets:
            scored_candidates: list[tuple[tuple[int, int, int], str]] = []
            for candidate in bucket:
                cleaned = self._clean_query(candidate)
                if not cleaned or cleaned in existing or cleaned in seen:
                    continue
                score = self._query_priority(topic, plan, answers, cleaned)
                scored_candidates.append((score, cleaned))
            scored_candidates.sort(reverse=True)
            for _, query in scored_candidates:
                if query in seen:
                    continue
                seen.add(query)
                ordered.append(query)
                if len(ordered) >= self.settings.max_expansion_queries_per_round:
                    return ordered
        return ordered

    def _debug_result_record(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result: SearchResult,
    ) -> dict[str, Any]:
        score, anchor_hits, query_hits, title_overlap = self._score_result(
            topic, plan, answers, result
        )
        result_terms = self._result_terms(result)
        return {
            "result_id": result.result_id,
            "title": result.title,
            "url": result.url,
            "backend": result.backend,
            "kind": result.kind,
            "year": result.year,
            "doi": result.doi,
            "arxiv_id": result.arxiv_id,
            "matched_queries": list(result.matched_queries),
            "citation_count": result.citation_count,
            "score": score,
            "score_components": {
                "anchor_hits": anchor_hits,
                "group_hits": self._matched_concept_group_count(topic, plan, answers, result_terms),
                "query_hits": query_hits,
                "title_overlap": title_overlap,
                "critic_bonus": self._critic_bonus(result),
                "generic_penalty": self._generic_page_penalty(topic, plan, answers, result),
                "thin_penalty": self._thin_result_penalty(result),
            },
            "critic": {
                "applied": result.critic_relevant is not None,
                "relevant": result.critic_relevant,
                "query": result.critic_query,
                "reason": result.critic_reason,
            },
        }

    def _anchor_phrases(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        heuristic = self._anchor_phrases_heuristic(topic, plan, answers)
        return self._clean_phrase_list(
            list(strategy.anchor_phrases) + heuristic,
            heuristic,
            limit=12,
        )

    def _facet_phrases(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        heuristic = self._facet_phrases_heuristic(topic, plan, answers)
        return self._clean_phrase_list(
            list(strategy.search_facets) + heuristic,
            heuristic,
            limit=10,
        )

    def _anchor_phrases_heuristic(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        phrases: list[str] = []
        generic_terms = self._generic_terms_heuristic(topic, plan, answers)
        texts = [
            *list(plan.must_cover),
            *list(plan.focus_areas),
            *list(plan.related_topics),
            topic,
            plan.rewritten_question,
            answers.get("objective", ""),
            answers.get("constraints", ""),
        ]
        for text in texts:
            phrases.extend(self._extract_phrases_from_text(text))
        return self._prune_strategy_phrases(
            [
                phrase
                for phrase in self._dedupe_preserve_order(phrases)
                if not self._phrase_is_generic(phrase, generic_terms)
            ],
            generic_terms,
            limit=10,
        )

    def _facet_phrases_heuristic(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        context = " ".join(
            [
                topic,
                plan.rewritten_question,
                " ".join(plan.must_cover),
                " ".join(plan.focus_areas),
                " ".join(answers.values()),
            ]
        ).lower()
        context_terms = set(self._ordered_terms(context))
        facets: list[str] = []
        if self._context_has_any(
            context,
            context_terms,
            [
                "simulation",
                "simulator",
                "full system",
                "full-system",
                "system level",
                "system-level",
                "cycle-accurate",
                "workload",
                "digital twin",
            ],
        ):
            facets.extend(
                [
                    "full system modeling",
                    "system level simulation",
                    "performance analysis",
                    "behavioral modeling",
                ]
            )
        if self._context_has_any(
            context,
            context_terms,
            [
                "architecture",
                "architectures",
                "microarchitecture",
                "simulator",
                "simulators",
                "gem5",
                "systemc",
                "verilog",
                "cpu",
                "gpu",
                "accelerator",
            ],
        ):
            facets.extend(
                [
                    "computer architecture",
                    "architecture simulation",
                    "system design",
                ]
            )
        if self._context_has_any(
            context,
            context_terms,
            [
                "full system",
                "full-system",
                "system level",
                "system-level",
                "integration",
                "workload",
                "platform",
            ],
        ):
            facets.append("full system modeling")
        if self._context_has_any(
            context,
            context_terms,
            ["logic", "circuit", "circuits", "electronic", "electronics", "device"],
        ):
            facets.extend(
                [
                    "behavioral hdl",
                    "logic circuits",
                    "digital systems",
                ]
            )
        facet_map = {
            "survey": {"survey", "review", "overview", "taxonomy"},
            "benchmark study": {"benchmark", "evaluation", "performance", "measurement"},
            "design space exploration": {"tradeoff", "optimization", "design", "scaling"},
            "full system modeling": {"integration", "workload", "platform", "stack"},
            "architecture simulation": {"architecture", "microarchitecture", "simulator"},
            "behavioral modeling": {"behavioral", "model", "modeling", "simulation"},
            "circuit modeling": {"circuit", "logic", "device", "hardware"},
            "implementation study": {"implementation", "prototype", "workflow", "toolchain"},
        }
        for facet, triggers in facet_map.items():
            if context_terms & triggers:
                facets.append(facet)
        return self._dedupe_preserve_order(facets)

    def _generic_terms_heuristic(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        generic_terms = [
            "analysis",
            "architecture",
            "benchmark",
            "compare",
            "comparison",
            "design",
            "evaluation",
            "implementation",
            "model",
            "modeling",
            "performance",
            "research",
            "simulation",
            "simulator",
            "study",
            "system",
            "systems",
            "workflow",
        ]
        context_terms = self._ordered_terms(
            " ".join(
                [
                    topic,
                    plan.rewritten_question,
                    " ".join(plan.focus_areas),
                    " ".join(answers.values()),
                ]
            )
        )
        boosted = [term for term in context_terms if term in set(generic_terms)]
        return self._dedupe_preserve_order(generic_terms + boosted)

    @staticmethod
    def _phrase_is_generic(phrase: str, generic_terms: list[str]) -> bool:
        terms = [term for term in phrase.split() if term]
        if not terms:
            return True
        generic = set(generic_terms)
        return all(term in generic for term in terms)

    def _clean_phrase_list(
        self,
        values: list[Any],
        fallback: list[str],
        *,
        limit: int,
    ) -> list[str]:
        phrases: list[str] = []
        for value in values:
            cleaned = self._clean_query(value)
            if cleaned:
                phrases.append(cleaned.lower())
        phrases = self._dedupe_preserve_order(phrases)
        if not phrases:
            phrases = self._dedupe_preserve_order([self._clean_query(item).lower() for item in fallback if self._clean_query(item)])
        return phrases[:limit]

    def _clean_term_list(
        self,
        values: list[Any],
        fallback: list[str],
        *,
        limit: int,
    ) -> list[str]:
        terms: list[str] = []
        for value in values:
            for term in re.findall(r"[a-z0-9]+", str(value).lower()):
                if len(term) <= 2:
                    continue
                terms.append(term)
        terms = self._dedupe_preserve_order(terms)
        if not terms:
            terms = self._dedupe_preserve_order(
                [term for item in fallback for term in re.findall(r"[a-z0-9]+", item.lower()) if len(term) > 2]
            )
        return terms[:limit]

    def _strategy_concept_groups(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[RetrievalConceptGroup]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        groups = [group for group in strategy.concept_groups if group.phrases]
        if groups:
            return groups
        return self._heuristic_retrieval_strategy(topic, plan, answers).concept_groups

    def _result_phrases(self, result: SearchResult) -> list[str]:
        phrases = self._extract_phrases_from_text(result.title)
        if result.abstract:
            phrases.extend(self._extract_phrases_from_text(result.abstract))
        return self._dedupe_preserve_order(phrases)

    def _extract_phrases_from_text(self, text: str, max_phrases: int = 8) -> list[str]:
        cleaned = re.sub(r"[\(\)\[\]\{\}:;,.!?]", " ", text)
        cleaned = re.sub(r"[-/]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        phrases: list[str] = []

        content = self._phrase_terms(cleaned)
        for size in (2, 3, 4):
            for index in range(0, max(0, len(content) - size + 1)):
                phrase = " ".join(content[index : index + size]).strip()
                if phrase:
                    phrases.append(phrase)

        specials = self._ordered_special_terms([text])
        if specials:
            phrases.append(" ".join(specials[: min(3, len(specials))]))

        words = cleaned.split()
        if 2 <= len(words) <= 7:
            phrases.append(cleaned.lower())

        ordered = []
        seen = set()
        for phrase in phrases:
            query = self._clean_query(phrase)
            if not query or query in seen:
                continue
            seen.add(query)
            ordered.append(query)
            if len(ordered) >= max_phrases:
                break
        return ordered

    def _query_priority(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        query: str,
    ) -> tuple[int, int, int]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        selection_terms = set(self._selection_terms(topic, plan, answers))
        generic_terms = set(strategy.generic_terms)
        query_terms = self._terms(query)
        specific_terms = query_terms - generic_terms
        coverage = min(4, len(specific_terms & selection_terms))
        anchor_hits = self._phrase_match_count(strategy.anchor_phrases, specific_terms)
        group_hits = 0
        group_priority = 0
        for group in strategy.concept_groups:
            group_terms = {
                term
                for phrase in group.phrases
                for term in re.findall(r"[a-z0-9]+", phrase.lower())
                if len(term) > 2
            }
            if specific_terms & group_terms:
                group_hits += 1
                group_priority += group.priority
        word_count = len(query.split())
        phrase_bonus = 1 if 3 <= word_count <= 6 else 0
        brevity_bonus = max(0, 7 - word_count)
        return (
            group_priority * 4 + anchor_hits * 4 + coverage + group_hits + phrase_bonus + brevity_bonus,
            group_priority,
            anchor_hits + coverage,
        )

    @staticmethod
    def _phrases_overlap(left: str, right: str) -> bool:
        left_terms = {term for term in left.split() if term}
        right_terms = {term for term in right.split() if term}
        if not left_terms or not right_terms:
            return False
        overlap = len(left_terms & right_terms)
        return overlap >= min(len(left_terms), len(right_terms))

    def _infer_citation_keys(
        self,
        text: str,
        source_notes: list[SourceNote],
        valid_keys: set[str],
        limit: int = 2,
    ) -> list[str]:
        scores = []
        for note in source_notes:
            if note.citation_key not in valid_keys:
                continue
            score = self._score_note_for_text(text, note)
            if score > 0:
                scores.append((score, note.citation_key))
        scores.sort(reverse=True)
        inferred = [cite_key for _, cite_key in scores[:limit]]
        return inferred

    def _score_note_for_text(self, text: str, note: SourceNote) -> int:
        text_terms = self._terms(text)
        if not text_terms:
            return 0
        note_terms = self._terms(
            " ".join(
                [
                    note.title,
                    note.summary,
                    " ".join(note.claims),
                    " ".join(note.related_topics),
                    " ".join(note.evidence_snippets),
                ]
            )
        )
        overlap = len(text_terms & note_terms)
        phrase_bonus = 0
        lowered_text = text.lower()
        for snippet in note.evidence_snippets[:3]:
            if snippet and snippet.lower() in lowered_text:
                phrase_bonus += 3
        return overlap + phrase_bonus

    def _fallback_synthesis_payload(
        self, topic: str, note_by_source: dict[str, SourceNote]
    ) -> dict[str, Any]:
        return {
            "title": topic.title(),
            "abstract": "Automatic synthesis was unavailable, so this report uses collected source notes.",
            "sections": [
                {
                    "heading": "Findings",
                    "paragraphs": [
                        {
                            "text": note.summary,
                            "citation_keys": [note.citation_key] if note.citation_key else [],
                        }
                        for note in note_by_source.values()
                        if note.summary
                    ],
                }
            ],
            "findings": [],
            "notes": [],
            "delete_citation_keys": [],
            "delete_finding_ids": [],
        }

    def _fallback_rewritten_question(self, topic: str, answers: dict[str, str]) -> str:
        objective = answers.get("objective", "").strip()
        audience = answers.get("audience", "").strip()
        constraints = answers.get("constraints", "").strip()
        parts = [topic.strip()]
        if objective:
            parts.append(f"Objective: {objective}")
        if audience:
            parts.append(f"Audience: {audience}")
        if constraints:
            parts.append(f"Constraints: {constraints}")
        return ". ".join(part for part in parts if part)

    def _fallback_queries(self, topic: str) -> list[str]:
        return [
            f"{topic} survey",
            f"{topic} review",
            f"{topic} benchmark",
            f"{topic} arxiv",
        ]

    def _render_report(self, synthesis: SynthesisResult) -> str:
        sections: list[str] = [
            "\\documentclass{article}",
            "\\usepackage{iftex}",
            "\\ifPDFTeX",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{textcomp}",
            "\\else",
            "\\usepackage{fontspec}",
            "\\defaultfontfeatures{Ligatures=TeX}",
            "\\setmainfont{Latin Modern Roman}",
            "\\fi",
            "\\IfFileExists{newunicodechar.sty}{%",
            "\\usepackage{newunicodechar}",
            "\\newunicodechar{≈}{\\ensuremath{\\approx}}",
            "\\newunicodechar{≤}{\\ensuremath{\\leq}}",
            "\\newunicodechar{≥}{\\ensuremath{\\geq}}",
            "\\newunicodechar{±}{\\ensuremath{\\pm}}",
            "\\newunicodechar{×}{\\ensuremath{\\times}}",
            "\\newunicodechar{→}{\\ensuremath{\\rightarrow}}",
            "\\newunicodechar{←}{\\ensuremath{\\leftarrow}}",
            "\\newunicodechar{…}{\\ldots{}}",
            "\\newunicodechar{–}{--}",
            "\\newunicodechar{—}{---}",
            "\\newunicodechar{📚}{}",
            "}{}",
            "\\usepackage{hyperref}",
            "\\usepackage{natbib}",
            "\\title{" + self._escape_latex(synthesis.title) + "}",
            "\\date{}",
            "\\begin{document}",
            "\\maketitle",
        ]
        if synthesis.abstract:
            sections.extend(
                [
                    "\\begin{abstract}",
                    self._escape_latex(synthesis.abstract),
                    "\\end{abstract}",
                ]
            )

        for section in synthesis.sections:
            sections.append("\\section{" + self._escape_latex(section.heading) + "}")
            for paragraph in section.paragraphs:
                text = self._escape_latex(str(paragraph.get("text", "")).strip())
                citation_keys = [key for key in paragraph.get("citation_keys", []) if key]
                if citation_keys:
                    text += " " + "\\cite{" + ",".join(citation_keys) + "}"
                sections.append(text + "\n")

        sections.extend(
            [
                "\\bibliographystyle{plainnat}",
                "\\bibliography{references}",
                "\\end{document}",
            ]
        )
        return "\n".join(sections) + "\n"

    @staticmethod
    def _sanitize_bibtex_entry(entry: str) -> str:
        lines = entry.splitlines()
        if not lines:
            return entry
        sanitized = [lines[0]]
        for line in lines[1:]:
            if "=" not in line:
                sanitized.append(line)
                continue
            prefix, value = line.split("=", 1)
            sanitized.append(prefix + "=" + ResearchPipeline._escape_bibtex_value(value))
        return "\n".join(sanitized)

    @staticmethod
    def _escape_bibtex_value(value: str) -> str:
        escaped = re.sub(r"(?<!\\)_", r"\\_", value)
        escaped = re.sub(r"(?<!\\)&", r"\\&", escaped)
        escaped = re.sub(r"(?<!\\)%", r"\\%", escaped)
        escaped = re.sub(r"(?<!\\)#", r"\\#", escaped)
        return escaped

    @staticmethod
    def _normalize_citation_keys(
        keys: list[str] | tuple[str, ...],
        valid_keys: set[str],
        limit: int = 2,
    ) -> list[str]:
        normalized = []
        seen = set()
        for key in keys:
            item = str(key).strip()
            if not item or item not in valid_keys or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
            if len(normalized) >= limit:
                break
        return normalized

    def _score_result(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result: SearchResult,
    ) -> tuple[int, int, int, int]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        context_terms = set(self._selection_terms(topic, plan, answers))
        generic_terms = set(strategy.generic_terms)
        result_terms = self._result_terms(result)
        title_terms = self._terms(result.title)
        title_overlap = len(context_terms & title_terms)
        body_overlap = len(context_terms & result_terms)
        specific_title_overlap = len((context_terms - generic_terms) & title_terms)
        specific_body_overlap = len((context_terms - generic_terms) & result_terms)
        query_hits = max(1, len(result.matched_queries))
        anchor_hits = self._phrase_match_count(self._anchor_phrases(topic, plan, answers), result_terms)
        facet_hits = self._phrase_match_count(self._facet_phrases(topic, plan, answers), result_terms)
        group_hits = self._matched_concept_group_count(topic, plan, answers, result_terms)
        group_priority = self._matched_concept_group_priority(topic, plan, answers, result_terms)

        paper_bonus = 9 if result.kind == "paper" else 0
        backend_bonus = {
            "semantic_scholar": 7,
            "arxiv": 6,
            "crossref": 2,
            "google_cse": 1,
            "serpapi": 1,
            "duckduckgo": 0,
        }.get(result.backend, 0)
        abstract_bonus = 3 if result.abstract else 0
        arxiv_bonus = 2 if result.arxiv_id else 0
        doi_bonus = 1 if result.doi else 0
        citation_bonus = min(4, result.citation_count // 50) if result.citation_count > 0 else 0
        score = paper_bonus + backend_bonus + abstract_bonus + arxiv_bonus + doi_bonus + citation_bonus
        score += self._critic_bonus(result)
        score += title_overlap * 2
        score += body_overlap
        score += specific_title_overlap * 4
        score += specific_body_overlap * 3
        score += query_hits * 4
        score += anchor_hits * 4
        score += facet_hits * 2
        score += group_priority * 3
        if group_hits >= 2:
            score += 10
        elif group_hits == 1:
            score += 3

        if self._covers_multiple_anchor_groups(topic, plan, answers, result):
            score += 8

        score -= self._generic_page_penalty(topic, plan, answers, result)
        score -= self._thin_result_penalty(result)
        return (score, anchor_hits, query_hits, specific_title_overlap)

    @staticmethod
    def _critic_bonus(result: SearchResult) -> int:
        if result.critic_relevant is True:
            return 14
        if result.critic_relevant is False:
            return -18
        return 0

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2}

    def _build_static_queries(
        self,
        topic: str,
        answers: dict[str, str],
        rewritten_question: str,
        related_topics: list[str],
        focus_areas: list[str],
        must_cover: list[str],
    ) -> list[str]:
        draft_plan = ResearchPlan(
            queries=[],
            related_topics=list(related_topics),
            focus_areas=list(focus_areas),
            rewritten_question=rewritten_question,
            must_cover=list(must_cover),
            source_requirements=[],
        )
        primary_texts = [
            topic,
            " ".join(related_topics),
            " ".join(focus_areas),
            " ".join(must_cover),
        ]
        secondary_texts = [
            rewritten_question,
            answers.get("objective", ""),
            answers.get("constraints", ""),
        ]
        texts = primary_texts + secondary_texts
        strategy = self._get_retrieval_strategy(topic, draft_plan, answers, allow_model=True)
        anchor_phrases = self._anchor_phrases(topic, draft_plan, answers)
        facet_phrases = self._facet_phrases(topic, draft_plan, answers)
        special_terms = self._ordered_special_terms(texts)
        primary_terms = self._ordered_terms(" ".join(primary_texts))
        secondary_terms = self._ordered_terms(" ".join(secondary_texts))
        context_terms = self._dedupe_preserve_order(primary_terms + secondary_terms)
        topic_terms = self._ordered_terms(topic)
        core_terms = self._dedupe_preserve_order(
            self._ordered_special_terms([topic]) + special_terms + topic_terms + primary_terms + secondary_terms
        )[:6]
        platform_terms = [term for term in special_terms if any(char.isdigit() for char in term)]
        domain_terms = [
            term for term in context_terms if term not in platform_terms and term not in {"simulation", "simulating", "model", "modeling"}
        ]

        queries: list[str] = []

        def add(query: str) -> None:
            cleaned = self._clean_query(query)
            if cleaned:
                queries.append(cleaned)

        if core_terms:
            add(" ".join(core_terms))
            add(" ".join(core_terms[:4]) + " survey")
            add(" ".join(core_terms[:4]) + " review")
            add(" ".join(core_terms[:4]) + " benchmark")
            for facet in facet_phrases[:2]:
                add(" ".join(core_terms[:4]) + f" {facet}")

        if topic_terms:
            add(" ".join(topic_terms[:4]))
            add(" ".join(topic_terms[:4]) + " benchmark")
            add(" ".join(topic_terms[:4]) + " survey")

        for anchor in anchor_phrases[:4]:
            add(anchor)
            for facet in facet_phrases[:3]:
                if not self._phrases_overlap(anchor, facet):
                    add(f"{anchor} {facet}")

        groups = [group for group in strategy.concept_groups if group.phrases]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                for left_phrase in left.phrases[:2]:
                    for right_phrase in right.phrases[:2]:
                        if self._phrases_overlap(left_phrase, right_phrase):
                            continue
                        add(f"{left_phrase} {right_phrase}")
                        for facet in facet_phrases[:2]:
                            if not self._phrases_overlap(right_phrase, facet):
                                add(f"{left_phrase} {right_phrase} {facet}")

        for item in must_cover[:2] + related_topics[:2] + focus_areas[:1]:
            phrase_terms = self._ordered_terms(item)[:6]
            if phrase_terms:
                add(" ".join(core_terms[:3] + phrase_terms))

        return queries

    @staticmethod
    def _clean_query(value: Any) -> str:
        original = str(value).strip()
        if not original:
            return ""

        query = re.sub(r"\$[^$]*\$", " ", original)
        query = re.sub(r"\\[a-zA-Z]+", " ", query)
        query = re.sub(r"[“”\"'`]", "", query)
        query = re.sub(r"[-/]", " ", query)
        query = re.sub(r"[\(\)\[\]\{\}:;,!?]", " ", query)
        query = re.sub(r"\s+", " ", query).strip()

        words = query.split()
        if len(words) <= 12 and "?" not in original:
            return query

        stopwords = {
            "about",
            "against",
            "and",
            "any",
            "are",
            "art",
            "capabilities",
            "capability",
            "comparable",
            "current",
            "compare",
            "concerning",
            "for",
            "from",
            "how",
            "into",
            "limits",
            "more",
            "question",
            "regarding",
            "should",
            "state",
            "that",
            "the",
            "their",
            "them",
            "these",
            "this",
            "through",
            "what",
            "when",
            "where",
            "which",
            "with",
        }
        terms = []
        seen = set()
        for term in re.findall(r"[a-z0-9]+", query.lower()):
            if len(term) <= 2 or term in stopwords or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= 10:
                break
        return " ".join(terms) if terms else query[:180]

    def _selection_terms(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        strategy = self._get_retrieval_strategy(topic, plan, answers)
        ordered = self._ordered_terms(
            " ".join(
                [
                    topic,
                    plan.rewritten_question,
                    " ".join(plan.must_cover),
                    " ".join(plan.related_topics),
                    " ".join(plan.focus_areas),
                    " ".join(answers.values()),
                ]
            )
        )
        phrases = list(strategy.anchor_phrases)
        for group in strategy.concept_groups:
            phrases.extend(group.phrases)
        phrase_terms = [
            term
            for phrase in phrases
            for term in re.findall(r"[a-z0-9]+", phrase.lower())
            if len(term) > 2
        ]
        special = self._ordered_special_terms(
            [topic, plan.rewritten_question, " ".join(plan.must_cover), " ".join(answers.values())]
        )
        generic_terms = set(strategy.generic_terms)
        filtered_ordered = [term for term in ordered if term not in generic_terms]
        filtered_special = [term for term in special if term not in generic_terms]
        return self._dedupe_preserve_order(filtered_special + phrase_terms + filtered_ordered[:18])

    def _domain_anchor_phrases(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
    ) -> list[str]:
        generic_terms = set(self._get_retrieval_strategy(topic, plan, answers).generic_terms)
        phrases: list[str] = []
        for phrase in self._anchor_phrases(topic, plan, answers):
            terms = set(phrase.split())
            if terms and not terms <= generic_terms:
                phrases.append(phrase)
        return phrases[:8]

    def _result_terms(self, result: SearchResult) -> set[str]:
        return self._terms(
            " ".join(
                [
                    result.title,
                    result.snippet,
                    result.abstract,
                    " ".join(result.authors),
                    result.url,
                ]
            )
        )

    def _covers_multiple_anchor_groups(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result: SearchResult,
    ) -> bool:
        return self._matched_concept_group_count(
            topic,
            plan,
            answers,
            self._result_terms(result),
        ) >= 2

    def _generic_page_penalty(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result: SearchResult,
    ) -> int:
        domain = self._domain(result.url)
        terms = self._result_terms(result)
        selection_terms = set(self._selection_terms(topic, plan, answers))
        generic_terms = set(self._get_retrieval_strategy(topic, plan, answers).generic_terms)
        overlap = len(terms & selection_terms)
        specific_overlap = len(terms & (selection_terms - generic_terms))
        title = result.title.lower()
        penalty = 0
        if title.startswith("github -") or domain in {"github.com", "gitlab.com"}:
            penalty += 8 if specific_overlap < 3 else 3
        if domain in {"medium.com"}:
            penalty += 3
        if "redirect" in result.snippet.lower() or "redirect" in result.title.lower():
            penalty += 6
        if result.kind == "web" and overlap > 0 and specific_overlap == 0:
            penalty += 4
        return penalty

    @staticmethod
    def _thin_result_penalty(result: SearchResult) -> int:
        penalty = 0
        if result.kind == "paper" and result.backend == "crossref" and not result.abstract:
            penalty += 4
        if result.kind == "web" and len((result.snippet or "").strip()) < 80:
            penalty += 2
        return penalty

    @staticmethod
    def _phrase_match_count(phrases: list[str], result_terms: set[str]) -> int:
        hits = 0
        for phrase in phrases:
            phrase_terms = {term for term in phrase.split() if term}
            if not phrase_terms:
                continue
            required = 1 if len(phrase_terms) <= 2 else 2
            if len(phrase_terms & result_terms) >= required:
                hits += 1
        return hits

    def _matched_concept_group_count(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result_terms: set[str],
    ) -> int:
        return sum(
            1
            for group in self._strategy_concept_groups(topic, plan, answers)
            if self._phrase_match_count(group.phrases, result_terms) > 0
        )

    def _matched_concept_group_priority(
        self,
        topic: str,
        plan: ResearchPlan,
        answers: dict[str, str],
        result_terms: set[str],
    ) -> int:
        total = 0
        for group in self._strategy_concept_groups(topic, plan, answers):
            if self._phrase_match_count(group.phrases, result_terms) > 0:
                total += group.priority
        return total

    @staticmethod
    def _special_terms(texts: list[str]) -> set[str]:
        seen = set()
        for text in texts:
            for match in re.findall(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Za-z]+[0-9]+)\b", text):
                lowered = re.sub(r"[^a-z0-9]+", "", match.lower())
                if lowered:
                    seen.add(lowered)
        return seen

    def _ordered_special_terms(self, texts: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for match in re.findall(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Za-z]+[0-9]+)\b", text):
                lowered = re.sub(r"[^a-z0-9]+", "", match.lower())
                if lowered not in seen:
                    seen.add(lowered)
                    ordered.append(lowered)
        return ordered

    @staticmethod
    def _ordered_terms(value: str) -> list[str]:
        stopwords = {
            "and",
            "are",
            "about",
            "against",
            "among",
            "analysis",
            "architecture",
            "audience",
            "based",
            "can",
            "compare",
            "comparison",
            "constraints",
            "current",
            "cover",
            "developer",
            "developers",
            "detailed",
            "digital",
            "electronics",
            "extension",
            "extensions",
            "feasibility",
            "focus",
            "for",
            "generic",
            "blog",
            "blogs",
            "how",
            "include",
            "including",
            "in",
            "into",
            "implementation",
            "is",
            "level",
            "logic",
            "model",
            "modeling",
            "objective",
            "outline",
            "oriented",
            "over",
            "page",
            "pages",
            "paper",
            "papers",
            "prioritize",
            "question",
            "related",
            "required",
            "research",
            "required",
            "researchers",
            "roadmap",
            "assess",
            "being",
            "news",
            "real",
            "should",
            "simulation",
            "simulator",
            "specific",
            "source",
            "sources",
            "state",
            "strategy",
            "study",
            "support",
            "survey",
            "systems",
            "targeting",
            "the",
            "to",
            "today",
            "stack",
            "used",
            "using",
            "via",
            "web",
            "what",
            "whether",
            "with",
            "within",
        }
        ordered: list[str] = []
        seen: set[str] = set()
        for term in re.findall(r"[a-z0-9]+", value.lower()):
            if len(term) <= 2 or term in stopwords or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered

    @staticmethod
    def _phrase_terms(value: str) -> list[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "assess",
            "be",
            "but",
            "by",
            "can",
            "cover",
            "both",
            "for",
            "from",
            "generic",
            "being",
            "how",
            "including",
            "in",
            "into",
            "is",
            "level",
            "of",
            "on",
            "or",
            "over",
            "page",
            "pages",
            "paper",
            "papers",
            "prioritize",
            "real",
            "required",
            "research",
            "stack",
            "support",
            "survey",
            "that",
            "the",
            "their",
            "to",
            "used",
            "using",
            "via",
            "web",
            "what",
            "whether",
            "with",
        }
        ordered: list[str] = []
        seen: set[tuple[str, int]] = set()
        terms = re.findall(r"[a-z0-9]+", value.lower())
        for index, term in enumerate(terms):
            if len(term) <= 2 or term in stopwords:
                continue
            key = (term, index)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(term)
        return ordered

    @staticmethod
    def _strip_inline_citation_markers(text: str, valid_keys: set[str]) -> str:
        cleaned = re.sub(r"\\cite[t|p]?{[^}]+}", " ", text)
        for key in sorted(valid_keys, key=len, reverse=True):
            cleaned = re.sub(rf"\[\s*@?{re.escape(key)}\s*\]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ,;")

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc.lower().lstrip("www.")

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen = set()
        output = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                output.append(item)
        return output

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "answer"

    @staticmethod
    def _escape_latex(value: str) -> str:
        unicode_replacements = {
            "\u00a0": " ",
            "\u2009": " ",
            "\u200b": "",
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "--",
            "\u2013": "--",
            "\u2014": "---",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2022": "*",
            "\u2026": "...",
            "\u2192": "->",
            "\u2190": "<-",
            "\u2212": "-",
            "\u2264": "<=",
            "\u2265": ">=",
            "\u00d7": "x",
            "\u00b1": "+/-",
            "\u2248": "~",
        }
        replacements = {
            "\\": "\\textbackslash{}",
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        normalized = unicodedata.normalize("NFKC", value)
        cleaned: list[str] = []
        for char in normalized:
            if char in unicode_replacements:
                cleaned.append(unicode_replacements[char])
                continue
            category = unicodedata.category(char)
            if ord(char) > 0xFFFF or category in {"Cs", "Co", "Cn"}:
                continue
            if category == "So":
                continue
            cleaned.append(char)
        return "".join(replacements.get(char, char) for char in "".join(cleaned))

    def _latex_failure_detail(self, report_path: Path) -> str:
        log_path = report_path.with_suffix(".log")
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("!"):
                continue
            detail = stripped.lstrip("!").strip()
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line and not next_line.startswith("See the LaTeX manual"):
                    detail = f"{detail} {next_line}".strip()
            return detail
        return ""

    @staticmethod
    def _combine_failure_message(base: str, detail: str) -> str:
        if not detail:
            return base
        return f"{base} | {detail}"

    @staticmethod
    def _preferred_latex_engine() -> Path | None:
        for name in ("lualatex", "xelatex", "pdflatex"):
            binary = shutil.which(name)
            if binary:
                return Path(binary)
        return None
