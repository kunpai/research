from __future__ import annotations

import json
import subprocess
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deep_research_ollama.config import Settings
from deep_research_ollama.models import (
    CitationRecord,
    CollaborationSession,
    ResearchPlan,
    SearchResult,
    SourceDocument,
    SourceNote,
)
from deep_research_ollama.ollama import OllamaError
from deep_research_ollama.pipeline import ResearchPipeline


class FakeOllama:
    def __init__(self, payload: dict | None = None, *, should_fail: bool = False) -> None:
        self.payload = payload or {}
        self.should_fail = should_fail

    def chat_json(self, system: str, user: str, **_: object) -> dict:
        if self.should_fail:
            raise OllamaError("forced failure")
        return self.payload


class SequenceOllama:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def chat_json(self, system: str, user: str, **_: object) -> dict:
        self.calls += 1
        if not self.payloads:
            raise AssertionError("unexpected extra model call")
        return self.payloads.pop(0)


class RoleAwareOllama:
    def __init__(self, mapping: dict[str, dict]) -> None:
        self.mapping = mapping

    def chat_json(self, system: str, user: str, **_: object) -> dict:
        for marker, payload in self.mapping.items():
            if marker in system:
                return payload
        raise AssertionError(f"unexpected model call: {system[:120]}")


class FallbackCriticOllama:
    def __init__(self, batch_text: str, single_text: dict[str, str] | None = None) -> None:
        self.batch_text = batch_text
        self.single_text = single_text or {}

    def chat_json(self, system: str, user: str, **_: object) -> dict:
        raise OllamaError("response_format unsupported")

    def chat_text(self, system: str, user: str, **_: object) -> str:
        if "Return one line per candidate" in system:
            return self.batch_text
        if "Return exactly one tab-separated line" in system:
            for result_id, payload in self.single_text.items():
                if result_id in user:
                    return payload
            return ""
        raise AssertionError(f"unexpected text call: {system[:120]}")


class FakeSearch:
    def __init__(self, mapping: dict[str, list[SearchResult]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def search(self, queries: list[str]) -> list[SearchResult]:
        assert len(queries) == 1
        query = queries[0]
        self.calls.append(query)
        results = []
        for item in self.mapping.get(query, []):
            data = item.to_dict()
            data["matched_queries"] = list(item.matched_queries) or [query]
            results.append(SearchResult(**data))
        return results


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, **settings_kwargs: object) -> tuple[ResearchPipeline, TemporaryDirectory[str]]:
        tempdir = TemporaryDirectory()
        pipeline = ResearchPipeline(Settings(**settings_kwargs), Path(tempdir.name))
        pipeline.constitution.load("test topic")
        return pipeline, tempdir

    def test_build_plan_fallback_creates_research_brief(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=5)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = pipeline.build_plan(
            "retrieval-augmented generation",
            {"objective": "compare architectures", "audience": "ML engineers"},
        )

        self.assertIn("retrieval-augmented generation", plan.rewritten_question.lower())
        self.assertGreaterEqual(len(plan.queries), 3)
        self.assertTrue(any("survey" in query.lower() or "review" in query.lower() for query in plan.queries))
        self.assertTrue(plan.source_requirements)

    def test_no_clarify_skips_model_question_generation(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        class FailIfCalled:
            def chat_json(self, system: str, user: str, **_: object) -> dict:
                raise AssertionError("clarifier model should not be called")

        pipeline.ollama = FailIfCalled()

        answers = pipeline.ask_clarifying_questions(
            "transformer scaling laws",
            prefilled_answers={"objective": "summarize findings"},
            interactive=False,
        )

        self.assertEqual(answers, {"objective": "summarize findings"})

    def test_run_resumes_from_summarizing_checkpoint(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            compile_latex=False,
            max_summary_model_calls=2,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        selected = SearchResult(
            result_id="paper-1",
            title="BioCoder",
            url="https://example.com/biocoder",
            snippet="benchmark",
            backend="arxiv",
            kind="paper",
        )
        document = SourceDocument(
            source_id="paper-1",
            title="BioCoder",
            url="https://example.com/biocoder",
            kind="paper",
            backend="arxiv",
            abstract="BioCoder benchmark abstract.",
            text_chunks=["BioCoder benchmark abstract."],
        )
        citation = CitationRecord(
            cite_key="Tang_2024",
            bibtex="@article{Tang_2024, title={BioCoder}}",
            title="BioCoder",
            url="https://example.com/biocoder",
            source_id="paper-1",
        )
        note = SourceNote(
            source_id="paper-1",
            title="BioCoder",
            url="https://example.com/biocoder",
            citation_key="Tang_2024",
            summary="BioCoder is a benchmark for bioinformatics code generation.",
        )
        plan = ResearchPlan(
            queries=["biocoder"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize BioCoder.",
            must_cover=[],
            source_requirements=[],
        )

        pipeline.constitution.checkpoint_sources(
            [note],
            [citation],
            checkpoint_stage="summarizing",
        )
        checkpoint = {
            "status": "summarizing",
            "topic": "BioCoder",
            "answers": {"objective": "summarize BioCoder"},
            "plan": plan.to_dict(),
            "retrieval": {
                "selected_sources": [{"result_id": "paper-1"}],
                "budget": {"max_summary_model_calls": 2},
            },
            "selected_sources": [selected.to_dict()],
            "documents": [document.to_dict()],
            "citations": [citation.to_dict()],
            "source_notes": [note.to_dict()],
            "compiled_pdf": "",
        }
        (pipeline.output_dir / pipeline.settings.run_filename).write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        with (
            patch.object(pipeline, "build_plan", side_effect=AssertionError("build_plan should not be called")),
            patch.object(
                pipeline,
                "retrieve_results",
                side_effect=AssertionError("retrieve_results should not be called"),
            ),
            patch.object(
                pipeline,
                "select_results",
                side_effect=AssertionError("select_results should not be called"),
            ),
            patch.object(
                pipeline.search,
                "fetch_document",
                side_effect=AssertionError("fetch_document should not be called"),
            ),
            patch.object(
                pipeline,
                "resolve_citations",
                side_effect=AssertionError("resolve_citations should not be called"),
            ),
            patch.object(
                pipeline,
                "summarize_documents",
                side_effect=AssertionError("summarize_documents should not be called"),
            ),
        ):
            result = pipeline.run(
                "BioCoder",
                interactive=False,
                initial_answers={"audience": "researchers"},
            )

        self.assertEqual(result["plan"]["rewritten_question"], "Summarize BioCoder.")
        self.assertEqual(result["answers"]["audience"], "researchers")
        run_payload = json.loads(
            (pipeline.output_dir / pipeline.settings.run_filename).read_text(encoding="utf-8")
        )
        constitution_payload = json.loads(
            (pipeline.output_dir / pipeline.settings.constitution_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(run_payload["status"], "completed")
        self.assertEqual(len(run_payload["source_notes"]), 1)
        self.assertEqual(constitution_payload["metadata"]["resume_count"], 1)
        self.assertEqual(constitution_payload["metadata"]["resume_from_status"], "summarizing")

    def test_run_resume_reapplies_budget_and_backfills_missing_citations(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            compile_latex=False,
            max_summary_model_calls=2,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        selected = [
            SearchResult(
                result_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                snippet="benchmark",
                backend="arxiv",
                kind="paper",
            ),
            SearchResult(
                result_id="paper-2",
                title="BioCoder Followup",
                url="https://example.com/biocoder-2",
                snippet="evaluation",
                backend="crossref",
                kind="paper",
            ),
        ]
        documents = {
            "paper-1": SourceDocument(
                source_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                kind="paper",
                backend="arxiv",
                abstract="BioCoder benchmark abstract.",
                text_chunks=["BioCoder benchmark abstract."],
            ),
            "paper-2": SourceDocument(
                source_id="paper-2",
                title="BioCoder Followup",
                url="https://example.com/biocoder-2",
                kind="paper",
                backend="crossref",
                abstract="BioCoder followup abstract.",
                text_chunks=["BioCoder followup abstract."],
            ),
        }
        citation_one = CitationRecord(
            cite_key="Tang_2024",
            bibtex="@article{Tang_2024, title={BioCoder}}",
            title="BioCoder",
            url="https://example.com/biocoder",
            source_id="paper-1",
        )
        citation_two = CitationRecord(
            cite_key="Lee_2025",
            bibtex="@article{Lee_2025, title={BioCoder Followup}}",
            title="BioCoder Followup",
            url="https://example.com/biocoder-2",
            source_id="paper-2",
        )
        note = SourceNote(
            source_id="paper-1",
            title="BioCoder",
            url="https://example.com/biocoder",
            citation_key="Tang_2024",
            summary="BioCoder is a benchmark for bioinformatics code generation.",
        )
        plan = ResearchPlan(
            queries=["biocoder"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize BioCoder and followup work.",
            must_cover=[],
            source_requirements=[],
        )

        checkpoint = {
            "status": "summarizing",
            "topic": "BioCoder",
            "answers": {"objective": "summarize BioCoder"},
            "plan": plan.to_dict(),
            "retrieval": {
                "selected_sources": [selected[0].to_dict()],
                "selected_sources_pre_budget": [item.to_dict() for item in selected],
                "budget": {"max_summary_model_calls": 1},
            },
            "selected_sources": [selected[0].to_dict()],
            "documents": [documents["paper-1"].to_dict()],
            "citations": [citation_one.to_dict()],
            "source_notes": [note.to_dict()],
            "compiled_pdf": "",
        }
        (pipeline.output_dir / pipeline.settings.run_filename).write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        resolved_calls: list[list[str]] = []

        def fake_fetch_document(result: SearchResult) -> SourceDocument:
            return documents[result.result_id]

        def fake_resolve_citations(
            pending_documents: list[SourceDocument],
            constitution: dict[str, object],
        ) -> list[CitationRecord]:
            del constitution
            resolved_calls.append([item.source_id for item in pending_documents])
            self.assertEqual([item.source_id for item in pending_documents], ["paper-2"])
            return [citation_two]

        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "BioCoder",
                "abstract": "Summary.",
                "sections": [
                    {
                        "heading": "Overview",
                        "paragraphs": [
                            {
                                "text": "BioCoder and followup work improve code-generation evaluation.",
                                "citation_keys": ["Tang_2024", "Lee_2025"],
                            }
                        ],
                    }
                ],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "BioCoder",
        )

        with (
            patch.object(
                pipeline,
                "build_plan",
                side_effect=AssertionError("build_plan should not be called"),
            ),
            patch.object(
                pipeline,
                "retrieve_results",
                side_effect=AssertionError("retrieve_results should not be called"),
            ),
            patch.object(
                pipeline,
                "select_results",
                side_effect=AssertionError("select_results should not be called"),
            ),
            patch.object(pipeline.search, "fetch_document", side_effect=fake_fetch_document),
            patch.object(pipeline, "resolve_citations", side_effect=fake_resolve_citations),
            patch.object(pipeline, "synthesize", return_value=synthesis),
        ):
            result = pipeline.run(
                "BioCoder",
                interactive=False,
                initial_answers={"audience": "researchers"},
            )

        self.assertEqual(
            [item["result_id"] for item in result["selected_sources"]],
            ["paper-1", "paper-2"],
        )
        self.assertEqual(result["retrieval"]["budget"]["max_summary_model_calls"], 2)
        self.assertEqual(resolved_calls, [["paper-2"]])

        run_payload = json.loads(
            (pipeline.output_dir / pipeline.settings.run_filename).read_text(encoding="utf-8")
        )
        self.assertEqual(run_payload["retrieval"]["budget"]["max_summary_model_calls"], 2)
        self.assertEqual(
            [item["source_id"] for item in run_payload["citations"]],
            ["paper-1", "paper-2"],
        )
        self.assertEqual(
            [item["source_id"] for item in run_payload["source_notes"]],
            ["paper-1", "paper-2"],
        )

    def test_build_plan_keeps_raw_topic_when_query_budget_is_one(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=1)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(
            {
                "rewritten_question": (
                    "What are the empirically established scaling laws for Transformer model "
                    "performance concerning model size ($N$), dataset size ($D$), and "
                    "compute budget ($C$)?"
                ),
                "queries": [],
                "related_topics": ["Chinchilla scaling laws"],
                "focus_areas": [],
                "must_cover": [],
                "source_requirements": [],
            }
        )

        plan = pipeline.build_plan("transformer scaling laws", {})

        self.assertEqual(plan.queries, ["transformer scaling laws"])

    def test_build_plan_adds_static_queries_for_niche_terms(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=10)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = pipeline.build_plan(
            "superconducting electronic simulation in gem5",
            {
                "constraints": (
                    "focus on SFQ RSFQ ERSFQ AQFP and Josephson-junction digital electronics"
                )
            },
        )

        joined_queries = " | ".join(plan.queries)
        self.assertIn("superconducting electronic simulation in gem5", plan.queries)
        self.assertIn("gem5", joined_queries)
        self.assertTrue(any("sfq" in query for query in plan.queries))
        self.assertTrue(any("review" in query or "survey" in query for query in plan.queries))
        self.assertTrue(
            any(
                "architecture simulation" in query or "computer architecture" in query
                for query in plan.queries
            )
        )

    def test_build_plan_filters_meta_instruction_terms_from_queries(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=6)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = pipeline.build_plan(
            "superconducting electronic simulation in gem5",
            {
                "objective": (
                    "assess whether gem5 can support architecture-level research on "
                    "superconducting digital systems and what modeling stack is required"
                ),
                "constraints": (
                    "prioritize papers over generic web pages and cover superconducting "
                    "architecture, SFQ or AQFP style logic, full-system modeling, and "
                    "gem5 or similar simulators"
                ),
            },
        )

        joined_queries = " | ".join(plan.queries)
        self.assertNotIn("research stack", joined_queries)
        self.assertNotIn("prioritize papers", joined_queries)
        self.assertNotIn(" art ", f" {joined_queries} ")
        self.assertTrue(any("superconducting" in query for query in plan.queries))

    def test_build_plan_broad_topic_uses_domain_clauses_not_architecture_noise(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=8)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = pipeline.build_plan(
            "AI for mathematics",
            {
                "objective": (
                    "survey how AI systems are being used for mathematics, including theorem "
                    "proving, olympiad problem solving, symbolic reasoning, and formal verification"
                ),
                "constraints": (
                    "prioritize papers and benchmark-oriented sources over generic blogs, and "
                    "include both general AI-for-math systems and formal math tools"
                ),
            },
        )

        joined_queries = " | ".join(plan.queries).lower()
        self.assertIn("theorem", joined_queries)
        self.assertNotIn("architecture simulation", joined_queries)
        self.assertNotIn("survey how being used", joined_queries)

    def test_bioinformatics_strategy_does_not_confuse_biological_with_logic(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=8)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        answers = {
            "objective": (
                "survey how AI is being used for bioinformatics coding and computational "
                "biology workflows, including code generation, sequence analysis, genomics "
                "pipelines, protein modeling, and agentic assistants for biological data analysis"
            ),
            "constraints": (
                "prioritize papers, benchmarks, and real bioinformatics toolchains over "
                "generic biotech or AI news"
            ),
        }
        plan = pipeline.build_plan("AI for bioinformatics code", answers)
        strategy = pipeline._get_retrieval_strategy(
            "AI for bioinformatics code",
            plan,
            answers,
        )

        joined_queries = " | ".join(plan.queries).lower()
        self.assertNotIn("behavioral hdl", strategy.search_facets)
        self.assertNotIn("logic circuits", strategy.search_facets)
        self.assertNotIn("digital systems", strategy.search_facets)
        self.assertNotIn("architecture", joined_queries)
        self.assertNotIn("simulation", joined_queries)
        self.assertTrue(any("bioinformatics" in query for query in plan.queries))
        self.assertIn("bioinformatics code benchmark", joined_queries)
        self.assertNotIn("find benchmarks practical", joined_queries)

    def test_select_results_enforces_diverse_mix(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            max_selected_sources=4,
            min_papers=2,
            min_web_sources=1,
            max_sources_per_backend=3,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = ResearchPlan(
            queries=["transformer scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Compare transformer scaling law evidence across papers and commentary.",
            must_cover=[],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id=f"web{i}",
                title="Transformer scaling blog analysis",
                url=f"https://blog{i}.example.com/post",
                snippet="commentary on transformer scaling laws",
                backend="duckduckgo",
                kind="web",
            )
            for i in range(5)
        ] + [
            SearchResult(
                result_id="paper1",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                snippet="paper",
                backend="arxiv",
                kind="paper",
                authors=["Kaplan"],
                year="2020",
                arxiv_id="2001.08361",
            ),
            SearchResult(
                result_id="paper2",
                title="Chinchilla scaling laws",
                url="https://example.org/chinchilla",
                snippet="paper",
                backend="crossref",
                kind="paper",
                authors=["Hoffmann"],
                year="2022",
                doi="10.0000/example",
            ),
        ]

        selected = pipeline.select_results("transformer scaling laws", plan, {}, results)

        self.assertEqual(len(selected), 4)
        self.assertGreaterEqual(sum(1 for item in selected if item.kind == "paper"), 2)
        self.assertGreaterEqual(sum(1 for item in selected if item.kind == "web"), 1)

    def test_select_results_prefers_arxiv_over_thin_crossref_match(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = ResearchPlan(
            queries=["transformer scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="paper-arxiv",
                title="Scaling Laws for Downstream Task Performance of Large Language Models",
                url="https://arxiv.org/abs/2402.04177",
                snippet="transformer scaling law evidence",
                backend="arxiv",
                kind="paper",
                authors=["Example"],
                year="2024",
                arxiv_id="2402.04177",
                abstract="Empirical study of scaling laws for language models.",
            ),
            SearchResult(
                result_id="paper-crossref",
                title="Scaling Laws in Generative AI: How Model Size and Data Influence Performance and Cost",
                url="https://doi.org/10.21275/example",
                snippet="model size and data influence performance",
                backend="crossref",
                kind="paper",
                authors=["Example"],
                year="2025",
                doi="10.21275/example",
                abstract="",
            ),
        ]

        selected = pipeline.select_results("transformer scaling laws", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["paper-arxiv"])

    def test_select_results_prefers_multi_query_paper_hits(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1)
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["superconducting gem5", "gem5 sfq", "sfq logic simulation"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Assess gem5 support for SFQ logic simulation.",
            must_cover=["gem5 support", "SFQ logic"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="paper-strong",
                title="Modeling SFQ Digital Logic for Architecture Simulation",
                url="https://arxiv.org/abs/2401.00001",
                snippet="gem5-compatible architecture simulation model for SFQ logic",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2401.00001",
                abstract="An architecture-facing model for SFQ digital logic.",
                matched_queries=["superconducting gem5", "gem5 sfq", "sfq logic simulation"],
            ),
            SearchResult(
                result_id="paper-weak",
                title="The gem5 Simulator: Version 20.0+",
                url="https://arxiv.org/abs/2007.03152",
                snippet="general architecture simulation",
                backend="arxiv",
                kind="paper",
                year="2020",
                arxiv_id="2007.03152",
                abstract="General gem5 paper.",
                matched_queries=["superconducting gem5"],
            ),
        ]

        selected = pipeline.select_results(
            "superconducting electronic simulation in gem5",
            plan,
            {"constraints": "SFQ RSFQ AQFP"},
            results,
        )

        self.assertEqual([item.result_id for item in selected], ["paper-strong"])

    def test_select_results_uses_critic_to_demote_tangential_match(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1, max_critic_results=2)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "CriticAgent": {
                    "judgments": [
                        {
                            "result_id": "generic-paper",
                            "relevant": False,
                            "reason": "Matches broad hardware and AI terms but not LLM-driven chip-design workflows.",
                        },
                        {
                            "result_id": "llm-paper",
                            "relevant": True,
                            "reason": "Directly studies LLMs bridging software and hardware design.",
                        },
                    ]
                }
            }
        )

        plan = ResearchPlan(
            queries=["AI for hardware chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="What work exists on using LLMs for hardware chip design?",
            must_cover=["LLMs", "chip design"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="generic-paper",
                title="Hardware-Software Co-Design for On-Chip Learning in AI Systems",
                url="https://doi.org/10.1145/example-generic",
                snippet="Hardware-software co-design for AI systems.",
                backend="crossref",
                kind="paper",
                year="2023",
                doi="10.1145/example-generic",
                abstract="A highly cited paper on on-chip learning and AI hardware systems.",
                citation_count=600,
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="llm-paper",
                title="C2HLSC: Leveraging Large Language Models to Bridge the Software-to-Hardware Design Gap",
                url="https://arxiv.org/abs/2412.00214v2",
                snippet="Large language models for hardware design generation.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2412.00214v2",
                abstract="Uses LLMs to translate software descriptions into hardware design artifacts.",
                matched_queries=["AI for hardware chip design"],
            ),
        ]

        selected = pipeline.select_results("AI for hardware chip design", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["llm-paper"])
        self.assertFalse(results[0].critic_relevant)
        self.assertTrue(results[1].critic_relevant)
        self.assertIn("LLM", results[1].critic_reason)

    def test_select_results_excludes_critic_rejected_sources_from_diversity_fill(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            max_selected_sources=2,
            min_papers=2,
            min_web_sources=0,
            max_critic_results=3,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "CriticAgent": {
                    "judgments": [
                        {
                            "result_id": "rejected-paper",
                            "relevant": False,
                            "reason": "Generic medicine review, not bioinformatics code generation.",
                        },
                        {
                            "result_id": "biocoder-paper",
                            "relevant": True,
                            "reason": "Direct benchmark for bioinformatics code generation.",
                        },
                    ]
                }
            }
        )

        plan = ResearchPlan(
            queries=["AI for bioinformatics code"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Find benchmarks and systems for AI in bioinformatics code generation.",
            must_cover=["bioinformatics", "code generation"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="rejected-paper",
                title="Custom Large Language Models Improve Accuracy for Evidence-Based Medicine",
                url="https://doi.org/10.1016/example-medicine",
                snippet="Medical evidence retrieval with custom large language models.",
                backend="crossref",
                kind="paper",
                year="2025",
                doi="10.1016/example-medicine",
                abstract="Clinical guidance retrieval with large language models.",
                citation_count=200,
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="biocoder-paper",
                title="BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models",
                url="https://arxiv.org/abs/2404.12345",
                snippet="Benchmark for bioinformatics code generation.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2404.12345",
                abstract="Introduces a benchmark for bioinformatics code generation and evaluates LLM systems.",
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="backup-paper",
                title="Supporting Workflow Reproducibility by Linking Bioinformatics Tools across Papers and Executable Code",
                url="https://example.org/reproducibility",
                snippet="Workflow reproducibility for bioinformatics tools.",
                backend="semantic_scholar",
                kind="paper",
                year="2026",
                abstract="Links bioinformatics tools across papers and executable code.",
                matched_queries=["AI for bioinformatics code"],
            ),
        ]

        selected = pipeline.select_results("AI for bioinformatics code", plan, {}, results)

        self.assertEqual(
            [item.result_id for item in selected],
            ["biocoder-paper", "backup-paper"],
        )
        self.assertNotIn("rejected-paper", [item.result_id for item in selected])

    def test_select_results_uses_plain_text_critic_fallback_when_schema_fails(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1, max_critic_results=2)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FallbackCriticOllama(
            "generic-paper\tNO\tGeneric AI hardware paper, not LLM chip design.\n"
            "llm-paper\tYES\tDirectly studies LLMs for hardware design generation.\n"
        )

        plan = ResearchPlan(
            queries=["AI for hardware chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="What work exists on using LLMs for hardware chip design?",
            must_cover=["LLMs", "chip design"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="generic-paper",
                title="Hardware-Software Co-Design for On-Chip Learning in AI Systems",
                url="https://doi.org/10.1145/example-generic",
                snippet="Hardware-software co-design for AI systems.",
                backend="crossref",
                kind="paper",
                year="2023",
                doi="10.1145/example-generic",
                abstract="A highly cited paper on on-chip learning and AI hardware systems.",
                citation_count=600,
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="llm-paper",
                title="C2HLSC: Leveraging Large Language Models to Bridge the Software-to-Hardware Design Gap",
                url="https://arxiv.org/abs/2412.00214v2",
                snippet="Large language models for hardware design generation.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2412.00214v2",
                abstract="Uses LLMs to translate software descriptions into hardware design artifacts.",
                matched_queries=["AI for hardware chip design"],
            ),
        ]

        selected = pipeline.select_results("AI for hardware chip design", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["llm-paper"])
        self.assertFalse(results[0].critic_relevant)
        self.assertTrue(results[1].critic_relevant)
        self.assertIn("Directly studies", results[1].critic_reason)

    def test_select_results_uses_single_item_critic_fallback_for_missing_batch_lines(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1, max_critic_results=2)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FallbackCriticOllama(
            "generic-paper\tNO\tStill generic.\n",
            single_text={
                "llm-paper": "YES\tDirectly studies LLMs for hardware design generation."
            },
        )

        plan = ResearchPlan(
            queries=["AI for hardware chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="What work exists on using LLMs for hardware chip design?",
            must_cover=["LLMs", "chip design"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="generic-paper",
                title="Hardware-Software Co-Design for On-Chip Learning in AI Systems",
                url="https://doi.org/10.1145/example-generic",
                snippet="Hardware-software co-design for AI systems.",
                backend="crossref",
                kind="paper",
                year="2023",
                doi="10.1145/example-generic",
                abstract="A highly cited paper on on-chip learning and AI hardware systems.",
                citation_count=600,
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="llm-paper",
                title="C2HLSC: Leveraging Large Language Models to Bridge the Software-to-Hardware Design Gap",
                url="https://arxiv.org/abs/2412.00214v2",
                snippet="Large language models for hardware design generation.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2412.00214v2",
                abstract="Uses LLMs to translate software descriptions into hardware design artifacts.",
                matched_queries=["AI for hardware chip design"],
            ),
        ]

        selected = pipeline.select_results("AI for hardware chip design", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["llm-paper"])
        self.assertTrue(results[1].critic_relevant)

    def test_select_results_default_critic_judges_entire_shortlist(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=3)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "CriticAgent": {
                    "judgments": [
                        {"result_id": "paper-a", "relevant": True, "reason": "Relevant."},
                        {"result_id": "paper-b", "relevant": True, "reason": "Relevant."},
                        {"result_id": "paper-c", "relevant": False, "reason": "Tangential."},
                    ]
                }
            }
        )

        plan = ResearchPlan(
            queries=["AI for hardware chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Find work on LLMs for hardware chip design.",
            must_cover=["LLMs", "chip design"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="paper-a",
                title="Large Language Models for Chip Design Generation",
                url="https://arxiv.org/abs/2501.00001",
                snippet="LLMs for chip design generation.",
                backend="arxiv",
                kind="paper",
                year="2025",
                arxiv_id="2501.00001",
                abstract="Uses LLMs to generate chip design artifacts.",
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="paper-b",
                title="LLMs for Hardware Verification",
                url="https://arxiv.org/abs/2501.00002",
                snippet="LLMs for verification.",
                backend="arxiv",
                kind="paper",
                year="2025",
                arxiv_id="2501.00002",
                abstract="Uses LLMs for hardware verification.",
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="paper-c",
                title="General Hardware Co-Design Survey",
                url="https://doi.org/10.1145/example-c",
                snippet="Survey.",
                backend="crossref",
                kind="paper",
                year="2024",
                doi="10.1145/example-c",
                abstract="Survey of hardware co-design.",
                matched_queries=["AI for hardware chip design"],
            ),
        ]

        pipeline.select_results("AI for hardware chip design", plan, {}, results)

        self.assertTrue(results[0].critic_relevant)
        self.assertTrue(results[1].critic_relevant)
        self.assertFalse(results[2].critic_relevant)

    def test_select_results_falls_back_when_critic_rejects_every_candidate(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1, max_critic_results=2)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "CriticAgent": {
                    "judgments": [
                        {
                            "result_id": "paper-a",
                            "relevant": False,
                            "reason": "Too generic.",
                        },
                        {
                            "result_id": "paper-b",
                            "relevant": False,
                            "reason": "Too generic.",
                        },
                    ]
                }
            }
        )

        plan = ResearchPlan(
            queries=["AI for hardware chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Find work on LLMs for hardware chip design.",
            must_cover=["LLMs", "chip design"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="paper-a",
                title="Large Language Models for Chip Design Generation",
                url="https://arxiv.org/abs/2501.00001",
                snippet="LLMs for chip design generation.",
                backend="arxiv",
                kind="paper",
                year="2025",
                arxiv_id="2501.00001",
                abstract="Uses LLMs to generate chip design artifacts.",
                matched_queries=["AI for hardware chip design"],
            ),
            SearchResult(
                result_id="paper-b",
                title="Hardware Co-Design Survey",
                url="https://doi.org/10.1145/example-b",
                snippet="Survey of hardware-software co-design.",
                backend="crossref",
                kind="paper",
                year="2024",
                doi="10.1145/example-b",
                abstract="Survey of hardware-software co-design.",
                matched_queries=["AI for hardware chip design"],
            ),
        ]

        selected = pipeline.select_results("AI for hardware chip design", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["paper-a"])

    def test_select_results_collapses_duplicate_source_variants(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            max_selected_sources=3,
            min_papers=2,
            min_web_sources=0,
        )
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["AI for bioinformatics code"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Find benchmarks and systems for AI in bioinformatics code generation.",
            must_cover=["bioinformatics", "code generation"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="biocoder-paper",
                title="BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models",
                url="https://arxiv.org/abs/2308.16458v5",
                snippet="Benchmark for bioinformatics code generation.",
                backend="arxiv",
                kind="paper",
                year="2023",
                arxiv_id="2308.16458v5",
                doi="10.1093/bioinformatics/btae230",
                abstract="Benchmark for bioinformatics code generation.",
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="biocoder-github",
                title="GitHub - gersteinlab/BioCoder: BioCoder: A Benchmark for Bioinformatics ...",
                url="https://github.com/gersteinlab/BioCoder",
                snippet="Repository for the BioCoder benchmark.",
                backend="duckduckgo",
                kind="web",
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="biocoder-pmc",
                title="BioCoder: a benchmark for bioinformatics code generation with large ...",
                url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11211839/",
                snippet="PMC mirror of the BioCoder article.",
                backend="duckduckgo",
                kind="web",
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="workflow-paper",
                title="From Prompt to Pipeline: Large Language Models for Scientific Workflow",
                url="https://arxiv.org/abs/2507.20122",
                snippet="Scientific workflow generation with LLMs.",
                backend="arxiv",
                kind="paper",
                year="2025",
                arxiv_id="2507.20122",
                abstract="Scientific workflow generation with LLMs.",
                matched_queries=["AI for bioinformatics code"],
            ),
        ]

        selected = pipeline.select_results("AI for bioinformatics code", plan, {}, results)

        self.assertEqual(
            [item.result_id for item in selected],
            ["biocoder-paper", "workflow-paper"],
        )

    def test_select_results_single_source_budget_skips_selector_model(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1)
        self.addCleanup(tempdir.cleanup)

        class CriticOnly:
            def chat_json(self, system: str, user: str, **_: object) -> dict:
                if "CriticAgent" in system:
                    return {
                        "judgments": [
                            {
                                "result_id": "paper-arxiv",
                                "relevant": True,
                                "reason": "Directly addresses transformer scaling laws.",
                            }
                        ]
                    }
                raise AssertionError("selector model should not be called")

        pipeline.ollama = CriticOnly()

        plan = ResearchPlan(
            queries=["transformer scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="paper-arxiv",
                title="Scaling Laws for Downstream Task Performance of Large Language Models",
                url="https://arxiv.org/abs/2402.04177",
                snippet="transformer scaling law evidence",
                backend="arxiv",
                kind="paper",
                authors=["Example"],
                year="2024",
                arxiv_id="2402.04177",
                abstract="Empirical study of scaling laws for language models.",
            )
        ]

        selected = pipeline.select_results("transformer scaling laws", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["paper-arxiv"])

    def test_select_results_penalizes_generic_repo_page(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=2, min_papers=2, min_web_sources=0)
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["superconducting electronic simulation in gem5", "gem5 sfq"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Assess gem5 support for SFQ logic simulation.",
            must_cover=["gem5 support", "SFQ logic"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="repo",
                title="GitHub - gem5/gem5: The official repository for the gem5 computer ...",
                url="https://github.com/gem5/gem5",
                snippet="The official repository for the gem5 simulator.",
                backend="duckduckgo",
                kind="web",
                matched_queries=["superconducting electronic simulation in gem5"],
            ),
            SearchResult(
                result_id="paper-gem5",
                title="The gem5 Simulator: Version 20.0+",
                url="https://arxiv.org/abs/2007.03152",
                snippet="General gem5 paper.",
                backend="arxiv",
                kind="paper",
                year="2020",
                arxiv_id="2007.03152",
                abstract="General gem5 paper.",
                matched_queries=["superconducting electronic simulation in gem5", "gem5 sfq"],
            ),
            SearchResult(
                result_id="paper-sfq",
                title="A behavioral-level HDL description of SFQ logic circuits for quantitative performance analysis",
                url="https://doi.org/10.1016/example",
                snippet="SFQ logic circuits for large-scale digital systems",
                backend="crossref",
                kind="paper",
                year="2003",
                doi="10.1016/example",
                abstract="Behavioral-level SFQ HDL for performance analysis.",
                matched_queries=["gem5 sfq"],
            ),
        ]

        selected = pipeline.select_results(
            "superconducting electronic simulation in gem5",
            plan,
            {"constraints": "SFQ RSFQ AQFP"},
            results,
        )

        self.assertCountEqual(
            [item.result_id for item in selected],
            ["paper-gem5", "paper-sfq"],
        )
        self.assertNotIn("repo", [item.result_id for item in selected])

    def test_select_results_penalizes_results_missing_core_topic_terms(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_selected_sources=1)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama({})

        plan = ResearchPlan(
            queries=["AI for bioinformatics code"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Find benchmarks and practical systems for AI in bioinformatics code generation.",
            must_cover=["bioinformatics", "code generation"],
            source_requirements=[],
        )
        results = [
            SearchResult(
                result_id="generic-ai",
                title="Beyond principlism: Practical strategies for ethical AI use in research practices",
                url="https://arxiv.org/abs/2402.00001",
                snippet="Practical strategies for ethical AI use in research workflows and benchmarks.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2402.00001",
                abstract="A practical discussion of AI use in research practices and benchmark culture.",
                citation_count=200,
                matched_queries=["AI for bioinformatics code"],
            ),
            SearchResult(
                result_id="biocoder",
                title="BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models",
                url="https://arxiv.org/abs/2404.12345",
                snippet="Benchmark for bioinformatics code generation with LLMs.",
                backend="arxiv",
                kind="paper",
                year="2024",
                arxiv_id="2404.12345",
                abstract="Introduces a benchmark for bioinformatics code generation and evaluates LLM systems.",
                citation_count=10,
                matched_queries=["AI for bioinformatics code"],
            ),
        ]

        selected = pipeline.select_results("AI for bioinformatics code", plan, {}, results)

        self.assertEqual([item.result_id for item in selected], ["biocoder"])

    def test_retrieve_results_expands_queries_from_facets_and_evidence(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            max_search_rounds=2,
            max_total_queries=6,
            max_query_batch_size=2,
            max_expansion_queries_per_round=3,
            max_ranked_results_for_expansion=4,
            max_selected_sources=2,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(should_fail=True)

        plan = ResearchPlan(
            queries=["superconducting architectures", "gem5 simulation"],
            related_topics=[],
            focus_areas=["full-system implications for architecture simulation"],
            rewritten_question="Assess full-system modeling of superconducting architectures in gem5.",
            must_cover=["superconducting architectures", "full-system modeling"],
            source_requirements=[],
        )
        target = SearchResult(
            result_id="target-paper",
            title="Implications of Full-System Modeling for Superconducting Architectures",
            url="https://doi.org/10.1145/3731599.3769278",
            snippet="full-system modeling for superconducting architectures",
            backend="crossref",
            kind="paper",
            year="2025",
            doi="10.1145/3731599.3769278",
            abstract="A study of full-system modeling implications.",
        )
        pipeline.search = FakeSearch(
            {
                "superconducting architectures": [
                    SearchResult(
                        result_id="anchor-paper",
                        title="Superconducting Architectures for Future Computing",
                        url="https://example.org/superconducting-architectures",
                        snippet="superconducting architectures and system design",
                        backend="crossref",
                        kind="paper",
                        year="2024",
                        doi="10.0000/anchor",
                        abstract="Study of superconducting architectures.",
                    )
                ],
                "gem5 simulation": [
                    SearchResult(
                        result_id="gem5-paper",
                        title="System Simulation with gem5 and SystemC",
                        url="https://example.org/gem5-system-simulation",
                        snippet="system simulation with gem5",
                        backend="crossref",
                        kind="paper",
                        year="2017",
                        doi="10.0000/gem5",
                        abstract="Study of gem5 system simulation.",
                    )
                ],
                "superconducting architectures full system modeling": [target],
                "full system implications superconducting electronic simulation": [target],
                "full system superconducting electronic simulation": [target],
                "full system modeling superconducting electronic simulation": [target],
                "superconducting architectures full system implications": [target],
            }
        )

        results, retrieval = pipeline.retrieve_results(
            "superconducting electronic simulation in gem5",
            plan,
            {"constraints": "SFQ RSFQ AQFP"},
        )

        self.assertTrue(
            any("superconducting" in query and "full system" in query for query in pipeline.search.calls)
        )
        self.assertIn("target-paper", [item.result_id for item in results])
        self.assertTrue(any(round_data["expansion_queries"] for round_data in retrieval["rounds"]))

    def test_retrieve_results_uses_strategy_groups_for_generic_topic(self) -> None:
        pipeline, tempdir = self.make_pipeline(
            max_search_rounds=2,
            max_total_queries=6,
            max_query_batch_size=2,
            max_expansion_queries_per_round=4,
            max_ranked_results_for_expansion=4,
            max_selected_sources=2,
        )
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "RetrievalStrategistAgent": {
                    "anchor_phrases": [
                        "federated learning",
                        "differential privacy",
                        "graph recommender systems",
                    ],
                    "search_facets": [
                        "privacy accounting",
                        "benchmark study",
                    ],
                    "generic_terms": [
                        "analysis",
                        "benchmark",
                        "study",
                        "system",
                    ],
                    "concept_groups": [
                        {
                            "label": "training",
                            "phrases": ["federated learning"],
                            "priority": 3,
                        },
                        {
                            "label": "privacy",
                            "phrases": ["differential privacy", "privacy accounting"],
                            "priority": 3,
                        },
                        {
                            "label": "application",
                            "phrases": ["graph recommender systems"],
                            "priority": 2,
                        },
                    ],
                }
            }
        )

        plan = ResearchPlan(
            queries=["federated learning", "graph recommender systems"],
            related_topics=[],
            focus_areas=["privacy guarantees for recommendation"],
            rewritten_question=(
                "Assess differential privacy methods for federated graph recommender systems."
            ),
            must_cover=["differential privacy", "graph recommender systems"],
            source_requirements=[],
        )
        target = SearchResult(
            result_id="target-paper",
            title="Differential Privacy for Federated Graph Recommender Systems",
            url="https://example.org/federated-graph-dp",
            snippet="privacy accounting for federated graph recommendation",
            backend="semantic_scholar",
            kind="paper",
            year="2024",
            doi="10.0000/fedgraphdp",
            abstract="A benchmark study of differential privacy for federated graph recommendation.",
        )
        pipeline.search = FakeSearch(
            {
                "federated learning": [],
                "graph recommender systems": [],
                "federated learning privacy accounting": [target],
                "federated learning differential privacy": [target],
                "privacy accounting graph recommender systems": [target],
                "federated learning graph recommender systems": [target],
            }
        )

        results, retrieval = pipeline.retrieve_results(
            "differential privacy in federated graph recommender systems",
            plan,
            {},
        )

        self.assertTrue(
            any(
                "federated learning" in query
                and ("differential privacy" in query or "privacy accounting" in query)
                for query in pipeline.search.calls
            )
        )
        self.assertIn("target-paper", [item.result_id for item in results])
        self.assertTrue(any(round_data["expansion_queries"] for round_data in retrieval["rounds"]))

    def test_synthesize_repairs_invalid_and_missing_citations(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(
            {
                "title": "Scaling Laws",
                "abstract": "A summary.",
                "sections": [
                    {
                        "heading": "Main Result",
                        "paragraphs": [
                            {
                                "text": (
                                    "Predictable scaling with model size appears in transformer "
                                    "training [kaplan2020scaling]."
                                ),
                                "citation_keys": [],
                            },
                            {
                                "text": "This paragraph cites a missing key but still discusses predictable scaling.",
                                "citation_keys": ["missing_key"],
                            },
                        ],
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding-1",
                        "claim": "Scaling is predictable.",
                        "evidence": "predictable scaling with model size",
                        "citation_keys": ["missing_key"],
                    },
                    {
                        "finding_id": "finding-2",
                        "claim": "Unrelated unsupported claim.",
                        "evidence": "No matching evidence",
                        "citation_keys": [],
                    },
                ],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            }
        )

        plan = ResearchPlan(
            queries=["scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        notes = {
            "s1": SourceNote(
                source_id="s1",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                citation_key="kaplan2020scaling",
                summary="Scaling laws show predictable scaling with model size, data, and compute.",
                claims=["predictable scaling with model size"],
                evidence_snippets=["predictable scaling with model size"],
            )
        }
        citations = [
            CitationRecord(
                cite_key="kaplan2020scaling",
                bibtex="@misc{kaplan2020scaling, title={Scaling Laws for Neural Language Models}}",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                source_id="s1",
            )
        ]

        synthesis = pipeline.synthesize(
            "scaling laws",
            plan,
            {},
            notes,
            citations,
            CollaborationSession(),
        )

        self.assertTrue(synthesis.sections)
        all_citation_keys = [
            key
            for section in synthesis.sections
            for paragraph in section.paragraphs
            for key in paragraph["citation_keys"]
        ]
        self.assertTrue(all_citation_keys)
        self.assertEqual(set(all_citation_keys), {"kaplan2020scaling"})
        self.assertEqual([finding.finding_id for finding in synthesis.findings], ["finding-1"])
        paragraph_text = synthesis.sections[0].paragraphs[0]["text"]
        self.assertNotIn("kaplan2020scaling", paragraph_text)

    def test_synthesize_filters_delete_requests_for_current_run_evidence(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = FakeOllama(
            {
                "title": "Scaling Laws",
                "abstract": "A summary.",
                "sections": [
                    {
                        "heading": "Main Result",
                        "paragraphs": [
                            {
                                "text": "Predictable scaling with model size appears in transformer training.",
                                "citation_keys": ["kaplan2020scaling"],
                            }
                        ],
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding-1",
                        "claim": "Scaling is predictable.",
                        "evidence": "predictable scaling with model size",
                        "citation_keys": ["kaplan2020scaling"],
                    }
                ],
                "notes": [],
                "delete_citation_keys": ["kaplan2020scaling", "legacy2022"],
                "delete_finding_ids": ["finding-1", "legacy-finding"],
            }
        )

        plan = ResearchPlan(
            queries=["scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        notes = {
            "s1": SourceNote(
                source_id="s1",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                citation_key="kaplan2020scaling",
                summary="Scaling laws show predictable scaling with model size, data, and compute.",
                claims=["predictable scaling with model size"],
                evidence_snippets=["predictable scaling with model size"],
            )
        }
        citations = [
            CitationRecord(
                cite_key="kaplan2020scaling",
                bibtex="@misc{kaplan2020scaling, title={Scaling Laws for Neural Language Models}}",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                source_id="s1",
            )
        ]

        synthesis = pipeline.synthesize(
            "scaling laws",
            plan,
            {},
            notes,
            citations,
            CollaborationSession(),
        )

        self.assertEqual(synthesis.delete_citation_keys, ["legacy2022"])
        self.assertEqual(synthesis.delete_finding_ids, ["legacy-finding"])
        self.assertTrue(
            any("current-run citation" in note for note in synthesis.notes),
            synthesis.notes,
        )
        self.assertTrue(
            any("current-run finding" in note for note in synthesis.notes),
            synthesis.notes,
        )

    def test_collaborate_runs_worker_debate_and_coordinator(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = RoleAwareOllama(
            {
                "You are EvidenceAgent.": {
                    "summary": "Primary evidence supports predictable scaling.",
                    "claims": [
                        {
                            "claim": "Scaling appears predictable with model size.",
                            "citation_keys": ["kaplan2020scaling"],
                            "status": "supported",
                        }
                    ],
                    "criticisms": [],
                    "open_questions": ["How robust is the compute scaling law?"],
                    "messages_to_next": ["Stress-test whether the evidence is overgeneralized."],
                },
                "You are SkepticAgent.": {
                    "summary": "The evidence is real but narrow.",
                    "claims": [
                        {
                            "claim": "Some scaling claims are narrower than broad headlines suggest.",
                            "citation_keys": ["kaplan2020scaling"],
                            "status": "challenged",
                        }
                    ],
                    "criticisms": ["Do not claim universality from one paper."],
                    "open_questions": [],
                    "messages_to_next": ["Check for missing subtopics or benchmarks."],
                },
                "You are GapAgent.": {
                    "summary": "The debate is missing evaluation coverage.",
                    "claims": [],
                    "criticisms": ["Benchmark diversity is still thin."],
                    "open_questions": ["What downstream tasks were covered?"],
                    "messages_to_next": ["Preserve the uncertainty in the final synthesis."],
                },
                "You are ChairAgent.": {
                    "consensus_claims": [
                        {
                            "claim": "Scaling with model size is supported, but scope limits should be stated.",
                            "citation_keys": ["kaplan2020scaling"],
                            "status": "supported",
                        }
                    ],
                    "disputed_claims": ["Universal scaling claims across all settings."],
                    "open_questions": ["What downstream tasks were covered?"],
                    "coordinator_notes": ["Keep the final report conservative and source-grounded."],
                },
            }
        )

        plan = ResearchPlan(
            queries=["scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        notes = {
            "s1": SourceNote(
                source_id="s1",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                citation_key="kaplan2020scaling",
                summary="Scaling laws show predictable scaling with model size, data, and compute.",
                claims=["predictable scaling with model size"],
                evidence_snippets=["predictable scaling with model size"],
            )
        }
        citations = [
            CitationRecord(
                cite_key="kaplan2020scaling",
                bibtex="@misc{kaplan2020scaling, title={Scaling Laws for Neural Language Models}}",
                title="Scaling Laws for Neural Language Models",
                url="https://arxiv.org/abs/2001.08361",
                source_id="s1",
            )
        ]

        session = pipeline.collaborate("scaling laws", plan, {}, notes, citations)

        self.assertEqual([turn.role for turn in session.turns], ["EvidenceAgent", "SkepticAgent", "GapAgent"])
        self.assertEqual(session.consensus_claims[0]["citation_keys"], ["kaplan2020scaling"])
        self.assertIn("Universal scaling claims", session.disputed_claims[0])
        self.assertIn("final report conservative", session.coordinator_notes[0])

    def test_build_plan_falls_back_when_ollama_times_out(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_queries=4)
        self.addCleanup(tempdir.cleanup)

        with patch.object(
            pipeline.ollama, "chat_json", side_effect=OllamaError("timeout")
        ):
            plan = pipeline.build_plan(
                "transformer scaling laws",
                {"objective": "summarize empirical findings and limits"},
            )

        self.assertTrue(plan.rewritten_question)
        self.assertTrue(plan.queries)
        self.assertIn("transformer scaling laws", plan.rewritten_question.lower())

    def test_single_chunk_summary_skips_merge_call(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_chunks_per_source=1)
        self.addCleanup(tempdir.cleanup)
        pipeline.ollama = SequenceOllama(
            [
                {
                    "summary": "Scaling laws show predictable power-law behavior.",
                    "claims": ["Loss decreases predictably with scale."],
                    "evidence_snippets": ["predictable power-law behavior"],
                    "related_topics": ["compute-optimal scaling"],
                }
            ]
        )

        plan = ResearchPlan(
            queries=["transformer scaling laws"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Review transformer scaling law evidence.",
            must_cover=[],
            source_requirements=[],
        )
        source_documents = [
            SourceDocument(
                source_id="paper-arxiv",
                title="Scaling Laws for Downstream Task Performance of Large Language Models",
                url="https://arxiv.org/abs/2402.04177",
                kind="paper",
                backend="arxiv",
                authors=["Example"],
                year="2024",
                arxiv_id="2402.04177",
                abstract="Empirical study of scaling laws for language models.",
                text="Empirical study of scaling laws for language models.",
                text_chunks=["Empirical study of scaling laws for language models."],
            )
        ]
        citations = [
            CitationRecord(
                cite_key="example2024scaling",
                bibtex="@misc{example2024scaling, title={Scaling Laws for Downstream Task Performance of Large Language Models}}",
                title=source_documents[0].title,
                url=source_documents[0].url,
                source_id=source_documents[0].source_id,
            )
        ]

        notes = pipeline.summarize_documents(plan, source_documents, citations)

        self.assertEqual(pipeline.ollama.calls, 1)
        self.assertEqual(
            notes[source_documents[0].source_id].summary,
            "Scaling laws show predictable power-law behavior.",
        )

    def test_apply_summary_budget_truncates_and_drops_sources(self) -> None:
        pipeline, tempdir = self.make_pipeline(max_summary_model_calls=2)
        self.addCleanup(tempdir.cleanup)

        selected = [
            SearchResult(
                result_id="paper-1",
                title="Doc One",
                url="https://example.com/1",
                snippet="doc one",
                backend="crossref",
                kind="paper",
            ),
            SearchResult(
                result_id="paper-2",
                title="Doc Two",
                url="https://example.com/2",
                snippet="doc two",
                backend="crossref",
                kind="paper",
            ),
            SearchResult(
                result_id="paper-3",
                title="Doc Three",
                url="https://example.com/3",
                snippet="doc three",
                backend="crossref",
                kind="paper",
            ),
        ]
        documents = [
            SourceDocument(
                source_id="paper-1",
                title="Doc One",
                url="https://example.com/1",
                kind="paper",
                backend="crossref",
                text_chunks=["a", "b", "c"],
            ),
            SourceDocument(
                source_id="paper-2",
                title="Doc Two",
                url="https://example.com/2",
                kind="paper",
                backend="crossref",
                text_chunks=["d", "e"],
            ),
            SourceDocument(
                source_id="paper-3",
                title="Doc Three",
                url="https://example.com/3",
                kind="paper",
                backend="crossref",
                text_chunks=["f"],
            ),
        ]

        budgeted_selected, budgeted_documents, budget = pipeline._apply_summary_budget(
            selected,
            documents,
        )

        self.assertEqual([item.result_id for item in budgeted_selected], ["paper-1", "paper-2"])
        self.assertEqual([item.source_id for item in budgeted_documents], ["paper-1", "paper-2"])
        self.assertEqual([len(item.text_chunks) for item in budgeted_documents], [1, 1])
        self.assertEqual(budget["estimated_summary_calls_before_budget"], 8)
        self.assertEqual(budget["estimated_summary_calls_after_budget"], 2)
        self.assertEqual(
            [item["source_id"] for item in budget["dropped_sources"]],
            ["paper-3"],
        )
        self.assertCountEqual(
            [item["source_id"] for item in budget["truncated_sources"]],
            ["paper-1", "paper-2"],
        )

    def test_write_checkpoint_writes_partial_run_and_references(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["biocoder"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize BioCoder.",
            must_cover=[],
            source_requirements=[],
        )
        retrieval = {"selected_sources": [{"result_id": "paper-1"}]}
        selected = [
            SearchResult(
                result_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                snippet="benchmark",
                backend="arxiv",
                kind="paper",
            )
        ]
        documents = [
            SourceDocument(
                source_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                kind="paper",
                backend="arxiv",
                text_chunks=["BioCoder is a benchmark."],
            )
        ]
        citations = [
            CitationRecord(
                cite_key="Tang_2024",
                bibtex="@article{Tang_2024, title={BioCoder}}",
                title="BioCoder",
                url="https://example.com/biocoder",
                source_id="paper-1",
            )
        ]
        notes = {
            "paper-1": SourceNote(
                source_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                citation_key="Tang_2024",
                summary="BioCoder is a benchmark.",
            )
        }

        pipeline._write_checkpoint(
            topic="BioCoder",
            status="summarizing",
            answers={},
            plan=plan,
            retrieval=retrieval,
            selected=selected,
            documents=documents,
            citations=citations,
            note_by_source=notes,
            progress={"summarized_sources": 1, "total_sources": 1},
        )

        run_payload = (pipeline.output_dir / pipeline.settings.run_filename).read_text(
            encoding="utf-8"
        )
        retrieval_payload = (
            pipeline.output_dir / pipeline.settings.retrieval_filename
        ).read_text(encoding="utf-8")
        references_payload = (
            pipeline.output_dir / pipeline.settings.references_filename
        ).read_text(encoding="utf-8")

        self.assertIn('"status": "summarizing"', run_payload)
        self.assertIn('"summarized_sources": 1', run_payload)
        self.assertIn('"selected_sources"', retrieval_payload)
        self.assertIn("@article{Tang_2024", references_payload)

    def test_write_outputs_compiles_pdf_with_latexmk(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["biocoder"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize BioCoder.",
            must_cover=[],
            source_requirements=[],
        )
        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "BioCoder",
                "abstract": "Summary.",
                "sections": [
                    {
                        "heading": "Overview",
                        "paragraphs": [
                            {"text": "BioCoder summary.", "citation_keys": []},
                        ],
                    }
                ],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "BioCoder",
        )
        source_notes = [
            SourceNote(
                source_id="paper-1",
                title="BioCoder",
                url="https://example.com/biocoder",
                citation_key="Tang_2024",
                summary="BioCoder summary.",
            )
        ]

        def fake_run(command: list[str], cwd: Path, **_: object) -> object:
            self.assertEqual(command[0], "/usr/bin/latexmk")
            pdf_path = Path(cwd) / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            return object()

        with (
            patch.object(shutil, "which", side_effect=lambda name: "/usr/bin/latexmk" if name == "latexmk" else None),
            patch("subprocess.run", side_effect=fake_run) as run_mock,
        ):
            paths = pipeline.write_outputs(
                topic="BioCoder",
                answers={},
                plan=plan,
                selected=[],
                documents=[],
                citations=[],
                source_notes=source_notes,
                synthesis=synthesis,
                retrieval={},
                collaboration=CollaborationSession(),
            )

        self.assertIn("pdf", paths)
        self.assertTrue(paths["pdf"].exists())
        run_payload = json.loads(
            (pipeline.output_dir / pipeline.settings.run_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(run_payload["latex"]["status"], "succeeded")
        self.assertEqual(run_payload["compiled_pdf"], str(paths["pdf"]))
        self.assertEqual(len(run_payload["source_notes"]), 1)
        run_mock.assert_called_once()

    def test_write_outputs_skips_compile_when_tex_missing(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["biocoder"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize BioCoder.",
            must_cover=[],
            source_requirements=[],
        )
        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "BioCoder",
                "abstract": "Summary.",
                "sections": [],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "BioCoder",
        )

        with (
            patch.object(shutil, "which", return_value=None),
            patch("subprocess.run") as run_mock,
        ):
            paths = pipeline.write_outputs(
                topic="BioCoder",
                answers={},
                plan=plan,
                selected=[],
                documents=[],
                citations=[],
                source_notes=[],
                synthesis=synthesis,
                retrieval={},
                collaboration=CollaborationSession(),
            )

        self.assertNotIn("pdf", paths)
        run_mock.assert_not_called()
        run_payload = json.loads(
            (pipeline.output_dir / pipeline.settings.run_filename).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(run_payload["latex"]["status"], "skipped")

    def test_write_outputs_sanitizes_bibtex_ampersands(self) -> None:
        pipeline, tempdir = self.make_pipeline(compile_latex=False)
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["chip design"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize chip design work.",
            must_cover=[],
            source_requirements=[],
        )
        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "Chip Design",
                "abstract": "Summary.",
                "sections": [],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "Chip Design",
        )
        citations = [
            CitationRecord(
                cite_key="chip2025",
                bibtex="@misc{chip2025,\n  title = {Recursive Self-Improvement via AI for Chip Design & Chip Design for AI},\n  year = {2025}\n}",
                title="Recursive Self-Improvement via AI for Chip Design & Chip Design for AI",
                url="https://example.com/chip",
                source_id="paper-1",
            )
        ]

        pipeline.write_outputs(
            topic="Chip Design",
            answers={},
            plan=plan,
            selected=[],
            documents=[],
            citations=citations,
            source_notes=[],
            synthesis=synthesis,
            retrieval={},
            collaboration=CollaborationSession(),
        )

        references_text = (pipeline.output_dir / pipeline.settings.references_filename).read_text(
            encoding="utf-8"
        )
        self.assertIn("Chip Design \\& Chip Design for AI", references_text)

    def test_write_outputs_filters_invalid_bibtex_entries(self) -> None:
        pipeline, tempdir = self.make_pipeline(compile_latex=False)
        self.addCleanup(tempdir.cleanup)

        plan = ResearchPlan(
            queries=["bioinformatics code"],
            related_topics=[],
            focus_areas=[],
            rewritten_question="Summarize bioinformatics code generation.",
            must_cover=[],
            source_requirements=[],
        )
        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "Bioinformatics Code",
                "abstract": "Summary.",
                "sections": [],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "Bioinformatics Code",
        )
        citations = [
            CitationRecord(
                cite_key="good2024",
                bibtex="@article{good2024,\n  title = {Valid Paper},\n  year = {2024}\n}",
                title="Valid Paper",
                url="https://example.com/valid",
                source_id="paper-1",
            ),
            CitationRecord(
                cite_key="bad2024",
                bibtex="<!doctype html><html><body>Google Scholar challenge</body></html>",
                title="Bad Paper",
                url="https://example.com/bad",
                source_id="paper-2",
            ),
        ]

        pipeline.write_outputs(
            topic="Bioinformatics Code",
            answers={},
            plan=plan,
            selected=[],
            documents=[],
            citations=citations,
            source_notes=[],
            synthesis=synthesis,
            retrieval={},
            collaboration=CollaborationSession(),
        )

        references_text = (
            pipeline.output_dir / pipeline.settings.references_filename
        ).read_text(encoding="utf-8")
        self.assertIn("@article{good2024", references_text)
        self.assertNotIn("<!doctype html>", references_text.lower())

    def test_render_report_strips_emoji_from_latex_text(self) -> None:
        pipeline, tempdir = self.make_pipeline(compile_latex=False)
        self.addCleanup(tempdir.cleanup)

        synthesis = pipeline._coerce_synthesis_payload(
            {
                "title": "RTL list 📚",
                "abstract": "Emoji 📚 should not reach pdflatex.",
                "sections": [
                    {
                        "heading": "Findings",
                        "paragraphs": [
                            {"text": "GitHub list 📚 with dash – preserved safely.", "citation_keys": []},
                        ],
                    }
                ],
                "findings": [],
                "notes": [],
                "delete_citation_keys": [],
                "delete_finding_ids": [],
            },
            "RTL list",
        )

        report = pipeline._render_report(synthesis)
        body = report.split("\\begin{document}", 1)[-1]

        self.assertNotIn("📚", body)
        self.assertIn("dash -- preserved safely.", report)
        self.assertIn("\\usepackage{iftex}", report)
        self.assertIn("\\IfFileExists{newunicodechar.sty}", report)

    def test_compile_latex_prefers_lualatex_with_latexmk(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        report_path = pipeline.output_dir / "report.tex"
        report_path.write_text("stub", encoding="utf-8")

        def fake_run(command: list[str], cwd: Path, **_: object) -> object:
            self.assertEqual(command[0], "/usr/bin/latexmk")
            self.assertIn("-lualatex", command)
            (Path(cwd) / "report.pdf").write_bytes(b"%PDF-1.4\n")
            return object()

        def fake_which(name: str) -> str | None:
            mapping = {
                "latexmk": "/usr/bin/latexmk",
                "lualatex": "/usr/bin/lualatex",
            }
            return mapping.get(name)

        with (
            patch.object(shutil, "which", side_effect=fake_which),
            patch("subprocess.run", side_effect=fake_run),
        ):
            result = pipeline._compile_latex(report_path)

        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["pdf"])

    def test_compile_latex_surfaces_fatal_log_detail(self) -> None:
        pipeline, tempdir = self.make_pipeline()
        self.addCleanup(tempdir.cleanup)

        report_path = pipeline.output_dir / "report.tex"
        report_path.write_text("stub", encoding="utf-8")
        (pipeline.output_dir / "report.log").write_text(
            "! LaTeX Error: Unicode character 📚 (U+1F4DA)\n"
            "not set up for use with LaTeX.\n",
            encoding="utf-8",
        )

        with (
            patch.object(shutil, "which", side_effect=lambda name: "/usr/bin/latexmk" if name == "latexmk" else None),
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(12, ["/usr/bin/latexmk"])),
        ):
            result = pipeline._compile_latex(report_path)

        self.assertEqual(result["status"], "failed")
        self.assertIn("Unicode character", result["message"])


if __name__ == "__main__":
    unittest.main()
