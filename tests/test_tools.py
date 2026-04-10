from __future__ import annotations

import unittest

from deep_research_ollama.config import Settings
from deep_research_ollama.models import SearchResult
from deep_research_ollama.tools import SearchToolkit


class StubSearchToolkit(SearchToolkit):
    def __init__(
        self,
        responses: dict[str, dict],
        text_responses: dict[str, str] | None = None,
    ) -> None:
        super().__init__(Settings())
        self.responses = responses
        self.text_responses = text_responses or {}

    def _fetch_json(self, url: str, headers: dict[str, str] | None = None) -> dict:
        for marker, payload in self.responses.items():
            if marker in url:
                return payload
        return {}

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        for marker, payload in self.text_responses.items():
            if marker in url:
                return payload
        return ""


class SearchToolkitTests(unittest.TestCase):
    def test_dedupe_results_merges_query_hits_and_prefers_richer_record(self) -> None:
        left = SearchResult(
            result_id="arxiv:1234.5678",
            title="Modeling SFQ Digital Logic",
            url="https://arxiv.org/abs/1234.5678",
            snippet="short",
            backend="arxiv",
            kind="paper",
            arxiv_id="1234.5678",
            abstract="A richer abstract for the paper.",
            matched_queries=["sfq modeling"],
        )
        right = SearchResult(
            result_id="semanticscholar:abc",
            title="Modeling SFQ Digital Logic",
            url="https://www.semanticscholar.org/paper/abc",
            snippet="A richer snippet for the same paper.",
            backend="semantic_scholar",
            kind="paper",
            doi="10.1000/example",
            citation_count=120,
            matched_queries=["superconducting logic", "sfq modeling"],
        )

        merged = SearchToolkit._dedupe_results([left, right])

        self.assertEqual(len(merged), 1)
        self.assertCountEqual(
            merged[0].matched_queries,
            ["sfq modeling", "superconducting logic"],
        )
        self.assertEqual(merged[0].citation_count, 120)
        self.assertEqual(merged[0].doi, "10.1000/example")

    def test_dedupe_results_merges_github_and_truncated_title_variants(self) -> None:
        paper = SearchResult(
            result_id="arxiv:2308.16458v5",
            title="BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models",
            url="https://arxiv.org/abs/2308.16458v5",
            snippet="Benchmark for bioinformatics code generation.",
            backend="arxiv",
            kind="paper",
            arxiv_id="2308.16458v5",
            doi="10.1093/bioinformatics/btae230",
            abstract="BioCoder benchmark paper.",
            matched_queries=["bioinformatics code generation"],
        )
        github = SearchResult(
            result_id="duck:github",
            title="GitHub - gersteinlab/BioCoder: BioCoder: A Benchmark for Bioinformatics ...",
            url="https://github.com/gersteinlab/BioCoder",
            snippet="GitHub repository for BioCoder.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["biocoder dataset"],
        )
        pmc = SearchResult(
            result_id="duck:pmc",
            title="BioCoder: a benchmark for bioinformatics code generation with large ...",
            url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11211839/",
            snippet="PMC mirror of the BioCoder article.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["bioinformatics code generation scientific"],
        )
        paper.critic_relevant = True
        paper.critic_reason = "Direct benchmark for bioinformatics code generation."
        paper.critic_query = "AI for bioinformatics code"

        merged = SearchToolkit._dedupe_results([paper, github, pmc])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].backend, "arxiv")
        self.assertEqual(merged[0].doi, "10.1093/bioinformatics/btae230")
        self.assertTrue(merged[0].critic_relevant)
        self.assertIn("benchmark", merged[0].critic_reason)
        self.assertCountEqual(
            merged[0].matched_queries,
            [
                "bioinformatics code generation",
                "biocoder dataset",
                "bioinformatics code generation scientific",
            ],
        )

    def test_merge_results_does_not_steal_text_from_unrelated_web_page(self) -> None:
        paper = SearchResult(
            result_id="crossref:10.1093/example",
            title="Skills for Doing",
            url="https://doi.org/10.1093/example",
            snippet="Workbook module for problem-solving therapy.",
            backend="crossref",
            kind="paper",
            doi="10.1093/example",
            abstract="Workbook module for depression treatment.",
            matched_queries=["bioinformatics code"],
        )
        web = SearchResult(
            result_id="duck:0",
            title="Bioinformatics Code Skills",
            url="https://example.com/bioinformatics-code-skills",
            snippet="This repository provides AI agents with expert knowledge for bioinformatics workflows.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["bioinformatics code"],
        )

        merged = SearchToolkit._merge_results(paper, web)

        self.assertEqual(merged.snippet, "Workbook module for problem-solving therapy.")
        self.assertEqual(merged.abstract, "Workbook module for depression treatment.")

    def test_promote_web_result_with_doi_url_to_crossref_paper(self) -> None:
        toolkit = StubSearchToolkit(
            {
                "api.crossref.org/works/10.1145%2F3731599.3769278": {
                    "message": {
                        "DOI": "10.1145/3731599.3769278",
                        "title": ["Implications of Full-System Modeling for Superconducting Architectures"],
                        "author": [{"given": "Kunal", "family": "Pai"}],
                        "abstract": "<jats:p>Full-system superconducting architecture modeling in gem5.</jats:p>",
                        "published-online": {"date-parts": [[2025]]},
                        "URL": "https://doi.org/10.1145/3731599.3769278",
                    }
                }
            }
        )
        web_result = SearchResult(
            result_id="duck:1",
            title="Implications of Full-System Modeling for Superconducting Architectures ...",
            url="https://dl.acm.org/doi/10.1145/3731599.3769278",
            snippet="This paper presents the first full-system superconducting architecture modeling in gem5.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["superconducting gem5"],
        )

        promoted = toolkit._promote_result(web_result)

        self.assertEqual(promoted.kind, "paper")
        self.assertEqual(promoted.backend, "crossref")
        self.assertEqual(promoted.doi, "10.1145/3731599.3769278")
        self.assertEqual(
            promoted.title,
            "Implications of Full-System Modeling for Superconducting Architectures",
        )
        self.assertEqual(promoted.url, "https://dl.acm.org/doi/10.1145/3731599.3769278")

    def test_promote_ieee_landing_page_by_title_lookup(self) -> None:
        toolkit = StubSearchToolkit(
            {
                "query.title=HDL-Based+Modeling+Approach+for+Digital+Simulation+of+Adiabatic+Quantum": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1109/TASC.2016.7582438",
                                "title": [
                                    "HDL-Based Modeling Approach for Digital Simulation of Adiabatic Quantum Flux Parametron Logic"
                                ],
                                "author": [{"given": "Naoki", "family": "Yamanashi"}],
                                "abstract": "<jats:p>A logic simulation model for AQFP logic.</jats:p>",
                                "published-online": {"date-parts": [[2016]]},
                                "URL": "https://doi.org/10.1109/TASC.2016.7582438",
                            }
                        ]
                    }
                }
            }
        )
        web_result = SearchResult(
            result_id="duck:2",
            title="HDL-Based Modeling Approach for Digital Simulation of Adiabatic Quantum ...",
            url="https://ieeexplore.ieee.org/document/7582438",
            snippet="A logic simulation model for AQFP logic.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["aqfp modeling"],
        )

        promoted = toolkit._promote_result(web_result)

        self.assertEqual(promoted.kind, "paper")
        self.assertEqual(promoted.backend, "crossref")
        self.assertEqual(promoted.doi, "10.1109/TASC.2016.7582438")
        self.assertIn("AQFP", promoted.abstract)

    def test_does_not_promote_generic_web_title_to_unrelated_crossref_paper(self) -> None:
        toolkit = StubSearchToolkit(
            {
                "query.title=Bioinformatics+Code+Skills": {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1093/med-psych/9780190068394.003.0004",
                                "title": ["Skills for Doing"],
                                "author": [{"given": "Julie", "family": "Loebach"}],
                                "abstract": "<jats:p>Workbook module for depression treatment.</jats:p>",
                                "published-online": {"date-parts": [[2021]]},
                                "URL": "https://doi.org/10.1093/med-psych/9780190068394.003.0004",
                            }
                        ]
                    }
                }
            }
        )
        web_result = SearchResult(
            result_id="duck:3",
            title="Bioinformatics Code Skills",
            url="https://example.com/bioinformatics-code-skills",
            snippet="This paper presents coding skills and AI support for bioinformatics workflows.",
            backend="duckduckgo",
            kind="web",
            matched_queries=["bioinformatics code"],
        )

        promoted = toolkit._promote_result(web_result)

        self.assertEqual(promoted.kind, "web")
        self.assertEqual(promoted.title, "Bioinformatics Code Skills")
        self.assertEqual(promoted.backend, "duckduckgo")

    def test_search_google_scholar_html_parses_metadata_and_cite_url(self) -> None:
        html = """
        <html><body>
        <div class="gs_r gs_or gs_scl" data-cid="abc123" data-rp="0">
          <div class="gs_ggs gs_fl">
            <div class="gs_ggsd"><div class="gs_or_ggsm"><a href="https://arxiv.org/pdf/2402.01801"><span class=gs_ctg2>[PDF]</span> arxiv.org</a></div></div>
          </div>
          <div class="gs_ri">
            <h3 class="gs_rt"><a href="https://arxiv.org/abs/2402.01801">Large language models for <b>time series</b>: A survey</a></h3>
            <div class="gs_a">X Zhang, RR Chowdhury, RK Gupta - arXiv preprint, 2024 - arxiv.org</div>
            <div class="gs_rs">Representative works, mathematical formulation, and applications.</div>
            <div class="gs_fl gs_flb"><a href="/scholar?cites=123&hl=en">Cited by 217</a></div>
          </div>
        </div>
        </body></html>
        """
        toolkit = StubSearchToolkit({}, text_responses={"scholar.google.com/scholar?": html})

        results = toolkit._search_google_scholar_html("llms for time series")

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.backend, "google_scholar")
        self.assertEqual(result.kind, "paper")
        self.assertEqual(result.title, "Large language models for time series: A survey")
        self.assertEqual(result.year, "2024")
        self.assertEqual(result.citation_count, 217)
        self.assertEqual(result.scholar_id, "abc123")
        self.assertIn("output=cite", result.scholar_cite_url)
        self.assertIn("X Zhang", result.authors)

    def test_promote_google_scholar_arxiv_result_preserves_scholar_metadata(self) -> None:
        toolkit = StubSearchToolkit({})
        scholar_result = SearchResult(
            result_id="google_scholar:abc123",
            title="Large language models for time series: A survey",
            url="https://arxiv.org/abs/2402.01801",
            snippet="Survey of LLMs for time series.",
            backend="google_scholar",
            kind="paper",
            authors=["X Zhang"],
            year="2024",
            abstract="Survey of LLMs for time series.",
            citation_count=217,
            matched_queries=["llms for time series"],
            scholar_id="abc123",
            scholar_cite_url="https://scholar.google.com/scholar?q=info%3Aabc123%3Ascholar.google.com%2F&output=cite",
        )

        promoted = toolkit._promote_result(scholar_result)

        self.assertEqual(promoted.backend, "arxiv")
        self.assertEqual(promoted.arxiv_id, "2402.01801")
        self.assertEqual(promoted.scholar_id, "abc123")
        self.assertIn("output=cite", promoted.scholar_cite_url)


if __name__ == "__main__":
    unittest.main()
