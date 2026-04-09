from __future__ import annotations

import unittest

from deep_research_ollama.citations import CitationResolver
from deep_research_ollama.config import Settings
from deep_research_ollama.models import SourceDocument


class CitationResolverTests(unittest.TestCase):
    def make_resolver(self) -> CitationResolver:
        return CitationResolver(Settings())

    def test_arxiv_source_does_not_use_title_lookup_doi(self) -> None:
        resolver = self.make_resolver()
        source = SourceDocument(
            source_id="arxiv:2412.07942v1",
            title="Neural Scaling Laws Rooted in the Data Distribution",
            url="https://arxiv.org/abs/2412.07942v1",
            kind="paper",
            backend="arxiv",
            authors=["Ari Brill"],
            year="2024",
            arxiv_id="2412.07942v1",
        )

        resolver._lookup_doi_by_title = lambda title: "10.0000/wrong-doi"  # type: ignore[method-assign]
        resolver._fetch_bibtex_for_doi = lambda doi: "@article{wrong, title={Wrong Paper}}"  # type: ignore[method-assign]

        citation = resolver.resolve(source)

        self.assertIn("archivePrefix = {arXiv}", citation.bibtex)
        self.assertEqual(citation.doi, "")

    def test_lookup_doi_by_title_rejects_weak_match(self) -> None:
        resolver = self.make_resolver()
        resolver._get_json = lambda url: {  # type: ignore[method-assign]
            "message": {
                "items": [
                    {
                        "DOI": "10.0000/wrong",
                        "title": ["Beyond Neural Scaling Laws: Beating Power Law Scaling Via Data Pruning"],
                    }
                ]
            }
        }

        doi = resolver._lookup_doi_by_title(
            "Neural Scaling Laws Rooted in the Data Distribution"
        )

        self.assertEqual(doi, "")

    def test_resolve_normalizes_crossref_bibtex_for_latex(self) -> None:
        resolver = self.make_resolver()
        source = SourceDocument(
            source_id="paper:biocoder",
            title="BioCoder: A Benchmark for Bioinformatics Code Generation with Large Language Models",
            url="https://arxiv.org/abs/2308.16458v5",
            kind="paper",
            backend="arxiv",
            authors=["Xiangru Tang"],
            year="2024",
            doi="10.1093/bioinformatics/btae230",
        )
        resolver._fetch_bibtex_for_doi = lambda doi: (  # type: ignore[method-assign]
            "@article{Tang_2024, title={BioCoder: a benchmark for bioinformatics code generation with large language models}, "
            "volume={40}, number={Supplement_1}, pages={i266–i276} }"
        )

        citation = resolver.resolve(source)

        self.assertIn("@article{Tang_2024,", citation.bibtex)
        self.assertIn("number={Supplement\\_1}", citation.bibtex)
        self.assertIn("pages={i266--i276}", citation.bibtex)

    def test_replace_cite_key_handles_numeric_prefix_without_regex_error(self) -> None:
        updated = CitationResolver._replace_cite_key(
            "@article{oldkey, title={Example}}",
            "2024example",
        )

        self.assertEqual(updated, "@article{2024example, title={Example}}")


if __name__ == "__main__":
    unittest.main()
