from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from deep_research_ollama.constitution import ConstitutionStore
from deep_research_ollama.models import CitationRecord, Finding, SourceNote, SynthesisResult


class ConstitutionStoreTests(unittest.TestCase):
    def test_checkpoint_sources_persists_partial_citations_and_notes(self) -> None:
        with TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            store = ConstitutionStore(base / "constitution.json", base / "constitution.bib")
            store.load("test topic")

            source_notes = [
                SourceNote(
                    source_id="current-source",
                    title="Current Paper",
                    url="https://example.com/current",
                    citation_key="current2024",
                    summary="Current source summary.",
                )
            ]
            citations = [
                CitationRecord(
                    cite_key="current2024",
                    bibtex="@misc{current2024, title={Current Citation}}",
                    title="Current Citation",
                    url="https://example.com/current",
                    source_id="current-source",
                )
            ]

            store.checkpoint_sources(source_notes, citations)

            self.assertIn("current-source", store.data["source_notes"])
            self.assertIn("current2024", store.data["citations"])
            self.assertTrue(store.json_path.exists())
            self.assertEqual(store.data["metadata"]["schema_version"], 2)
            self.assertEqual(store.data["metadata"]["last_checkpoint_stage"], "sources")
            self.assertEqual(
                store.data["source_notes"]["current-source"]["_meta"]["kind"],
                "source_note",
            )
            self.assertIn(
                store.data["source_notes"]["current-source"]["_meta"]["confidence_label"],
                {"low", "medium", "high"},
            )
            self.assertGreaterEqual(
                store.data["source_notes"]["current-source"]["_meta"]["confidence"],
                0.05,
            )
            self.assertEqual(
                store.data["citations"]["current2024"]["_meta"]["kind"],
                "citation",
            )
            self.assertIn(
                store.data["citations"]["current2024"]["_meta"]["confidence_label"],
                {"low", "medium", "high"},
            )
            self.assertIn("confidence_summary", store.data["metadata"])
            self.assertIn("@misc{current2024", store.bib_path.read_text(encoding="utf-8"))

    def test_apply_run_keeps_current_run_evidence_while_deleting_stale_entries(self) -> None:
        with TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            store = ConstitutionStore(base / "constitution.json", base / "constitution.bib")
            store.load("test topic")
            store.data["citations"] = {
                "stale2020": {
                    "cite_key": "stale2020",
                    "bibtex": "@misc{stale2020, title={Stale Citation}}",
                    "title": "Stale Citation",
                    "url": "https://example.com/stale",
                    "source_id": "stale-source",
                    "authors": [],
                    "year": "2020",
                    "doi": "",
                }
            }
            store.data["findings"] = [
                {
                    "finding_id": "stale-finding",
                    "claim": "Old claim",
                    "evidence": "Old evidence",
                    "citation_keys": ["stale2020"],
                }
            ]

            source_notes = [
                SourceNote(
                    source_id="current-source",
                    title="Current Paper",
                    url="https://example.com/current",
                    citation_key="current2024",
                    summary="Current source summary.",
                )
            ]
            citations = [
                CitationRecord(
                    cite_key="current2024",
                    bibtex="@misc{current2024, title={Current Citation}}",
                    title="Current Citation",
                    url="https://example.com/current",
                    source_id="current-source",
                )
            ]
            synthesis = SynthesisResult(
                title="Current Topic",
                abstract="Summary.",
                findings=[
                    Finding(
                        finding_id="current-finding",
                        claim="Current claim",
                        evidence="Current evidence",
                        citation_keys=["current2024"],
                    )
                ],
                notes=["Retained current evidence."],
                delete_citation_keys=["stale2020", "current2024"],
                delete_finding_ids=["stale-finding", "current-finding"],
            )

            store.apply_run(source_notes, citations, synthesis)

            self.assertNotIn("stale2020", store.data["citations"])
            self.assertIn("current2024", store.data["citations"])
            self.assertEqual(
                [finding["finding_id"] for finding in store.data["findings"]],
                ["current-finding"],
            )
            self.assertIn("Retained current evidence.", store.data["notes"])
            self.assertIn("current-source", store.data["source_notes"])
            self.assertEqual(store.data["metadata"]["last_checkpoint_stage"], "completed")
            self.assertEqual(store.data["findings"][0]["_meta"]["kind"], "finding")
            self.assertEqual(store.data["findings"][0]["_meta"]["citation_count"], 1)
            self.assertIn(
                store.data["findings"][0]["_meta"]["confidence_label"],
                {"low", "medium", "high"},
            )
            self.assertIsInstance(store.data["findings"][0]["_meta"]["confidence_reason"], str)
            self.assertIn("@misc{current2024", store.bib_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
