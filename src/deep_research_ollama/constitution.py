from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deep_research_ollama.models import CitationRecord, Finding, SourceNote, SynthesisResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConstitutionStore:
    def __init__(self, json_path: Path, bib_path: Path) -> None:
        self.json_path = json_path
        self.bib_path = bib_path
        self.data: dict[str, Any] = {}

    def load(self, topic: str) -> dict[str, Any]:
        if self.json_path.exists():
            self.data = json.loads(self.json_path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "topic": topic,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "metadata": {
                    "schema_version": 2,
                    "last_saved_at": "",
                    "last_checkpoint_stage": "",
                    "last_resumed_at": "",
                    "resume_from_status": "",
                    "resume_count": 0,
                },
                "notes": [],
                "findings": [],
                "citations": {},
                "source_notes": {},
            }
        self._ensure_store_metadata(topic)
        return self.data

    def save(self, *, checkpoint_stage: str | None = None) -> None:
        self._refresh_confidence_metadata()
        now = utc_now()
        self.data["updated_at"] = now
        metadata = self._ensure_store_metadata(self.data.get("topic", ""))
        metadata["last_saved_at"] = now
        if checkpoint_stage is not None:
            metadata["last_checkpoint_stage"] = checkpoint_stage
        self.json_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.bib_path.write_text(self.export_bibtex(), encoding="utf-8")

    def prompt_snapshot(self, limit: int = 12) -> dict[str, Any]:
        citations = list(self.data.get("citations", {}).keys())[-limit:]
        findings = self.data.get("findings", [])[-limit:]
        notes = self.data.get("notes", [])[-limit:]
        return {"citations": citations, "findings": findings, "notes": notes}

    def apply_run(
        self,
        source_notes: list[SourceNote],
        citations: list[CitationRecord],
        synthesis: SynthesisResult,
    ) -> None:
        self._upsert_source_notes(source_notes)

        self.delete_citations(synthesis.delete_citation_keys)
        self.delete_findings(synthesis.delete_finding_ids)

        self._upsert_citations(citations)
        self._upsert_findings(synthesis.findings)
        self._append_notes(synthesis.notes)
        self.save(checkpoint_stage="completed")

    def checkpoint_sources(
        self,
        source_notes: list[SourceNote],
        citations: list[CitationRecord],
        *,
        checkpoint_stage: str = "sources",
    ) -> None:
        self._upsert_source_notes(source_notes)
        self._upsert_citations(citations)
        self.save(checkpoint_stage=checkpoint_stage)

    def mark_resumed(self, status: str) -> None:
        metadata = self._ensure_store_metadata(self.data.get("topic", ""))
        metadata["resume_count"] = int(metadata.get("resume_count", 0) or 0) + 1
        metadata["last_resumed_at"] = utc_now()
        metadata["resume_from_status"] = status
        self.save(checkpoint_stage=metadata.get("last_checkpoint_stage", ""))

    def delete_citations(self, cite_keys: list[str]) -> None:
        if not cite_keys:
            return
        citation_map = self.data.setdefault("citations", {})
        for key in cite_keys:
            citation_map.pop(key, None)
        now = utc_now()
        for finding in self.data.setdefault("findings", []):
            current = finding.get("citation_keys", [])
            finding["citation_keys"] = [key for key in current if key not in cite_keys]
            metadata = finding.setdefault("_meta", {})
            metadata["updated_at"] = now
            metadata["citation_count"] = len(finding["citation_keys"])

    def delete_findings(self, finding_ids: list[str]) -> None:
        if not finding_ids:
            return
        self.data["findings"] = [
            finding
            for finding in self.data.setdefault("findings", [])
            if finding.get("finding_id") not in set(finding_ids)
        ]

    def export_bibtex(self) -> str:
        citation_map = self.data.get("citations", {})
        entries = [entry.get("bibtex", "").strip() for _, entry in sorted(citation_map.items())]
        return "\n\n".join(entry for entry in entries if entry) + ("\n" if entries else "")

    def _upsert_citations(self, citations: list[CitationRecord]) -> None:
        citation_map = self.data.setdefault("citations", {})
        for citation in citations:
            payload = citation.to_dict()
            citation_map[citation.cite_key] = self._with_record_metadata(
                citation_map.get(citation.cite_key),
                payload,
                kind="citation",
                extra={
                    "source_id": citation.source_id,
                    "citation_key": citation.cite_key,
                    "author_count": len(citation.authors),
                    "has_doi": bool(citation.doi),
                    "year": citation.year,
                },
            )

    def _upsert_source_notes(self, source_notes: list[SourceNote]) -> None:
        note_map = self.data.setdefault("source_notes", {})
        for note in source_notes:
            payload = note.to_dict()
            note_map[note.source_id] = self._with_record_metadata(
                note_map.get(note.source_id),
                payload,
                kind="source_note",
                extra={
                    "source_id": note.source_id,
                    "citation_key": note.citation_key,
                    "claim_count": len(note.claims),
                    "related_topic_count": len(note.related_topics),
                    "evidence_count": len(note.evidence_snippets),
                    "summary_chars": len(note.summary),
                },
            )

    def _upsert_findings(self, findings: list[Finding]) -> None:
        by_id = {
            item.get("finding_id"): item
            for item in self.data.setdefault("findings", [])
            if item.get("finding_id")
        }
        for finding in findings:
            payload = asdict(finding)
            by_id[finding.finding_id] = self._with_record_metadata(
                by_id.get(finding.finding_id),
                payload,
                kind="finding",
                extra={
                    "finding_id": finding.finding_id,
                    "citation_count": len(finding.citation_keys),
                },
            )
        self.data["findings"] = list(by_id.values())

    def _append_notes(self, notes: list[str]) -> None:
        seen = set(self.data.setdefault("notes", []))
        for note in notes:
            if note and note not in seen:
                self.data["notes"].append(note)
                seen.add(note)

    def _ensure_store_metadata(self, topic: str) -> dict[str, Any]:
        if topic and not self.data.get("topic"):
            self.data["topic"] = topic
        metadata = self.data.setdefault("metadata", {})
        metadata.setdefault("schema_version", 2)
        metadata.setdefault("last_saved_at", "")
        metadata.setdefault("last_checkpoint_stage", "")
        metadata.setdefault("last_resumed_at", "")
        metadata.setdefault("resume_from_status", "")
        metadata.setdefault("resume_count", 0)
        return metadata

    def _refresh_confidence_metadata(self) -> None:
        citation_scores: dict[str, float] = {}
        note_scores: dict[str, float] = {}
        finding_scores: dict[str, float] = {}

        citations = self.data.setdefault("citations", {})
        for cite_key, entry in citations.items():
            score, label, reason = self._score_citation_entry(entry)
            meta = entry.setdefault("_meta", {})
            meta["confidence"] = score
            meta["confidence_label"] = label
            meta["confidence_reason"] = reason
            citation_scores[cite_key] = score

        source_notes = self.data.setdefault("source_notes", {})
        for source_id, entry in source_notes.items():
            score, label, reason = self._score_source_note_entry(entry, citation_scores)
            meta = entry.setdefault("_meta", {})
            meta["confidence"] = score
            meta["confidence_label"] = label
            meta["confidence_reason"] = reason
            note_scores[source_id] = score

        findings = self.data.setdefault("findings", [])
        for entry in findings:
            score, label, reason = self._score_finding_entry(entry, citation_scores)
            meta = entry.setdefault("_meta", {})
            meta["confidence"] = score
            meta["confidence_label"] = label
            meta["confidence_reason"] = reason
            finding_id = str(entry.get("finding_id", "")).strip()
            if finding_id:
                finding_scores[finding_id] = score

        metadata = self._ensure_store_metadata(self.data.get("topic", ""))
        metadata["confidence_summary"] = {
            "citations": self._confidence_summary(citation_scores.values()),
            "source_notes": self._confidence_summary(note_scores.values()),
            "findings": self._confidence_summary(finding_scores.values()),
        }

    @staticmethod
    def _with_record_metadata(
        existing: dict[str, Any] | None,
        payload: dict[str, Any],
        *,
        kind: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        record = dict(payload)
        previous_meta = dict(existing.get("_meta", {})) if existing else {}
        metadata = {
            "kind": kind,
            "status": "active",
            "created_at": previous_meta.get("created_at", now),
            "updated_at": now,
            "last_seen_at": now,
            "seen_count": int(previous_meta.get("seen_count", 0) or 0) + 1,
        }
        if extra:
            metadata.update(extra)
        if previous_meta:
            metadata = {**previous_meta, **metadata}
        record["_meta"] = metadata
        return record

    @staticmethod
    def _score_citation_entry(entry: dict[str, Any]) -> tuple[float, str, str]:
        score = 0.2
        reasons: list[str] = []
        doi = str(entry.get("doi", "")).strip()
        url = str(entry.get("url", "")).strip().lower()
        source_id = str(entry.get("source_id", "")).strip().lower()
        bibtex = str(entry.get("bibtex", ""))
        authors = entry.get("authors", []) or []
        year = str(entry.get("year", "")).strip()
        title = str(entry.get("title", "")).strip()

        if doi:
            score += 0.35
            reasons.append("DOI-backed citation")
        if "archiveprefix = {arxiv}" in bibtex.lower() or "arxiv.org" in url or source_id.startswith(
            "arxiv:"
        ):
            score += 0.2
            reasons.append("arXiv-backed citation")
        if authors:
            score += 0.1
            reasons.append("author metadata present")
        if year:
            score += 0.05
            reasons.append("publication year present")
        if title:
            score += 0.05
        if url.startswith("https://doi.org/") or "arxiv.org" in url:
            score += 0.05

        score = ConstitutionStore._clamp_confidence(score)
        label = ConstitutionStore._confidence_label(score)
        reason = "; ".join(reasons[:3]) or "limited citation metadata"
        return score, label, reason

    @staticmethod
    def _score_source_note_entry(
        entry: dict[str, Any],
        citation_scores: dict[str, float],
    ) -> tuple[float, str, str]:
        score = 0.15
        reasons: list[str] = []
        citation_key = str(entry.get("citation_key", "")).strip()
        summary = str(entry.get("summary", "")).strip()
        claims = entry.get("claims", []) or []
        related_topics = entry.get("related_topics", []) or []
        evidence_snippets = entry.get("evidence_snippets", []) or []

        if citation_key and citation_key in citation_scores:
            citation_score = citation_scores[citation_key]
            score += 0.35 * citation_score
            reasons.append(
                f"grounded by {ConstitutionStore._confidence_label(citation_score)}-confidence citation"
            )
        elif citation_key:
            score += 0.1
        else:
            score -= 0.05

        if evidence_snippets:
            score += min(0.2, 0.06 * len(evidence_snippets))
            reasons.append("source evidence snippets captured")
        else:
            score -= 0.08
        if claims:
            score += min(0.15, 0.05 * len(claims))
        if related_topics:
            score += min(0.08, 0.02 * len(related_topics))
        if len(summary) >= 180:
            score += 0.1
        elif len(summary) >= 80:
            score += 0.05
        elif len(summary) < 30:
            score -= 0.05

        score = ConstitutionStore._clamp_confidence(score)
        label = ConstitutionStore._confidence_label(score)
        reason = "; ".join(reasons[:3]) or "thin source-note evidence"
        return score, label, reason

    @staticmethod
    def _score_finding_entry(
        entry: dict[str, Any],
        citation_scores: dict[str, float],
    ) -> tuple[float, str, str]:
        score = 0.1
        reasons: list[str] = []
        citation_keys = [str(key).strip() for key in entry.get("citation_keys", []) if str(key).strip()]
        claim = str(entry.get("claim", "")).strip()
        evidence = str(entry.get("evidence", "")).strip()

        if citation_keys:
            score += min(0.3, 0.12 * len(citation_keys))
            known_scores = [citation_scores[key] for key in citation_keys if key in citation_scores]
            if known_scores:
                avg_score = sum(known_scores) / len(known_scores)
                score += 0.4 * avg_score
                reasons.append(
                    f"supported by {len(citation_keys)} citation(s) with {ConstitutionStore._confidence_label(avg_score)} average confidence"
                )
        else:
            score -= 0.1

        if len(evidence) >= 120:
            score += 0.1
            reasons.append("substantial evidence text present")
        elif len(evidence) >= 40:
            score += 0.05
        if ConstitutionStore._text_overlap(claim, evidence) >= 2:
            score += 0.05
        if not claim or not evidence:
            score -= 0.08

        score = ConstitutionStore._clamp_confidence(score)
        label = ConstitutionStore._confidence_label(score)
        reason = "; ".join(reasons[:3]) or "limited direct support"
        return score, label, reason

    @staticmethod
    def _confidence_summary(scores: Any) -> dict[str, Any]:
        values = [float(score) for score in scores]
        if not values:
            return {"count": 0, "mean": 0.0, "high": 0, "medium": 0, "low": 0}
        labels = [ConstitutionStore._confidence_label(score) for score in values]
        return {
            "count": len(values),
            "mean": round(sum(values) / len(values), 2),
            "high": sum(1 for label in labels if label == "high"),
            "medium": sum(1 for label in labels if label == "medium"),
            "low": sum(1 for label in labels if label == "low"),
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    @staticmethod
    def _clamp_confidence(score: float) -> float:
        return round(min(0.99, max(0.05, score)), 2)

    @staticmethod
    def _text_overlap(left: str, right: str) -> int:
        left_terms = {term for term in re.findall(r"[a-z0-9]+", left.lower()) if len(term) > 2}
        right_terms = {term for term in re.findall(r"[a-z0-9]+", right.lower()) if len(term) > 2}
        return len(left_terms & right_terms)
