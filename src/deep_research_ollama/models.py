from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchResult:
    result_id: str
    title: str
    url: str
    snippet: str
    backend: str
    kind: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    citation_count: int = 0
    matched_queries: list[str] = field(default_factory=list)
    scholar_id: str = ""
    scholar_cite_url: str = ""
    critic_relevant: bool | None = None
    critic_reason: str = ""
    critic_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceDocument:
    source_id: str
    title: str
    url: str
    kind: str
    backend: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""
    arxiv_id: str = ""
    scholar_id: str = ""
    scholar_cite_url: str = ""
    abstract: str = ""
    text: str = ""
    text_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CitationRecord:
    cite_key: str
    bibtex: str
    title: str
    url: str
    source_id: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    doi: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    finding_id: str
    claim: str
    evidence: str
    citation_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceNote:
    source_id: str
    title: str
    url: str
    citation_key: str
    summary: str
    claims: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchPlan:
    queries: list[str]
    related_topics: list[str]
    focus_areas: list[str]
    rewritten_question: str = ""
    must_cover: list[str] = field(default_factory=list)
    source_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalConceptGroup:
    label: str
    phrases: list[str] = field(default_factory=list)
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalStrategy:
    anchor_phrases: list[str] = field(default_factory=list)
    search_facets: list[str] = field(default_factory=list)
    generic_terms: list[str] = field(default_factory=list)
    concept_groups: list[RetrievalConceptGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_phrases": list(self.anchor_phrases),
            "search_facets": list(self.search_facets),
            "generic_terms": list(self.generic_terms),
            "concept_groups": [group.to_dict() for group in self.concept_groups],
        }


@dataclass
class ReportSection:
    heading: str
    paragraphs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesisResult:
    title: str
    abstract: str
    sections: list[ReportSection] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    delete_citation_keys: list[str] = field(default_factory=list)
    delete_finding_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "sections": [section.to_dict() for section in self.sections],
            "findings": [finding.to_dict() for finding in self.findings],
            "notes": list(self.notes),
            "delete_citation_keys": list(self.delete_citation_keys),
            "delete_finding_ids": list(self.delete_finding_ids),
        }


@dataclass
class CollaborationTurn:
    role: str
    summary: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    criticisms: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    messages_to_next: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollaborationSession:
    turns: list[CollaborationTurn] = field(default_factory=list)
    consensus_claims: list[dict[str, Any]] = field(default_factory=list)
    disputed_claims: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    coordinator_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "consensus_claims": list(self.consensus_claims),
            "disputed_claims": list(self.disputed_claims),
            "open_questions": list(self.open_questions),
            "coordinator_notes": list(self.coordinator_notes),
        }
