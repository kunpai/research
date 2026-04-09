from __future__ import annotations

import json
from typing import Any


def schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=True, separators=(",", ":"))


def clarifier_schema(max_questions: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                    },
                    "required": ["id", "question"],
                },
                "maxItems": max_questions,
            },
            "assumptions": _string_array_schema(),
        },
        "required": ["questions", "assumptions"],
    }


def planner_schema(max_queries: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rewritten_question": {"type": "string"},
            "queries": _string_array_schema(max_items=max_queries),
            "related_topics": _string_array_schema(),
            "focus_areas": _string_array_schema(),
            "must_cover": _string_array_schema(),
            "source_requirements": _string_array_schema(),
        },
        "required": [
            "rewritten_question",
            "queries",
            "related_topics",
            "focus_areas",
            "must_cover",
            "source_requirements",
        ],
    }


def retrieval_strategy_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "anchor_phrases": _string_array_schema(max_items=10),
            "search_facets": _string_array_schema(max_items=10),
            "generic_terms": _string_array_schema(max_items=16),
            "concept_groups": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "phrases": _string_array_schema(max_items=6),
                        "priority": {"type": "integer"},
                    },
                    "required": ["label", "phrases", "priority"],
                },
            },
        },
        "required": [
            "anchor_phrases",
            "search_facets",
            "generic_terms",
            "concept_groups",
        ],
    }


def selector_schema(max_sources: int, candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_ids": {
                "type": "array",
                "items": _enum_or_string_schema(candidate_ids),
                "maxItems": max_sources,
            },
            "discarded_topics": _string_array_schema(),
        },
        "required": ["selected_ids", "discarded_topics"],
    }


def relevance_critic_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "judgments": {
                "type": "array",
                "maxItems": len(candidate_ids) if candidate_ids else 0,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "result_id": _enum_or_string_schema(candidate_ids),
                        "relevant": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["result_id", "relevant", "reason"],
                },
            }
        },
        "required": ["judgments"],
    }


def source_note_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "claims": _string_array_schema(),
            "evidence_snippets": _string_array_schema(),
            "related_topics": _string_array_schema(),
        },
        "required": ["summary", "claims", "evidence_snippets", "related_topics"],
    }


def collaboration_turn_schema(citation_keys: list[str]) -> dict[str, Any]:
    citation_key_schema = _enum_or_string_schema(citation_keys)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string"},
                        "citation_keys": {"type": "array", "items": citation_key_schema},
                        "status": {
                            "type": "string",
                            "enum": ["supported", "tentative", "challenged", "rejected"],
                        },
                    },
                    "required": ["claim", "citation_keys", "status"],
                },
            },
            "criticisms": _string_array_schema(),
            "open_questions": _string_array_schema(),
            "messages_to_next": _string_array_schema(),
        },
        "required": [
            "summary",
            "claims",
            "criticisms",
            "open_questions",
            "messages_to_next",
        ],
    }


def collaboration_session_schema(citation_keys: list[str]) -> dict[str, Any]:
    citation_key_schema = _enum_or_string_schema(citation_keys)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "consensus_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string"},
                        "citation_keys": {"type": "array", "items": citation_key_schema},
                        "status": {"type": "string", "enum": ["supported", "tentative"]},
                    },
                    "required": ["claim", "citation_keys", "status"],
                },
            },
            "disputed_claims": _string_array_schema(),
            "open_questions": _string_array_schema(),
            "coordinator_notes": _string_array_schema(),
        },
        "required": [
            "consensus_claims",
            "disputed_claims",
            "open_questions",
            "coordinator_notes",
        ],
    }


def writer_schema(citation_keys: list[str]) -> dict[str, Any]:
    citation_key_schema = _enum_or_string_schema(citation_keys)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "abstract": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "heading": {"type": "string"},
                        "paragraphs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "citation_keys": {
                                        "type": "array",
                                        "items": citation_key_schema,
                                    },
                                },
                                "required": ["text", "citation_keys"],
                            },
                        },
                    },
                    "required": ["heading", "paragraphs"],
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "finding_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "evidence": {"type": "string"},
                        "citation_keys": {
                            "type": "array",
                            "items": citation_key_schema,
                        },
                    },
                    "required": ["finding_id", "claim", "evidence", "citation_keys"],
                },
            },
            "notes": _string_array_schema(),
            "delete_citation_keys": _string_array_schema(),
            "delete_finding_ids": _string_array_schema(),
        },
        "required": [
            "title",
            "abstract",
            "sections",
            "findings",
            "notes",
            "delete_citation_keys",
            "delete_finding_ids",
        ],
    }


def _string_array_schema(*, max_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _enum_or_string_schema(values: list[str]) -> dict[str, Any]:
    cleaned = [value for value in values if value]
    if cleaned:
        return {"type": "string", "enum": cleaned}
    return {"type": "string"}
