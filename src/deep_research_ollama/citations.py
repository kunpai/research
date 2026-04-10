from __future__ import annotations

import json
import re
from datetime import datetime
from urllib import error, parse, request

from deep_research_ollama.config import Settings
from deep_research_ollama.models import CitationRecord, SourceDocument


class CitationResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(
        self, source: SourceDocument, existing_keys: set[str] | None = None
    ) -> CitationRecord:
        existing_keys = existing_keys or set()
        bibtex = ""

        doi = source.doi
        if not doi and not source.arxiv_id:
            doi = self._lookup_doi_by_title(source.title)
        if doi:
            bibtex = self._coerce_valid_bibtex(self._fetch_bibtex_for_doi(doi))

        if not bibtex and (source.scholar_cite_url or source.scholar_id):
            bibtex = self._fetch_bibtex_from_google_scholar(source)

        if not bibtex and source.arxiv_id:
            bibtex = self._build_arxiv_bibtex(source)

        if not bibtex:
            bibtex = self._build_web_bibtex(source)

        bibtex = self._normalize_bibtex(bibtex)
        if not self.is_valid_bibtex(bibtex):
            bibtex = self._normalize_bibtex(self._build_fallback_bibtex(source))
        cite_key = self._extract_cite_key(bibtex) or self._make_base_cite_key(source)
        cite_key = self._dedupe_cite_key(cite_key, existing_keys)
        bibtex = self._replace_cite_key(bibtex, cite_key)

        return CitationRecord(
            cite_key=cite_key,
            bibtex=bibtex,
            title=source.title,
            url=source.url,
            source_id=source.source_id,
            authors=list(source.authors),
            year=source.year,
            doi=doi or source.doi,
        )

    def _lookup_doi_by_title(self, title: str) -> str:
        if not title:
            return ""
        endpoint = (
            "https://api.crossref.org/works?"
            + parse.urlencode({"query.title": title, "rows": 5})
        )
        payload = self._get_json(endpoint)
        items = payload.get("message", {}).get("items", [])
        best_doi = ""
        best_score = 0.0
        for item in items:
            candidate_title = " ".join(item.get("title") or []).strip()
            score = self._title_match_score(title, candidate_title)
            if score > best_score:
                best_score = score
                best_doi = item.get("DOI", "") or ""
        return best_doi if best_score >= 0.6 else ""

    def _fetch_bibtex_for_doi(self, doi: str) -> str:
        endpoint = (
            "https://api.crossref.org/works/"
            f"{parse.quote(doi, safe='')}/transform/application/x-bibtex"
        )
        req = request.Request(
            endpoint,
            headers={"User-Agent": self.settings.user_agent},
            method="GET",
        )
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                return response.read().decode("utf-8").strip()
        except error.URLError:
            return ""

    def _get_json(self, url: str) -> dict:
        req = request.Request(
            url,
            headers={"User-Agent": self.settings.user_agent},
            method="GET",
        )
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, json.JSONDecodeError):
            return {}

    def _fetch_bibtex_from_google_scholar(self, source: SourceDocument) -> str:
        cite_url = source.scholar_cite_url or self._build_google_scholar_cite_url(
            source.scholar_id
        )
        if not cite_url:
            return ""
        cite_html = self._get_text(cite_url, headers=self._google_scholar_headers())
        if not cite_html:
            return ""
        bibtex_match = re.search(
            r'href="(https://scholar\.googleusercontent\.com/scholar\.bib[^"]+)"',
            cite_html,
            re.I,
        )
        if not bibtex_match:
            return ""
        bibtex_url = bibtex_match.group(1)
        return self._coerce_valid_bibtex(
            self._get_text(
                bibtex_url,
                headers=self._google_scholar_headers(),
            )
        )

    def _build_arxiv_bibtex(self, source: SourceDocument) -> str:
        key = self._make_base_cite_key(source)
        authors = " and ".join(source.authors) if source.authors else "Unknown"
        year = source.year or str(datetime.utcnow().year)
        title = self._escape_field(source.title)
        return (
            f"@misc{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            f"  eprint = {{{source.arxiv_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  url = {{{source.url}}}\n"
            "}"
        )

    def _build_web_bibtex(self, source: SourceDocument) -> str:
        key = self._make_base_cite_key(source)
        year = source.year or str(datetime.utcnow().year)
        title = self._escape_field(source.title)
        return (
            f"@misc{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  year = {{{year}}},\n"
            f"  howpublished = {{\\url{{{source.url}}}}},\n"
            f"  note = {{Accessed {datetime.utcnow().date().isoformat()}}}\n"
            "}"
        )

    def _build_fallback_bibtex(self, source: SourceDocument) -> str:
        if source.arxiv_id:
            return self._build_arxiv_bibtex(source)
        return self._build_web_bibtex(source)

    @staticmethod
    def _extract_cite_key(bibtex: str) -> str:
        match = re.match(r"@\w+\{([^,]+),", bibtex.strip())
        return match.group(1).strip() if match else ""

    @staticmethod
    def _replace_cite_key(bibtex: str, cite_key: str) -> str:
        return re.sub(
            r"(@\w+\{)([^,]+)(,)",
            lambda match: f"{match.group(1)}{cite_key}{match.group(3)}",
            bibtex,
            count=1,
        )

    @classmethod
    def _coerce_valid_bibtex(cls, bibtex: str) -> str:
        candidate = bibtex.strip()
        return candidate if cls.is_valid_bibtex(candidate) else ""

    @staticmethod
    def _looks_like_html_document(value: str) -> bool:
        lowered = value.lstrip().lower()
        return (
            lowered.startswith("<!doctype html")
            or lowered.startswith("<html")
            or "<html" in lowered[:512]
        )

    @classmethod
    def is_valid_bibtex(cls, bibtex: str) -> bool:
        candidate = bibtex.strip()
        if not candidate or cls._looks_like_html_document(candidate):
            return False
        if not re.match(r"^@\w+\s*\{\s*[^,\s][^,]*,", candidate):
            return False
        return "=" in candidate

    @classmethod
    def _normalize_bibtex(cls, bibtex: str) -> str:
        if not bibtex.strip():
            return bibtex

        normalized = (
            bibtex.replace("\u2013", "--")
            .replace("\u2014", "---")
            .replace("\u2212", "-")
            .replace("\u00a0", " ")
        )

        lines = normalized.splitlines()
        if len(lines) > 1:
            output = [lines[0]]
            for line in lines[1:]:
                output.append(cls._normalize_bibtex_line(line))
            return "\n".join(output)

        match = re.match(r"^(@\w+\{[^,]+,)(.*)(\}\s*)$", normalized.strip(), re.S)
        if not match:
            return normalized
        prefix, body, suffix = match.groups()
        body = cls._normalize_bibtex_fields_inline(body)
        return prefix + body + suffix

    @classmethod
    def _normalize_bibtex_fields_inline(cls, body: str) -> str:
        pattern = re.compile(r"(\s*[A-Za-z][A-Za-z0-9_-]*\s*=\s*)(\{.*?\}|\".*?\"|[^,]+)(\s*,?)", re.S)

        def repl(match: re.Match[str]) -> str:
            prefix, value, suffix = match.groups()
            return prefix + cls._normalize_bibtex_value(value.strip()) + suffix

        return pattern.sub(repl, body)

    @classmethod
    def _normalize_bibtex_line(cls, line: str) -> str:
        if "=" not in line:
            return line.replace("\u2013", "--").replace("\u2014", "---").replace("\u2212", "-")
        prefix, value = line.split("=", 1)
        trailing_comma = "," if value.rstrip().endswith(",") else ""
        core = value.rstrip()
        if trailing_comma:
            core = core[:-1]
        normalized_value = cls._normalize_bibtex_value(core.strip())
        return prefix + "= " + normalized_value + trailing_comma

    @staticmethod
    def _normalize_bibtex_value(value: str) -> str:
        normalized = (
            value.replace("\u2013", "--")
            .replace("\u2014", "---")
            .replace("\u2212", "-")
            .replace("\u00a0", " ")
        )
        if len(normalized) >= 2 and normalized[0] == "{" and normalized[-1] == "}":
            inner = normalized[1:-1]
            inner = re.sub(r"(?<!\\)_", r"\\_", inner)
            return "{" + inner + "}"
        if len(normalized) >= 2 and normalized[0] == '"' and normalized[-1] == '"':
            inner = normalized[1:-1]
            inner = re.sub(r"(?<!\\)_", r"\\_", inner)
            return '"' + inner + '"'
        return re.sub(r"(?<!\\)_", r"\\_", normalized)

    def _make_base_cite_key(self, source: SourceDocument) -> str:
        surname = "source"
        if source.authors:
            surname = re.sub(r"[^a-z0-9]+", "", source.authors[0].split()[-1].lower()) or surname
        year = source.year or "nd"
        words = re.findall(r"[a-z0-9]+", source.title.lower())
        title_word = next((word for word in words if len(word) > 3), "ref")
        return f"{surname}{year}{title_word}"

    @staticmethod
    def _dedupe_cite_key(base: str, existing_keys: set[str]) -> str:
        if base not in existing_keys:
            return base
        suffix = 2
        while f"{base}{suffix}" in existing_keys:
            suffix += 1
        return f"{base}{suffix}"

    @staticmethod
    def _escape_field(value: str) -> str:
        return value.replace("{", "\\{").replace("}", "\\}")

    def _get_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        req_headers = {"User-Agent": self.settings.user_agent}
        if headers:
            req_headers.update(headers)
        req = request.Request(
            url,
            headers=req_headers,
            method="GET",
        )
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                return response.read().decode("utf-8", errors="ignore")
        except error.URLError:
            return ""

    @staticmethod
    def _google_scholar_headers() -> dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://scholar.google.com/",
        }

    @staticmethod
    def _build_google_scholar_cite_url(scholar_id: str) -> str:
        if not scholar_id:
            return ""
        return "https://scholar.google.com/scholar?" + parse.urlencode(
            {
                "q": f"info:{scholar_id}:scholar.google.com/",
                "output": "cite",
                "scirp": 0,
                "hl": "en",
            }
        )

    @staticmethod
    def _title_match_score(expected: str, candidate: str) -> float:
        expected_terms = CitationResolver._title_terms(expected)
        candidate_terms = CitationResolver._title_terms(candidate)
        if not expected_terms or not candidate_terms:
            return 0.0
        overlap = len(expected_terms & candidate_terms)
        return overlap / max(len(expected_terms), len(candidate_terms))

    @staticmethod
    def _title_terms(value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", value.lower())
            if len(term) > 2
        }
