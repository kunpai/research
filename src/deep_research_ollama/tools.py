from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from urllib import error, parse, request

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    PdfReader = None

from deep_research_ollama.config import Settings
from deep_research_ollama.models import SearchResult, SourceDocument


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text + " ")

    def text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return unescape(text).strip()


class SearchToolkit:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._crossref_doi_cache: dict[str, dict] = {}
        self._crossref_title_cache: dict[str, dict | None] = {}

    def search(self, queries: list[str]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for query in queries:
            query_results: list[SearchResult] = []
            query_results.extend(self._search_arxiv(query))
            query_results.extend(self._search_semantic_scholar(query))
            query_results.extend(self._search_google_scholar(query))
            query_results.extend(self._search_crossref(query))
            if self.settings.google_api_key and self.settings.google_cse_id:
                query_results.extend(self._search_google_cse(query))
            elif self.settings.serpapi_api_key:
                query_results.extend(self._search_serpapi(query))
            else:
                query_results.extend(self._search_duckduckgo(query))
            results.extend(self._promote_results(query_results))
        return self._dedupe_results(results)

    def fetch_document(self, result: SearchResult) -> SourceDocument:
        if not result.url:
            text = result.abstract or result.snippet
            return SourceDocument(
                source_id=result.result_id,
                title=result.title,
                url=result.url,
                kind=result.kind,
                backend=result.backend,
                authors=list(result.authors),
                year=result.year,
                doi=result.doi,
                arxiv_id=result.arxiv_id,
                scholar_id=result.scholar_id,
                scholar_cite_url=result.scholar_cite_url,
                abstract=result.abstract,
                text=text,
                text_chunks=self.select_chunk_sample(self.chunk_text(text)),
            )
        if result.arxiv_id:
            return self._fetch_arxiv_document(result)
        if result.url.lower().endswith(".pdf"):
            return self._fetch_pdf_document(result)
        return self._fetch_web_document(result)

    def chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self.settings.chunk_chars)
            if end < len(normalized):
                pivot = normalized.rfind("\n\n", start + self.settings.chunk_chars // 2, end)
                if pivot != -1:
                    end = pivot
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(0, end - self.settings.chunk_overlap_chars)
        return chunks

    def select_chunk_sample(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= self.settings.max_chunks_per_source:
            return chunks
        if self.settings.max_chunks_per_source == 1:
            return [chunks[0]]
        last_index = len(chunks) - 1
        sample_indices = sorted(
            {
                round(i * last_index / (self.settings.max_chunks_per_source - 1))
                for i in range(self.settings.max_chunks_per_source)
            }
        )
        return [chunks[index] for index in sample_indices]

    def _search_arxiv(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://export.arxiv.org/api/query?"
            + parse.urlencode(
                {"search_query": f"all:{query}", "start": 0, "max_results": self.settings.max_paper_results_per_query}
            )
        )
        payload = self._fetch_text(endpoint)
        if not payload:
            return []
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            return []
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        results: list[SearchResult] = []
        for entry in root.findall("atom:entry", ns):
            id_url = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = id_url.rsplit("/", 1)[-1]
            title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
            abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=ns).split())
            authors = [
                author.findtext("atom:name", default="", namespaces=ns)
                for author in entry.findall("atom:author", ns)
            ]
            year = entry.findtext("atom:published", default="", namespaces=ns)[:4]
            doi = entry.findtext("arxiv:doi", default="", namespaces=ns)
            results.append(
                SearchResult(
                    result_id=f"arxiv:{arxiv_id}",
                    title=title,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    snippet=abstract[:320],
                    backend="arxiv",
                    kind="paper",
                    authors=[author for author in authors if author],
                    year=year,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    abstract=abstract,
                    matched_queries=[query],
                )
            )
        return results

    def _search_semantic_scholar(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            + parse.urlencode(
                {
                    "query": query,
                    "limit": self.settings.max_paper_results_per_query,
                    "fields": "title,abstract,url,year,authors,externalIds,citationCount",
                }
            )
        )
        headers = {"User-Agent": self.settings.user_agent}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key
        payload = self._fetch_json(endpoint, headers=headers)
        results: list[SearchResult] = []
        for item in payload.get("data", []):
            paper_id = item.get("paperId")
            title = item.get("title", "")
            abstract = item.get("abstract", "") or ""
            url = item.get("url", "") or (
                f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
            )
            authors = [author.get("name", "") for author in item.get("authors", [])]
            external_ids = item.get("externalIds", {}) or {}
            doi = external_ids.get("DOI", "") or ""
            results.append(
                SearchResult(
                    result_id=f"semanticscholar:{paper_id or len(results)}",
                    title=title,
                    url=url,
                    snippet=abstract[:320],
                    backend="semantic_scholar",
                    kind="paper",
                    authors=[author for author in authors if author],
                    year=str(item.get("year", "") or ""),
                    doi=doi,
                    abstract=abstract,
                    citation_count=int(item.get("citationCount", 0) or 0),
                    matched_queries=[query],
                )
            )
        return results

    def _search_crossref(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://api.crossref.org/works?"
            + parse.urlencode(
                {
                    "query.bibliographic": query,
                    "rows": self.settings.max_paper_results_per_query,
                    "select": "DOI,title,author,abstract,published-print,published-online,URL",
                }
            )
        )
        payload = self._fetch_json(endpoint)
        results: list[SearchResult] = []
        for item in payload.get("message", {}).get("items", []):
            parsed = self._crossref_item_to_result(item, matched_queries=[query])
            if parsed is not None:
                results.append(parsed)
        return results

    def _search_google_scholar(self, query: str) -> list[SearchResult]:
        if not self.settings.enable_google_scholar:
            return []
        results: list[SearchResult] = []
        if self.settings.serpapi_api_key:
            results.extend(self._search_serpapi_google_scholar(query))
        if results:
            return results
        return self._search_google_scholar_html(query)

    def _promote_results(self, results: list[SearchResult]) -> list[SearchResult]:
        promoted: list[SearchResult] = []
        for result in results:
            promoted.append(self._promote_result(result))
        return promoted

    def _promote_result(self, result: SearchResult) -> SearchResult:
        if result.backend == "google_scholar":
            arxiv_id = self._extract_arxiv_id(result.url) or self._extract_arxiv_id(
                " ".join([result.title, result.snippet, result.abstract])
            )
            if arxiv_id:
                promoted = SearchResult(
                    result_id=f"arxiv:{arxiv_id}",
                    title=self._clean_result_title(result.title),
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    snippet=result.snippet,
                    backend="arxiv",
                    kind="paper",
                    authors=list(result.authors),
                    year=result.year,
                    doi=result.doi,
                    arxiv_id=arxiv_id,
                    abstract=result.abstract or result.snippet,
                    citation_count=result.citation_count,
                    matched_queries=list(result.matched_queries),
                    scholar_id=result.scholar_id,
                    scholar_cite_url=result.scholar_cite_url,
                )
                return self._merge_results(promoted, result)

            doi = self._extract_doi(result.url) or self._extract_doi(
                " ".join([result.title, result.snippet, result.abstract])
            )
            if doi:
                promoted = self._crossref_result_for_doi(doi, result.matched_queries)
                if promoted is not None:
                    promoted.url = self._preferred_publisher_url(result.url, promoted.url, doi)
                    return self._merge_results(promoted, result)

            promoted = self._crossref_result_for_title(result.title, result.matched_queries)
            if promoted is not None:
                promoted.url = self._preferred_publisher_url(result.url, promoted.url, promoted.doi)
                return self._merge_results(promoted, result)

            return result

        if result.kind != "web":
            return result

        arxiv_id = self._extract_arxiv_id(result.url) or self._extract_arxiv_id(
            " ".join([result.title, result.snippet])
        )
        if arxiv_id:
            promoted = SearchResult(
                result_id=f"arxiv:{arxiv_id}",
                title=self._clean_result_title(result.title),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                snippet=result.snippet,
                backend="arxiv",
                kind="paper",
                arxiv_id=arxiv_id,
                abstract=result.snippet,
                matched_queries=list(result.matched_queries),
            )
            return self._merge_results(promoted, result)

        doi = self._extract_doi(result.url) or self._extract_doi(
            " ".join([result.title, result.snippet])
        )
        if doi:
            promoted = self._crossref_result_for_doi(doi, result.matched_queries)
            if promoted is not None:
                promoted.url = self._preferred_publisher_url(result.url, promoted.url, doi)
                return self._merge_results(promoted, result)
            fallback = SearchResult(
                result_id=f"crossref:{doi}",
                title=self._clean_result_title(result.title),
                url=self._preferred_publisher_url(result.url, "", doi),
                snippet=result.snippet,
                backend="crossref",
                kind="paper",
                doi=doi,
                abstract=result.snippet,
                matched_queries=list(result.matched_queries),
            )
            return self._merge_results(fallback, result)

        if self._looks_like_paper_landing_page(result):
            promoted = self._crossref_result_for_title(result.title, result.matched_queries)
            if promoted is not None:
                promoted.url = self._preferred_publisher_url(result.url, promoted.url, promoted.doi)
                return self._merge_results(promoted, result)
            if self._has_strong_paper_landing_signal(result):
                fallback = SearchResult(
                    result_id=f"{result.backend}-paper:{self._title_key(result) or result.result_id}",
                    title=self._clean_result_title(result.title),
                    url=result.url,
                    snippet=result.snippet,
                    backend=result.backend,
                    kind="paper",
                    abstract=result.snippet,
                    matched_queries=list(result.matched_queries),
                )
                return self._merge_results(fallback, result)

        return result

    def _search_google_cse(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://www.googleapis.com/customsearch/v1?"
            + parse.urlencode(
                {
                    "key": self.settings.google_api_key,
                    "cx": self.settings.google_cse_id,
                    "q": query,
                    "num": self.settings.max_web_results_per_query,
                }
            )
        )
        payload = self._fetch_json(endpoint)
        results: list[SearchResult] = []
        for index, item in enumerate(payload.get("items", []), start=1):
            results.append(
                SearchResult(
                    result_id=f"google:{query}:{index}",
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    backend="google_cse",
                    kind="web",
                    matched_queries=[query],
                )
            )
        return results

    def _search_serpapi(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://serpapi.com/search.json?"
            + parse.urlencode(
                {
                    "engine": "google",
                    "q": query,
                    "num": self.settings.max_web_results_per_query,
                    "api_key": self.settings.serpapi_api_key,
                }
            )
        )
        payload = self._fetch_json(endpoint)
        results: list[SearchResult] = []
        for index, item in enumerate(payload.get("organic_results", []), start=1):
            results.append(
                SearchResult(
                    result_id=f"serpapi:{query}:{index}",
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    backend="serpapi",
                    kind="web",
                    matched_queries=[query],
                )
            )
        return results

    def _search_serpapi_google_scholar(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://serpapi.com/search.json?"
            + parse.urlencode(
                {
                    "engine": "google_scholar",
                    "q": query,
                    "num": self.settings.max_paper_results_per_query,
                    "hl": "en",
                    "api_key": self.settings.serpapi_api_key,
                }
            )
        )
        payload = self._fetch_json(endpoint)
        results: list[SearchResult] = []
        for index, item in enumerate(payload.get("organic_results", []), start=1):
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue
            publication_info = item.get("publication_info", {}) or {}
            publication_summary = str(publication_info.get("summary", "") or "")
            authors = [
                str(author.get("name", "")).strip()
                for author in publication_info.get("authors", []) or []
                if str(author.get("name", "")).strip()
            ]
            if not authors:
                authors = self._parse_scholar_authors(publication_summary)
            inline_links = item.get("inline_links", {}) or {}
            cited_by = inline_links.get("cited_by", {}) or {}
            citation_count = int(cited_by.get("total", 0) or 0)
            resources = item.get("resources", []) or []
            resource_link = ""
            for resource in resources:
                candidate = str(resource.get("link", "") or "").strip()
                if candidate:
                    resource_link = candidate
                    break
            scholar_id = str(item.get("result_id", "") or "")
            url = str(item.get("link", "") or "").strip() or resource_link
            snippet = str(item.get("snippet", "") or "")
            results.append(
                SearchResult(
                    result_id=f"google_scholar:{scholar_id or index}",
                    title=title,
                    url=url,
                    snippet=snippet,
                    backend="google_scholar",
                    kind="paper",
                    authors=authors,
                    year=self._parse_scholar_year(publication_summary),
                    doi=self._extract_doi(" ".join([url, snippet, publication_summary])),
                    abstract=snippet,
                    citation_count=citation_count,
                    matched_queries=[query],
                    scholar_id=scholar_id,
                    scholar_cite_url=self._build_google_scholar_cite_url(
                        scholar_id, rank=index - 1
                    ),
                )
            )
            if len(results) >= self.settings.max_paper_results_per_query:
                break
        return results

    def _search_google_scholar_html(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://scholar.google.com/scholar?"
            + parse.urlencode({"hl": "en", "as_sdt": "0,5", "q": query})
        )
        html = self._fetch_text(endpoint, headers=self._google_scholar_headers())
        if not html or self._looks_like_google_scholar_challenge(html):
            return []
        results: list[SearchResult] = []
        for index, block in enumerate(self._split_google_scholar_blocks(html), start=1):
            parsed_result = self._parse_google_scholar_block(
                block, query=query, rank=index
            )
            if parsed_result is None:
                continue
            results.append(parsed_result)
            if len(results) >= self.settings.max_paper_results_per_query:
                break
        return results

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        endpoint = (
            "https://duckduckgo.com/html/?"
            + parse.urlencode({"q": query, "kl": "us-en"})
        )
        html = self._fetch_text(endpoint)
        if not html:
            return []

        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
            re.S,
        )
        results: list[SearchResult] = []
        for index, match in enumerate(pattern.finditer(html), start=1):
            url = self._clean_duckduckgo_url(match.group("url"))
            title = self._strip_tags(match.group("title"))
            snippet = self._strip_tags(match.group("snippet"))
            results.append(
                SearchResult(
                    result_id=f"duckduckgo:{query}:{index}",
                    title=title,
                    url=url,
                    snippet=snippet,
                    backend="duckduckgo",
                    kind="web",
                    matched_queries=[query],
                )
            )
            if index >= self.settings.max_web_results_per_query:
                break
        return results

    def _fetch_arxiv_document(self, result: SearchResult) -> SourceDocument:
        pdf_url = f"https://arxiv.org/pdf/{result.arxiv_id}.pdf"
        text = result.abstract
        pdf_bytes = self._fetch_bytes(pdf_url)
        if pdf_bytes and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = []
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    if extracted:
                        pages.append(extracted.strip())
                if pages:
                    text = "\n\n".join(pages)
            except Exception:
                text = result.abstract
        text = text[: self.settings.max_source_chars]
        return SourceDocument(
            source_id=result.result_id,
            title=result.title,
            url=result.url,
            kind=result.kind,
            backend=result.backend,
            authors=list(result.authors),
            year=result.year,
            doi=result.doi,
            arxiv_id=result.arxiv_id,
            scholar_id=result.scholar_id,
            scholar_cite_url=result.scholar_cite_url,
            abstract=result.abstract,
            text=text,
            text_chunks=self.select_chunk_sample(self.chunk_text(text)),
        )

    def _fetch_pdf_document(self, result: SearchResult) -> SourceDocument:
        text = result.snippet
        pdf_bytes = self._fetch_bytes(result.url)
        if pdf_bytes and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages = []
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    if extracted:
                        pages.append(extracted.strip())
                if pages:
                    text = "\n\n".join(pages)
            except Exception:
                pass
        text = text[: self.settings.max_source_chars]
        return SourceDocument(
            source_id=result.result_id,
            title=result.title,
            url=result.url,
            kind=result.kind,
            backend=result.backend,
            authors=list(result.authors),
            year=result.year,
            doi=result.doi,
            arxiv_id=result.arxiv_id,
            scholar_id=result.scholar_id,
            scholar_cite_url=result.scholar_cite_url,
            abstract=result.abstract,
            text=text,
            text_chunks=self.select_chunk_sample(self.chunk_text(text)),
        )

    def _fetch_web_document(self, result: SearchResult) -> SourceDocument:
        html = self._fetch_text(result.url)
        parser = VisibleTextParser()
        if html:
            parser.feed(html)
        text = parser.text() or result.snippet
        text = text[: self.settings.max_source_chars]
        return SourceDocument(
            source_id=result.result_id,
            title=result.title,
            url=result.url,
            kind=result.kind,
            backend=result.backend,
            authors=list(result.authors),
            year=result.year,
            doi=result.doi,
            arxiv_id=result.arxiv_id,
            scholar_id=result.scholar_id,
            scholar_cite_url=result.scholar_cite_url,
            abstract=result.abstract,
            text=text,
            text_chunks=self.select_chunk_sample(self.chunk_text(text)),
        )

    def _fetch_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        req_headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain;q=0.9,*/*;q=0.8",
        }
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

    def _fetch_bytes(self, url: str) -> bytes:
        req = request.Request(
            url,
            headers={"User-Agent": self.settings.user_agent},
            method="GET",
        )
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                return response.read()
        except error.URLError:
            return b""

    def _fetch_json(self, url: str, headers: dict[str, str] | None = None) -> dict:
        req_headers = {"User-Agent": self.settings.user_agent}
        if headers:
            req_headers.update(headers)
        req = request.Request(url, headers=req_headers, method="GET")
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _clean_duckduckgo_url(url: str) -> str:
        parsed = parse.urlparse(url)
        params = parse.parse_qs(parsed.query)
        redirected = params.get("uddg", [])
        return redirected[0] if redirected else url

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
    def _looks_like_google_scholar_challenge(html: str) -> bool:
        lowered = html.lower()
        return any(
            marker in lowered
            for marker in (
                "unusual traffic",
                "please show you're not a robot",
                "/sorry/",
                "recaptcha",
            )
        )

    @staticmethod
    def _split_google_scholar_blocks(html: str) -> list[str]:
        blocks = re.split(r'(?=<div class="gs_r gs_or gs_scl")', html)
        return [
            block
            for block in blocks
            if block.startswith('<div class="gs_r gs_or gs_scl"')
        ]

    def _parse_google_scholar_block(
        self,
        block: str,
        *,
        query: str,
        rank: int,
    ) -> SearchResult | None:
        scholar_id_match = re.search(r'data-cid="([^"]+)"', block)
        scholar_id = scholar_id_match.group(1).strip() if scholar_id_match else ""

        title_match = re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', block, re.S)
        if not title_match:
            return None
        title_html = title_match.group(1)
        title_link_match = re.search(r'href="([^"]+)"', title_html)
        title = self._clean_result_title(self._strip_tags(title_html))
        if not title:
            return None
        url = (
            self._absolute_google_scholar_url(title_link_match.group(1))
            if title_link_match
            else ""
        )

        snippet_match = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', block, re.S)
        snippet = self._strip_tags(snippet_match.group(1)) if snippet_match else ""
        meta_match = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', block, re.S)
        meta = self._strip_tags(meta_match.group(1)) if meta_match else ""

        resource_link_match = re.search(
            r'<div class="gs_or_ggsm"[^>]*>.*?<a[^>]*href="([^"]+)"',
            block,
            re.S,
        )
        if not url and resource_link_match:
            url = self._absolute_google_scholar_url(resource_link_match.group(1))

        citation_match = re.search(r'>Cited by\s+([0-9,]+)<', block, re.I)
        citation_count = (
            int(citation_match.group(1).replace(",", "")) if citation_match else 0
        )

        return SearchResult(
            result_id=f"google_scholar:{scholar_id or rank}",
            title=title,
            url=url,
            snippet=snippet,
            backend="google_scholar",
            kind="paper",
            authors=self._parse_scholar_authors(meta),
            year=self._parse_scholar_year(meta),
            doi=self._extract_doi(" ".join([url, title, snippet, meta])),
            abstract=snippet,
            citation_count=citation_count,
            matched_queries=[query],
            scholar_id=scholar_id,
            scholar_cite_url=self._build_google_scholar_cite_url(scholar_id, rank=rank - 1),
        )

    @staticmethod
    def _absolute_google_scholar_url(url: str) -> str:
        cleaned = unescape(url or "").strip()
        if not cleaned:
            return ""
        return parse.urljoin("https://scholar.google.com", cleaned)

    @staticmethod
    def _build_google_scholar_cite_url(scholar_id: str, *, rank: int = 0) -> str:
        if not scholar_id:
            return ""
        return "https://scholar.google.com/scholar?" + parse.urlencode(
            {
                "q": f"info:{scholar_id}:scholar.google.com/",
                "output": "cite",
                "scirp": max(0, rank),
                "hl": "en",
            }
        )

    @staticmethod
    def _parse_scholar_year(value: str) -> str:
        match = re.search(r"\b(?:19|20)\d{2}\b", value or "")
        return match.group(0) if match else ""

    @staticmethod
    def _parse_scholar_authors(value: str) -> list[str]:
        cleaned = (value or "").replace("…", "").strip()
        if not cleaned:
            return []
        prefix = cleaned.split(" - ", 1)[0]
        parts = [part.strip() for part in prefix.replace("&", ",").split(",")]
        authors: list[str] = []
        for part in parts:
            lowered = part.lower()
            if not part or lowered in {"et al.", "et al", "..."}:
                continue
            if any(
                marker in lowered
                for marker in (
                    "arxiv",
                    "journal",
                    "conference",
                    "proceedings",
                    "preprint",
                )
            ):
                continue
            if re.search(r"\b(?:19|20)\d{2}\b", part):
                continue
            authors.append(part)
        return authors

    @staticmethod
    def _strip_tags(value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value)
        return unescape(value).strip()

    def _crossref_result_for_doi(
        self,
        doi: str,
        matched_queries: list[str],
    ) -> SearchResult | None:
        normalized = doi.strip().lower()
        if not normalized:
            return None
        if normalized not in self._crossref_doi_cache:
            endpoint = (
                "https://api.crossref.org/works/"
                f"{parse.quote(doi, safe='')}"
            )
            payload = self._fetch_json(endpoint)
            self._crossref_doi_cache[normalized] = payload.get("message", {}) if payload else {}
        item = self._crossref_doi_cache.get(normalized) or {}
        if not item:
            return None
        return self._crossref_item_to_result(item, matched_queries=matched_queries)

    def _crossref_result_for_title(
        self,
        title: str,
        matched_queries: list[str],
    ) -> SearchResult | None:
        normalized_title = self._clean_result_title(title)
        if not normalized_title:
            return None
        expected_terms = self._title_terms(normalized_title)
        if len(expected_terms) < 3:
            return None
        cache_key = normalized_title.lower()
        if cache_key not in self._crossref_title_cache:
            endpoint = (
                "https://api.crossref.org/works?"
                + parse.urlencode({"query.title": normalized_title, "rows": 5})
            )
            payload = self._fetch_json(endpoint)
            items = payload.get("message", {}).get("items", [])
            best_item: dict | None = None
            best_score = 0.0
            for item in items:
                candidate_title = " ".join(item.get("title") or []).strip()
                candidate_terms = self._title_terms(candidate_title)
                overlap = len(expected_terms & candidate_terms)
                score = self._title_match_score(normalized_title, candidate_title)
                minimum_overlap = 3 if len(expected_terms) >= 4 else 2
                if overlap < minimum_overlap and score < 0.9:
                    continue
                if score > best_score:
                    best_score = score
                    best_item = item
            self._crossref_title_cache[cache_key] = best_item if best_score >= 0.72 else None
        item = self._crossref_title_cache.get(cache_key)
        if not item:
            return None
        return self._crossref_item_to_result(item, matched_queries=matched_queries)

    def _crossref_item_to_result(
        self,
        item: dict,
        *,
        matched_queries: list[str],
    ) -> SearchResult | None:
        doi = str(item.get("DOI", "") or "").strip()
        title = " ".join(item.get("title") or []).strip()
        if not title:
            return None
        abstract = self._strip_tags(item.get("abstract", "") or "")
        authors = []
        for author in item.get("author", []):
            given = str(author.get("given", "")).strip()
            family = str(author.get("family", "")).strip()
            name = " ".join(part for part in [given, family] if part).strip()
            if name:
                authors.append(name)
        year = ""
        for field in ("published-print", "published-online", "published"):
            parts = item.get(field, {}).get("date-parts", [])
            if parts and parts[0]:
                year = str(parts[0][0])
                break
        url = item.get("URL", "") or (f"https://doi.org/{doi}" if doi else "")
        return SearchResult(
            result_id=f"crossref:{doi or title.lower()}",
            title=title,
            url=url,
            snippet=(abstract or title)[:320],
            backend="crossref",
            kind="paper",
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            matched_queries=list(matched_queries),
        )

    @staticmethod
    def _preferred_publisher_url(original_url: str, canonical_url: str, doi: str) -> str:
        doi_lower = doi.strip().lower()
        original = original_url.strip()
        if original:
            lowered = original.lower()
            if doi_lower and doi_lower in lowered:
                return original
            if any(
                domain in lowered
                for domain in (
                    "dl.acm.org",
                    "ieeexplore.ieee.org",
                    "link.springer.com",
                    "sciencedirect.com",
                    "aclanthology.org",
                    "openreview.net",
                    "proceedings.mlr.press",
                )
            ):
                return original
        return canonical_url or original

    @staticmethod
    def _extract_doi(value: str) -> str:
        if not value:
            return ""
        match = re.search(
            r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
            parse.unquote(value),
            re.I,
        )
        if not match:
            return ""
        doi = match.group(1).rstrip(").,;:]}")
        return doi

    @staticmethod
    def _extract_arxiv_id(value: str) -> str:
        if not value:
            return ""
        match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", value, re.I)
        return match.group(1) if match else ""

    def _looks_like_paper_landing_page(self, result: SearchResult) -> bool:
        url = result.url.lower()
        domain = parse.urlparse(url).netloc.lower()
        title = result.title.lower()
        snippet = result.snippet.lower()
        if url.endswith(".pdf"):
            return True
        if domain in {
            "dl.acm.org",
            "ieeexplore.ieee.org",
            "link.springer.com",
            "www.nature.com",
            "nature.com",
            "www.sciencedirect.com",
            "sciencedirect.com",
            "aclanthology.org",
            "openreview.net",
            "proceedings.mlr.press",
            "par.nsf.gov",
        }:
            return True
        return any(
            marker in title or marker in snippet
            for marker in (
                "conference on",
                "journal of",
                "proceedings",
                "paper presents",
                "this paper",
                "abstract:",
                "doi",
            )
        )

    def _has_strong_paper_landing_signal(self, result: SearchResult) -> bool:
        url = result.url.lower()
        domain = parse.urlparse(url).netloc.lower()
        if url.endswith(".pdf") or self._extract_doi(url):
            return True
        return domain in {
            "dl.acm.org",
            "ieeexplore.ieee.org",
            "link.springer.com",
            "www.nature.com",
            "nature.com",
            "www.sciencedirect.com",
            "sciencedirect.com",
            "aclanthology.org",
            "openreview.net",
            "proceedings.mlr.press",
            "par.nsf.gov",
        }

    @staticmethod
    def _clean_result_title(title: str) -> str:
        cleaned = SearchToolkit._strip_tags(title)
        cleaned = re.sub(r"^\s*pdf\s+", "", cleaned, flags=re.I)
        cleaned = re.sub(r"^\s*github\s*-\s*[^:]+:\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+([:;,?.!])", r"\1", cleaned)
        cleaned = re.sub(r"\s*\.\.\.\s*", " ", cleaned)
        cleaned = re.sub(
            r"\s*[-|]\s*(github|wikipedia|arxiv\.org|nsf public access|pubmed central|pmc)\s*$",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _title_match_score(expected: str, candidate: str) -> float:
        expected_terms = SearchToolkit._title_terms(expected)
        candidate_terms = SearchToolkit._title_terms(candidate)
        if not expected_terms or not candidate_terms:
            return 0.0
        overlap = len(expected_terms & candidate_terms)
        containment = overlap / max(1, min(len(expected_terms), len(candidate_terms)))
        exactish = overlap / max(len(expected_terms), len(candidate_terms))
        prefix_bonus = 0.15 if candidate.lower().startswith(expected.lower()) or expected.lower().startswith(candidate.lower()) else 0.0
        return max(exactish, containment * 0.85 + prefix_bonus)

    @staticmethod
    def _title_terms(value: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", SearchToolkit._clean_result_title(value).lower())
            if len(term) > 2
        }

    @staticmethod
    def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
        deduped: dict[str, SearchResult] = {}
        aliases: dict[str, str] = {}
        for result in results:
            key = SearchToolkit._result_key(result)
            if not key:
                continue
            title_key = SearchToolkit._title_key(result) if result.kind == "paper" else ""
            canonical_key = aliases.get(key) or (aliases.get(title_key) if title_key else None)
            if canonical_key is None:
                canonical_key = SearchToolkit._equivalent_title_key(result, deduped) or key
            existing = deduped.get(canonical_key)
            if existing is None:
                deduped[canonical_key] = result
                aliases[key] = canonical_key
                if title_key:
                    aliases[title_key] = canonical_key
                continue
            merged = SearchToolkit._merge_results(existing, result)
            deduped[canonical_key] = merged
            aliases[key] = canonical_key
            if title_key:
                aliases[title_key] = canonical_key
        return list(deduped.values())

    @staticmethod
    def _result_key(result: SearchResult) -> str:
        if result.doi:
            return f"doi:{result.doi.strip().lower()}"
        if result.arxiv_id:
            return f"arxiv:{result.arxiv_id.strip().lower()}"
        if result.kind == "paper":
            title = SearchToolkit._title_key(result)
            if title:
                return title
        url = result.url.strip().lower()
        return f"url:{url}" if url else ""

    @staticmethod
    def _title_key(result: SearchResult) -> str:
        title = re.sub(
            r"[^a-z0-9]+",
            " ",
            SearchToolkit._clean_result_title(result.title).lower(),
        ).strip()
        return f"title:{title}" if title else ""

    @staticmethod
    def _equivalent_title_key(
        result: SearchResult,
        deduped: dict[str, SearchResult],
    ) -> str | None:
        for canonical_key, existing in deduped.items():
            if SearchToolkit._same_work_by_title(existing, result):
                return canonical_key
        return None

    @staticmethod
    def _merge_results(left: SearchResult, right: SearchResult) -> SearchResult:
        primary, secondary = sorted(
            [left, right],
            key=SearchToolkit._result_quality_key,
            reverse=True,
        )
        share_text = SearchToolkit._content_compatible(primary, secondary)
        authoritative_paper = (
            primary.kind == "paper"
            and primary.backend in {"crossref", "semantic_scholar", "arxiv"}
        )
        merged_queries = SearchToolkit._dedupe_strings(
            list(primary.matched_queries) + list(secondary.matched_queries)
        )
        critic_relevant = (
            primary.critic_relevant
            if primary.critic_relevant is not None
            else secondary.critic_relevant
        )
        critic_reason = (
            primary.critic_reason
            if primary.critic_reason
            else secondary.critic_reason
        )
        critic_query = (
            primary.critic_query
            if primary.critic_query
            else secondary.critic_query
        )
        return SearchResult(
            result_id=primary.result_id,
            title=primary.title or secondary.title,
            url=SearchToolkit._preferred_url(primary, secondary),
            snippet=(
                secondary.snippet
                if share_text
                and not (authoritative_paper and primary.snippet)
                and len(secondary.snippet) > len(primary.snippet)
                else primary.snippet
            ),
            backend=primary.backend,
            kind=primary.kind,
            authors=primary.authors if len(primary.authors) >= len(secondary.authors) else secondary.authors,
            year=primary.year or secondary.year,
            doi=primary.doi or secondary.doi,
            arxiv_id=primary.arxiv_id or secondary.arxiv_id,
            abstract=(
                secondary.abstract
                if share_text
                and not (authoritative_paper and primary.abstract)
                and len(secondary.abstract) > len(primary.abstract)
                else primary.abstract
            ),
            citation_count=max(primary.citation_count, secondary.citation_count),
            matched_queries=merged_queries,
            scholar_id=primary.scholar_id or secondary.scholar_id,
            scholar_cite_url=primary.scholar_cite_url or secondary.scholar_cite_url,
            critic_relevant=critic_relevant,
            critic_reason=critic_reason,
            critic_query=critic_query,
        )

    @staticmethod
    def _content_compatible(left: SearchResult, right: SearchResult) -> bool:
        if left.doi and right.doi and left.doi.strip().lower() == right.doi.strip().lower():
            return True
        if left.arxiv_id and right.arxiv_id and left.arxiv_id.strip().lower() == right.arxiv_id.strip().lower():
            return True
        left_title = SearchToolkit._clean_result_title(left.title)
        right_title = SearchToolkit._clean_result_title(right.title)
        if not left_title or not right_title:
            return False
        overlap = len(SearchToolkit._title_terms(left_title) & SearchToolkit._title_terms(right_title))
        return overlap >= 2 and SearchToolkit._title_match_score(left_title, right_title) >= 0.72

    @staticmethod
    def _same_work_by_title(left: SearchResult, right: SearchResult) -> bool:
        if left.doi and right.doi and left.doi.strip().lower() == right.doi.strip().lower():
            return True
        if left.arxiv_id and right.arxiv_id and left.arxiv_id.strip().lower() == right.arxiv_id.strip().lower():
            return True
        if left.kind != "paper" and right.kind != "paper":
            return False

        left_title = SearchToolkit._clean_result_title(left.title)
        right_title = SearchToolkit._clean_result_title(right.title)
        if not left_title or not right_title:
            return False

        left_terms = SearchToolkit._title_terms(left_title)
        right_terms = SearchToolkit._title_terms(right_title)
        if not left_terms or not right_terms:
            return False

        overlap = len(left_terms & right_terms)
        minimum_overlap = max(4, min(len(left_terms), len(right_terms)) - 1)
        if overlap < minimum_overlap:
            return False

        score = SearchToolkit._title_match_score(left_title, right_title)
        if score >= 0.86:
            return True

        return (
            overlap >= 5
            and (
                left_title.lower().startswith(right_title.lower())
                or right_title.lower().startswith(left_title.lower())
            )
        )

    @staticmethod
    def _result_quality_key(result: SearchResult) -> tuple[int, int, int, int, int]:
        backend_rank = {
            "semantic_scholar": 5,
            "arxiv": 4,
            "google_scholar": 3,
            "crossref": 2,
            "google_cse": 1,
            "serpapi": 1,
            "duckduckgo": 0,
        }.get(result.backend, 0)
        kind_rank = 2 if result.kind == "paper" else 0
        abstract_rank = 1 if result.abstract else 0
        doi_rank = 1 if result.doi else 0
        return (
            kind_rank,
            backend_rank,
            abstract_rank,
            result.citation_count,
            doi_rank,
        )

    @staticmethod
    def _preferred_url(primary: SearchResult, secondary: SearchResult) -> str:
        for candidate in [primary, secondary]:
            if candidate.arxiv_id:
                return f"https://arxiv.org/abs/{candidate.arxiv_id}"
            if candidate.doi:
                return candidate.url or f"https://doi.org/{candidate.doi}"
            if candidate.url:
                return candidate.url
        return ""

    @staticmethod
    def _dedupe_strings(items: list[str]) -> list[str]:
        seen = set()
        output = []
        for item in items:
            cleaned = str(item).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            output.append(cleaned)
        return output
