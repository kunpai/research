from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from deep_research_ollama.config import Settings
from deep_research_ollama.providers import (
    default_api_base,
    provider_api_env_overrides,
    provider_metadata,
    provider_names,
    suggested_models,
)


def list_available_models(settings: Settings, provider: str) -> list[str]:
    normalized = provider.strip().lower() or settings.llm_provider
    if normalized == "ollama":
        models = _list_models_from_manifests()
        if models:
            return models
        return _list_models_from_ollama_cli(timeout_seconds=2)
    return suggested_models(normalized)


def build_run_command(
    *,
    topic: str,
    output_dir: Path,
    answers: dict[str, str],
    provider: str,
    model: str,
    api_base: str,
    max_summary_model_calls: int | None,
    max_queries: int | None,
    max_total_queries: int | None,
    max_search_rounds: int | None,
    max_web_results_per_query: int | None,
    max_paper_results_per_query: int | None,
    max_selected_sources: int | None,
    max_critic_results: int | None,
    max_chunks_per_source: int | None,
    no_clarify: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "deep_research_ollama",
        "run",
        topic,
        "--output-dir",
        str(output_dir),
    ]
    if provider.strip():
        command.extend(["--provider", provider.strip().lower()])
    if model.strip():
        command.extend(["--model", model.strip()])
    if api_base.strip():
        command.extend(["--api-base", api_base.strip()])
    if max_summary_model_calls is not None:
        command.extend(["--max-summary-model-calls", str(int(max_summary_model_calls))])
    if max_queries is not None:
        command.extend(["--max-queries", str(int(max_queries))])
    if max_total_queries is not None:
        command.extend(["--max-total-queries", str(int(max_total_queries))])
    if max_search_rounds is not None:
        command.extend(["--max-search-rounds", str(int(max_search_rounds))])
    if max_web_results_per_query is not None:
        command.extend(["--max-web-results-per-query", str(int(max_web_results_per_query))])
    if max_paper_results_per_query is not None:
        command.extend(["--max-paper-results-per-query", str(int(max_paper_results_per_query))])
    if max_selected_sources is not None:
        command.extend(["--max-selected-sources", str(int(max_selected_sources))])
    if max_critic_results is not None:
        command.extend(["--max-critic-results", str(int(max_critic_results))])
    if max_chunks_per_source is not None:
        command.extend(["--max-chunks-per-source", str(int(max_chunks_per_source))])
    if no_clarify:
        command.append("--no-clarify")
    for key, value in answers.items():
        if key and value:
            command.extend(["--answer", f"{key}={value}"])
    return command


def collect_run_status(
    *,
    output_dir: Path,
    settings: Settings,
    process: subprocess.Popen[str] | None = None,
    process_provider: str = "",
    process_model: str = "",
    process_budget: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    artifacts = artifact_paths(output_dir, settings)
    run_payload = _read_json(artifacts["run"])
    constitution_payload = _read_json(artifacts["constitution"])

    active = process is not None and process.poll() is None
    exit_code = None if process is None else process.poll()
    stage = str(run_payload.get("status", "")).strip()
    if active:
        status = f"running:{stage or 'starting'}"
    elif stage:
        status = stage
    elif exit_code not in (None, 0):
        status = "failed"
    elif any(path.exists() for path in artifacts.values()):
        status = "idle"
    else:
        status = "empty"

    retrieval = run_payload.get("retrieval", {}) if isinstance(run_payload, dict) else {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    budget = retrieval.get("budget", {})
    if not isinstance(budget, dict):
        budget = {}
    if process_budget is not None and "max_summary_model_calls" not in budget:
        budget = {"max_summary_model_calls": process_budget, **budget}
    constitution_metadata = (
        constitution_payload.get("metadata", {})
        if isinstance(constitution_payload, dict)
        else {}
    )
    confidence_summary = constitution_metadata.get("confidence_summary", {})
    process_state = {
        "active": active,
        "exit_code": exit_code,
    }
    if process is not None:
        process_state["pid"] = process.pid

    return {
        "output_dir": str(output_dir),
        "topic": str(run_payload.get("topic", "") or constitution_payload.get("topic", "")),
        "provider": str(run_payload.get("provider", "") or process_provider),
        "model": str(run_payload.get("model", "") or process_model),
        "status": status,
        "run_stage": stage,
        "progress": run_payload.get("progress", {}),
        "latex": run_payload.get("latex", {}) if isinstance(run_payload, dict) else {},
        "process": process_state,
        "artifacts": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
            for name, path in artifacts.items()
        },
        "counts": {
            "selected_sources": len(run_payload.get("selected_sources", [])),
            "citations": len(run_payload.get("citations", [])),
            "source_notes": len(run_payload.get("source_notes", [])),
            "findings": len(
                constitution_payload.get("findings", [])
                if isinstance(constitution_payload, dict)
                else []
            ),
        },
        "budget": budget,
        "constitution": {
            "metadata": constitution_metadata,
            "confidence_summary": confidence_summary,
        },
        "log_excerpt": _tail_text(artifacts["log"]),
    }


def artifact_paths(output_dir: Path, settings: Settings) -> dict[str, Path]:
    return {
        "program": output_dir / settings.program_filename,
        "report": output_dir / settings.report_filename,
        "references": output_dir / settings.references_filename,
        "run": output_dir / settings.run_filename,
        "retrieval": output_dir / settings.retrieval_filename,
        "constitution": output_dir / settings.constitution_filename,
        "constitution_bib": output_dir / settings.constitution_bib_filename,
        "pdf": output_dir / "report.pdf",
        "log": output_dir / "gui_run.log",
    }


def _list_models_from_ollama_cli(timeout_seconds: int = 5) -> list[str]:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return _parse_ollama_list_output(completed.stdout)


def _parse_ollama_list_output(output: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("name ") or stripped.lower() == "name":
            continue
        name = stripped.split()[0]
        if not name or name in seen:
            continue
        seen.add(name)
        models.append(name)
    return models


def _list_models_from_manifests(manifests_root: Path | None = None) -> list[str]:
    root = manifests_root or Path.home() / ".ollama" / "models" / "manifests"
    if not root.exists():
        return []

    models: list[str] = []
    seen: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        name = _manifest_path_to_model_name(relative)
        if not name or name in seen:
            continue
        seen.add(name)
        models.append(name)
    return sorted(models)


def _manifest_path_to_model_name(relative: Path) -> str:
    parts = list(relative.parts)
    if len(parts) < 2:
        return ""
    if "." in parts[0]:
        parts = parts[1:]
    if len(parts) < 2:
        return ""
    tag = parts[-1]
    base_parts = parts[:-1]
    if base_parts and base_parts[0] == "library":
        base_parts = base_parts[1:]
    if not base_parts:
        return ""
    return f"{'/'.join(base_parts)}:{tag}"


class GuiApp:
    def __init__(self, settings: Settings, output_root: Path, launch_cwd: Path) -> None:
        self.settings = settings
        self.output_root = output_root.expanduser().resolve()
        self.launch_cwd = launch_cwd
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.process_providers: dict[str, str] = {}
        self.process_models: dict[str, str] = {}
        self.process_budgets: dict[str, int] = {}

    def serve(self, host: str, port: int, *, open_browser: bool = False) -> None:
        handler = self._make_handler()
        server = HTTPServer((host, port), handler)
        url = f"http://{host}:{port}"
        print(f"Deep Research GUI listening on {url}")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Deep Research GUI.")
        finally:
            server.server_close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                app.handle_get(self)

            def do_POST(self) -> None:  # noqa: N802
                app.handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler

    def handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        if parsed.path == "/":
            self._send_html(handler, self.render_index())
            return
        if parsed.path == "/api/models":
            query = parse_qs(parsed.query)
            provider = str(query.get("provider", [self.settings.llm_provider])[0]).strip().lower()
            metadata = provider_metadata(provider)
            default_provider_base = (
                self.settings.llm_api_base
                if provider == self.settings.llm_provider and self.settings.llm_api_base.strip()
                else default_api_base(provider)
            )
            self._send_json(
                handler,
                {
                    "provider": provider,
                    "providers": [
                        {"id": name, "label": provider_metadata(name)["label"]}
                        for name in provider_names()
                    ],
                    "models": list_available_models(self.settings, provider),
                    "default_model": self.settings.llm_model,
                    "default_provider": self.settings.llm_provider,
                    "default_api_base": default_provider_base,
                    "default_max_summary_model_calls": self.settings.max_summary_model_calls,
                    "requires_api_key": bool(metadata.get("requires_api_key")),
                    "api_key_label": str(metadata.get("api_key_label", "API Key")),
                    "api_base_label": str(metadata.get("api_base_label", "API Base")),
                    "api_base_placeholder": str(metadata.get("api_base_placeholder", "")),
                    "model_hint": str(metadata.get("model_hint", "")),
                },
            )
            return
        if parsed.path == "/api/status":
            query = parse_qs(parsed.query)
            output_dir = self._resolve_output_dir(query.get("output_dir", [""])[0])
            process = self.processes.get(str(output_dir))
            process_provider = self.process_providers.get(str(output_dir), "")
            process_model = self.process_models.get(str(output_dir), "")
            process_budget = self.process_budgets.get(str(output_dir))
            self._send_json(
                handler,
                collect_run_status(
                    output_dir=output_dir,
                    settings=self.settings,
                    process=process,
                    process_provider=process_provider,
                    process_model=process_model,
                    process_budget=process_budget,
                ),
            )
            return
        if parsed.path == "/api/artifact":
            query = parse_qs(parsed.query)
            output_dir = self._resolve_output_dir(query.get("output_dir", [""])[0])
            name = str(query.get("name", [""])[0]).strip()
            self._send_json(handler, self._load_artifact(output_dir, name))
            return
        if parsed.path == "/artifact-file":
            query = parse_qs(parsed.query)
            output_dir = self._resolve_output_dir(query.get("output_dir", [""])[0])
            name = str(query.get("name", [""])[0]).strip()
            self._send_artifact_file(handler, output_dir, name)
            return
        self._send_json(handler, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        payload = self._read_json_body(handler)
        if parsed.path == "/api/run":
            self._start_run(handler, payload)
            return
        if parsed.path == "/api/stop":
            self._stop_run(handler, payload)
            return
        self._send_json(handler, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _start_run(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            self._send_json(
                handler,
                {"error": "Topic is required."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        output_dir = self._resolve_output_dir(str(payload.get("output_dir", "")).strip())
        process = self.processes.get(str(output_dir))
        if process is not None and process.poll() is None:
            self._send_json(
                handler,
                {"error": "A run is already active for that output directory."},
                status=HTTPStatus.CONFLICT,
            )
            return

        answers_payload = payload.get("answers", {})
        answers = {
            str(key).strip(): str(value).strip()
            for key, value in answers_payload.items()
            if str(key).strip() and str(value).strip()
        } if isinstance(answers_payload, dict) else {}
        no_clarify = bool(payload.get("no_clarify", True))
        provider = str(payload.get("provider", "")).strip().lower() or self.settings.llm_provider
        model = str(payload.get("model", "")).strip() or self.settings.llm_model
        api_base = str(payload.get("api_base", "")).strip() or default_api_base(provider)
        api_key = str(payload.get("api_key", "")).strip()
        serpapi_api_key = str(payload.get("serpapi_api_key", "")).strip()
        semantic_scholar_api_key = str(
            payload.get("semantic_scholar_api_key", "")
        ).strip()
        budget_value = payload.get("max_summary_model_calls", self.settings.max_summary_model_calls)
        try:
            max_summary_model_calls = max(1, int(budget_value))
        except (TypeError, ValueError):
            max_summary_model_calls = self.settings.max_summary_model_calls
        knob_values = {
            "max_queries": _coerce_positive_int(payload.get("max_queries"), self.settings.max_queries),
            "max_total_queries": _coerce_positive_int(
                payload.get("max_total_queries"),
                self.settings.max_total_queries,
            ),
            "max_search_rounds": _coerce_positive_int(
                payload.get("max_search_rounds"),
                self.settings.max_search_rounds,
            ),
            "max_web_results_per_query": _coerce_positive_int(
                payload.get("max_web_results_per_query"),
                self.settings.max_web_results_per_query,
            ),
            "max_paper_results_per_query": _coerce_positive_int(
                payload.get("max_paper_results_per_query"),
                self.settings.max_paper_results_per_query,
            ),
            "max_selected_sources": _coerce_positive_int(
                payload.get("max_selected_sources"),
                self.settings.max_selected_sources,
            ),
            "max_critic_results": _coerce_non_negative_int(
                payload.get("max_critic_results"),
                self.settings.max_critic_results,
            ),
            "max_chunks_per_source": _coerce_positive_int(
                payload.get("max_chunks_per_source"),
                self.settings.max_chunks_per_source,
            ),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = artifact_paths(output_dir, self.settings)["log"]
        log_handle = log_path.open("a", encoding="utf-8")
        command = build_run_command(
            topic=topic,
            output_dir=output_dir,
            answers=answers,
            provider=provider,
            model=model,
            api_base=api_base,
            max_summary_model_calls=max_summary_model_calls,
            max_queries=knob_values["max_queries"],
            max_total_queries=knob_values["max_total_queries"],
            max_search_rounds=knob_values["max_search_rounds"],
            max_web_results_per_query=knob_values["max_web_results_per_query"],
            max_paper_results_per_query=knob_values["max_paper_results_per_query"],
            max_selected_sources=knob_values["max_selected_sources"],
            max_critic_results=knob_values["max_critic_results"],
            max_chunks_per_source=knob_values["max_chunks_per_source"],
            no_clarify=no_clarify,
        )
        env = os.environ.copy()
        env.update(provider_api_env_overrides(provider, api_key, api_base))
        if serpapi_api_key:
            env["SERPAPI_API_KEY"] = serpapi_api_key
        if semantic_scholar_api_key:
            env["SEMANTIC_SCHOLAR_API_KEY"] = semantic_scholar_api_key
        process = subprocess.Popen(
            command,
            cwd=str(self.launch_cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log_handle.close()
        self.processes[str(output_dir)] = process
        self.process_providers[str(output_dir)] = provider
        self.process_models[str(output_dir)] = (
            f"{provider}/{model}" if model and not model.startswith(f"{provider}/") else model
        )
        self.process_budgets[str(output_dir)] = max_summary_model_calls
        self._send_json(
            handler,
            {
                "started": True,
                "pid": process.pid,
                "output_dir": str(output_dir),
                "provider": provider,
                "model": self.process_models[str(output_dir)],
                "max_summary_model_calls": max_summary_model_calls,
                "knobs": knob_values,
                "command": command,
            },
            status=HTTPStatus.ACCEPTED,
        )

    def _stop_run(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        output_dir = self._resolve_output_dir(str(payload.get("output_dir", "")).strip())
        process = self.processes.get(str(output_dir))
        if process is None or process.poll() is not None:
            self._send_json(
                handler,
                {"stopped": False, "error": "No active run for that output directory."},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        process.terminate()
        stopped = False
        try:
            process.wait(timeout=5)
            stopped = True
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            stopped = True
        self.process_providers.pop(str(output_dir), None)
        self.process_models.pop(str(output_dir), None)
        self.process_budgets.pop(str(output_dir), None)
        self._send_json(handler, {"stopped": stopped, "pid": process.pid})

    def _load_artifact(self, output_dir: Path, name: str) -> dict[str, Any]:
        paths = artifact_paths(output_dir, self.settings)
        if name not in paths:
            return {"error": f"Unknown artifact '{name}'."}
        path = paths[name]
        if not path.exists():
            return {"name": name, "path": str(path), "exists": False, "content": ""}
        if path.suffix.lower() == ".pdf":
            return {
                "name": name,
                "path": str(path),
                "exists": True,
                "content": "",
                "content_type": "application/pdf",
                "viewer_url": (
                    f"/artifact-file?output_dir={quote(str(output_dir), safe='')}"
                    f"&name={quote(name, safe='')}"
                ),
            }
        return {
            "name": name,
            "path": str(path),
            "exists": True,
            "content": path.read_text(encoding="utf-8", errors="replace"),
            "content_type": "text/plain; charset=utf-8",
        }

    def _resolve_output_dir(self, value: str) -> Path:
        if value:
            return Path(value).expanduser().resolve()
        return (self.output_root / "latest").resolve()

    @staticmethod
    def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = handler.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _send_html(handler: BaseHTTPRequestHandler, html: str) -> None:
        body = html.encode("utf-8")
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _send_artifact_file(
        self,
        handler: BaseHTTPRequestHandler,
        output_dir: Path,
        name: str,
    ) -> None:
        paths = artifact_paths(output_dir, self.settings)
        path = paths.get(name)
        if path is None or not path.exists():
            self._send_json(handler, {"error": "Artifact not found."}, status=HTTPStatus.NOT_FOUND)
            return
        if path.suffix.lower() == ".pdf":
            content_type = "application/pdf"
        else:
            content_type = "application/octet-stream"
        body = path.read_bytes()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def render_index(self) -> str:
        default_output = str((self.output_root / "latest").resolve())
        default_provider = self.settings.llm_provider
        default_model_display = _escape_html(self.settings.model_display_name())
        default_provider_model = _escape_html(self.settings.llm_model)
        default_api_base_value = self.settings.llm_api_base or default_api_base(default_provider)
        default_api_base_html = _escape_html(default_api_base_value)
        default_budget = str(self.settings.max_summary_model_calls)
        default_max_queries = str(self.settings.max_queries)
        default_max_total_queries = str(self.settings.max_total_queries)
        default_max_search_rounds = str(self.settings.max_search_rounds)
        default_max_web_results = str(self.settings.max_web_results_per_query)
        default_max_paper_results = str(self.settings.max_paper_results_per_query)
        default_max_selected_sources = str(self.settings.max_selected_sources)
        default_max_critic_results = str(self.settings.max_critic_results)
        default_max_chunks_per_source = str(self.settings.max_chunks_per_source)
        provider_info = provider_metadata(default_provider)
        provider_options = _provider_options_html(default_provider)
        model_options = _model_options_html(
            list_available_models(self.settings, default_provider),
            self.settings.llm_model,
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Deep Research GUI</title>
  <style>
    :root {{
      --bg: #f3ede3;
      --panel: rgba(255,255,255,0.74);
      --ink: #18212a;
      --muted: #5f6a73;
      --line: rgba(24,33,42,0.12);
      --accent: #b74d23;
      --accent-soft: #f1c9ae;
      --success: #265d46;
      --warning: #895714;
      --shadow: 0 18px 50px rgba(24,33,42,0.12);
      --font-display: "Avenir Next", "Gill Sans", "Segoe UI", sans-serif;
      --font-body: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      --radius: 22px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--font-body);
      background:
        radial-gradient(circle at top left, rgba(183,77,35,0.22), transparent 36%),
        radial-gradient(circle at bottom right, rgba(38,93,70,0.16), transparent 26%),
        linear-gradient(180deg, #f8f4ee 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1320px, calc(100vw - 40px));
      margin: 28px auto;
      display: grid;
      gap: 18px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}
    .hero {{
      padding: 26px 28px;
      display: grid;
      gap: 10px;
    }}
    .eyebrow {{
      font: 700 12px/1 var(--font-display);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    h1 {{
      margin: 0;
      font: 800 clamp(2rem, 3vw, 3.2rem)/0.95 var(--font-display);
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      max-width: 74ch;
      margin: 0;
      color: var(--muted);
      font-size: 1.02rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      gap: 18px;
    }}
    .panel {{
      padding: 22px;
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font: 700 1.05rem/1 var(--font-display);
      letter-spacing: 0.02em;
    }}
    .run-form {{
      display: grid;
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font: 700 0.84rem/1.2 var(--font-display);
      color: var(--muted);
      letter-spacing: 0.02em;
    }}
    input, textarea {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(24,33,42,0.14);
      background: rgba(255,255,255,0.84);
      color: var(--ink);
      font: 500 0.98rem/1.45 var(--font-body);
      resize: vertical;
    }}
    textarea {{ min-height: 88px; }}
    .model-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }}
    .model-row button {{
      white-space: nowrap;
    }}
    .slider-value {{
      color: var(--accent);
      font: 700 0.82rem/1 var(--font-display);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .knob-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid rgba(24,33,42,0.08);
      background: rgba(255,255,255,0.56);
    }}
    .knob-grid label {{
      gap: 4px;
    }}
    .knob-grid input {{
      padding: 10px 12px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 4px;
    }}
    .hidden {{
      display: none !important;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: 700 0.9rem/1 var(--font-display);
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease, background 140ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    button:disabled {{
      cursor: wait;
      opacity: 0.72;
      transform: none;
    }}
    .button-with-spinner {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .spinner {{
      width: 0.9rem;
      height: 0.9rem;
      border-radius: 50%;
      border: 2px solid currentColor;
      border-right-color: transparent;
      animation: spin 0.75s linear infinite;
      flex: 0 0 auto;
    }}
    .spinner.subtle {{
      width: 0.8rem;
      height: 0.8rem;
      border-width: 1.5px;
      opacity: 0.72;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .primary {{
      background: var(--ink);
      color: white;
    }}
    .secondary {{
      background: rgba(24,33,42,0.08);
      color: var(--ink);
    }}
    .danger {{
      background: #f4d2c5;
      color: #6e2612;
    }}
    .status-grid {{
      display: grid;
      gap: 14px;
    }}
    .status-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(183,77,35,0.09), rgba(24,33,42,0.05));
      border: 1px solid rgba(24,33,42,0.08);
    }}
    .status-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.8);
      font: 700 0.8rem/1 var(--font-display);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(24,33,42,0.08);
    }}
    .stat .k {{
      display: block;
      color: var(--muted);
      font: 700 0.74rem/1 var(--font-display);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .v {{
      display: block;
      margin-top: 8px;
      font: 800 1.2rem/1 var(--font-display);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      padding: 14px;
      border-radius: 18px;
      border: 1px solid rgba(24,33,42,0.08);
      background: rgba(255,255,255,0.68);
    }}
    .artifact-toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .artifact-button {{
      background: rgba(24,33,42,0.06);
      color: var(--ink);
    }}
    .artifact-loading {{
      margin-bottom: 12px;
      min-height: 1.2rem;
    }}
    .artifact-view {{
      min-height: 420px;
      max-height: 60vh;
      overflow: auto;
      border-radius: 18px;
      padding: 18px;
      background: #141a20;
      color: #eef4f7;
      border: 1px solid rgba(24,33,42,0.1);
      font: 500 0.9rem/1.55 "SFMono-Regular", "Menlo", monospace;
      white-space: pre-wrap;
    }}
    .artifact-embed {{
      width: 100%;
      min-height: 420px;
      height: 60vh;
      border: 1px solid rgba(24,33,42,0.1);
      border-radius: 18px;
      background: white;
    }}
    .small {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .inline {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .checkbox {{
      grid-template-columns: auto 1fr;
      align-items: center;
      gap: 10px;
    }}
    .checkbox input {{
      width: 18px;
      height: 18px;
      margin: 0;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Deep Research GUI</div>
      <h1>Run, resume, inspect, and debug local research runs.</h1>
      <p class="subtitle">This interface launches the existing pipeline in a background subprocess, tracks progress from checkpoint artifacts, and lets you inspect retrieval traces, constitutions, and generated reports from one place.</p>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Launch</h2>
        <form id="run-form" class="run-form">
          <label>Topic
            <textarea id="topic" name="topic" placeholder="AI for math proof assistants"></textarea>
          </label>
          <label>Output Directory
            <input id="output-dir" name="output_dir" value="{_escape_html(default_output)}">
          </label>
          <label>Provider
            <input id="provider" name="provider" list="provider-options" value="{_escape_html(default_provider)}" placeholder="ollama, gemini, openai, nvidia_nim..." spellcheck="false" autocomplete="off">
            <datalist id="provider-options">{provider_options}</datalist>
          </label>
          <div class="model-row">
            <label>Model
              <input id="model" name="model" list="model-options" value="{default_provider_model}" placeholder="Type any LiteLLM model name" spellcheck="false" autocomplete="off">
              <datalist id="model-options">{model_options}</datalist>
              <span class="small" id="model-hint">{_escape_html(str(provider_info.get("model_hint", "")))}</span>
            </label>
            <button class="secondary button-with-spinner" id="refresh-models-button" type="button">
              <span>Refresh Models</span>
              <span class="spinner hidden" id="refresh-models-spinner" aria-hidden="true"></span>
            </button>
          </div>
          <label id="api-base-label">{_escape_html(str(provider_info.get("api_base_label", "API Base")))}
            <input
              id="api-base"
              name="api_base"
              value="{default_api_base_html}"
              placeholder="{_escape_html(str(provider_info.get('api_base_placeholder', '')))}"
            >
          </label>
          <label id="api-key-row"{" class=\"hidden\"" if not provider_info.get("requires_api_key") else ""}>{_escape_html(str(provider_info.get("api_key_label", "API Key")))}
            <input id="api-key" name="api_key" type="password" placeholder="Used only for the launched subprocess; never written to run artifacts.">
          </label>
          <div class="knob-grid">
            <label>SerpAPI Key
              <input id="serpapi-api-key" name="serpapi_api_key" type="password" placeholder="Optional. Enables structured Google Scholar results.">
            </label>
            <label>Semantic Scholar Key
              <input id="semantic-scholar-api-key" name="semantic_scholar_api_key" type="password" placeholder="Optional. Improves Semantic Scholar quota/reliability.">
            </label>
          </div>
          <label>Summary Budget
            <div class="inline" style="justify-content:space-between;">
              <span class="small">Max summary model calls</span>
              <span class="slider-value" id="budget-slider-value">{default_budget}</span>
            </div>
            <input id="max-summary-model-calls" name="max_summary_model_calls" type="range" min="4" max="48" step="1" value="{default_budget}">
          </label>
          <div class="knob-grid">
            <label>Seed Queries
              <input id="max-queries" name="max_queries" type="number" min="1" step="1" value="{default_max_queries}">
            </label>
            <label>Total Queries
              <input id="max-total-queries" name="max_total_queries" type="number" min="1" step="1" value="{default_max_total_queries}">
            </label>
            <label>Search Rounds
              <input id="max-search-rounds" name="max_search_rounds" type="number" min="1" step="1" value="{default_max_search_rounds}">
            </label>
            <label>Selected Sources
              <input id="max-selected-sources" name="max_selected_sources" type="number" min="1" step="1" value="{default_max_selected_sources}">
            </label>
            <label>Web Results / Query
              <input id="max-web-results-per-query" name="max_web_results_per_query" type="number" min="1" step="1" value="{default_max_web_results}">
            </label>
            <label>Paper Results / Query
              <input id="max-paper-results-per-query" name="max_paper_results_per_query" type="number" min="1" step="1" value="{default_max_paper_results}">
            </label>
            <label>Critic Shortlist
              <input id="max-critic-results" name="max_critic_results" type="number" min="0" step="1" value="{default_max_critic_results}">
              <span class="small">Use <code>0</code> to judge the full shortlist.</span>
            </label>
            <label>Chunks / Source
              <input id="max-chunks-per-source" name="max_chunks_per_source" type="number" min="1" step="1" value="{default_max_chunks_per_source}">
            </label>
          </div>
          <label>Objective
            <textarea id="objective" name="objective" placeholder="What exact question should the report answer?"></textarea>
          </label>
          <label>Audience
            <input id="audience" name="audience" placeholder="ML engineers, researchers, students">
          </label>
          <label>Constraints
            <textarea id="constraints" name="constraints" placeholder="Prefer surveys, benchmarks, and production systems."></textarea>
          </label>
          <label>Comparison Targets
            <input id="comparison-targets" name="comparison_targets" placeholder="Any systems or methods to compare against">
          </label>
          <label class="checkbox">
            <input id="no-clarify" type="checkbox" checked>
            <span>Skip interactive clarification and use the answers above.</span>
          </label>
          <div class="actions">
            <button class="primary button-with-spinner" id="start-button" type="submit">
              <span>Start or Resume Run</span>
              <span class="spinner hidden" id="start-spinner" aria-hidden="true"></span>
            </button>
            <button class="secondary button-with-spinner" id="refresh-button" type="button">
              <span>Refresh Status</span>
              <span class="spinner hidden" id="refresh-spinner" aria-hidden="true"></span>
            </button>
            <button class="danger button-with-spinner" id="stop-button" type="button">
              <span>Stop Active Run</span>
              <span class="spinner hidden" id="stop-spinner" aria-hidden="true"></span>
            </button>
          </div>
        </form>
      </div>

      <div class="panel status-grid">
        <div class="status-bar">
          <div>
            <div class="eyebrow">Current Output</div>
            <div id="current-output" class="small">{_escape_html(default_output)}</div>
          </div>
          <div class="status-chip" id="status-chip">
            <span class="spinner subtle hidden" id="status-spinner" aria-hidden="true"></span>
            <span id="status-chip-label">idle</span>
          </div>
        </div>

        <div class="stats">
          <div class="stat"><span class="k">Selected</span><span class="v" id="count-selected">0</span></div>
          <div class="stat"><span class="k">Citations</span><span class="v" id="count-citations">0</span></div>
          <div class="stat"><span class="k">Notes</span><span class="v" id="count-notes">0</span></div>
          <div class="stat"><span class="k">Findings</span><span class="v" id="count-findings">0</span></div>
        </div>

        <div class="meta-grid">
          <div class="card">
            <h2>Run</h2>
            <div class="small inline"><strong>Stage:</strong> <span id="run-stage">-</span></div><br>
            <div class="small inline"><strong>Provider:</strong> <span id="run-provider">{_escape_html(default_provider)}</span></div><br>
            <div class="small inline"><strong>Model:</strong> <span id="run-model">{default_model_display}</span></div><br>
            <div class="small inline"><strong>PID:</strong> <span id="run-pid">-</span></div><br>
            <div class="small inline"><strong>Exit:</strong> <span id="run-exit">-</span></div><br>
            <div class="small inline"><strong>Budget:</strong> <span id="budget-summary">-</span></div><br>
            <div class="small inline"><strong>LaTeX:</strong> <span id="latex-summary">-</span></div>
          </div>
          <div class="card">
            <h2>Constitution</h2>
            <div class="small inline"><strong>Resume Count:</strong> <span id="resume-count">0</span></div><br>
            <div class="small inline"><strong>Resumed From:</strong> <span id="resume-from">-</span></div><br>
            <div class="small inline"><strong>Checkpoint:</strong> <span id="checkpoint-stage">-</span></div><br>
            <div class="small inline"><strong>Confidence:</strong> <span id="confidence-summary">-</span></div>
          </div>
        </div>

        <div class="card">
          <h2>Progress / Log</h2>
          <div id="progress-summary" class="small">No run data yet.</div>
          <pre class="artifact-view" id="log-view"></pre>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Artifacts</h2>
      <div class="artifact-toolbar">
        <button class="artifact-button" data-artifact="run">run.json</button>
        <button class="artifact-button" data-artifact="retrieval">retrieval.json</button>
        <button class="artifact-button" data-artifact="report">report.tex</button>
        <button class="artifact-button" data-artifact="references">references.bib</button>
        <button class="artifact-button" data-artifact="pdf">report.pdf</button>
        <button class="artifact-button" data-artifact="constitution">constitution.json</button>
        <button class="artifact-button" data-artifact="constitution_bib">constitution.bib</button>
        <button class="artifact-button" data-artifact="log">gui_run.log</button>
      </div>
      <div class="artifact-loading small inline hidden" id="artifact-loading">
        <span class="spinner subtle" aria-hidden="true"></span>
        <span id="artifact-loading-label">Loading artifact...</span>
      </div>
      <pre class="artifact-view" id="artifact-view">Select an artifact to inspect.</pre>
      <object class="artifact-embed" id="artifact-pdf" style="display:none;" type="application/pdf"></object>
      <div class="small" id="artifact-pdf-meta" style="display:none; margin-top:10px;">
        PDF viewer unavailable here.
        <a id="artifact-pdf-link" target="_blank" rel="noopener noreferrer">Open the PDF directly</a>.
      </div>
    </section>
  </div>

  <script>
    const form = document.getElementById('run-form');
    const statusChip = document.getElementById('status-chip');
    const statusChipLabel = document.getElementById('status-chip-label');
    const statusSpinner = document.getElementById('status-spinner');
    const currentOutput = document.getElementById('current-output');
    const artifactView = document.getElementById('artifact-view');
    const artifactPdf = document.getElementById('artifact-pdf');
    const artifactPdfMeta = document.getElementById('artifact-pdf-meta');
    const artifactPdfLink = document.getElementById('artifact-pdf-link');
    const artifactLoading = document.getElementById('artifact-loading');
    const artifactLoadingLabel = document.getElementById('artifact-loading-label');
    const logView = document.getElementById('log-view');
    const providerInput = document.getElementById('provider');
    const providerOptions = document.getElementById('provider-options');
    const modelInput = document.getElementById('model');
    const modelOptions = document.getElementById('model-options');
    const modelHint = document.getElementById('model-hint');
    const apiBaseLabel = document.getElementById('api-base-label');
    const apiBaseInput = document.getElementById('api-base');
    const apiKeyRow = document.getElementById('api-key-row');
    const apiKeyInput = document.getElementById('api-key');
    const serpApiKeyInput = document.getElementById('serpapi-api-key');
    const semanticScholarApiKeyInput = document.getElementById('semantic-scholar-api-key');
    const budgetSlider = document.getElementById('max-summary-model-calls');
    const budgetSliderValue = document.getElementById('budget-slider-value');
    const maxQueriesInput = document.getElementById('max-queries');
    const maxTotalQueriesInput = document.getElementById('max-total-queries');
    const maxSearchRoundsInput = document.getElementById('max-search-rounds');
    const maxSelectedSourcesInput = document.getElementById('max-selected-sources');
    const maxWebResultsInput = document.getElementById('max-web-results-per-query');
    const maxPaperResultsInput = document.getElementById('max-paper-results-per-query');
    const maxCriticResultsInput = document.getElementById('max-critic-results');
    const maxChunksPerSourceInput = document.getElementById('max-chunks-per-source');
    const startButton = document.getElementById('start-button');
    const startSpinner = document.getElementById('start-spinner');
    const refreshButton = document.getElementById('refresh-button');
    const refreshSpinner = document.getElementById('refresh-spinner');
    const refreshModelsButton = document.getElementById('refresh-models-button');
    const refreshModelsSpinner = document.getElementById('refresh-models-spinner');
    const stopButton = document.getElementById('stop-button');
    const stopSpinner = document.getElementById('stop-spinner');
    const artifactButtons = Array.from(document.querySelectorAll('.artifact-button'));
    let currentOutputDir = document.getElementById('output-dir').value;
    let refreshInFlight = false;
    let providerRefreshTimeout = null;

    function setSpinner(spinner, active) {{
      if (spinner) {{
        spinner.classList.toggle('hidden', !active);
      }}
    }}

    function setButtonBusy(button, spinner, active) {{
      if (button) {{
        button.disabled = active;
      }}
      setSpinner(spinner, active);
    }}

    function setStatusBusy(active) {{
      setSpinner(statusSpinner, active);
    }}

    function setArtifactBusy(active, message = 'Loading artifact...') {{
      artifactLoading.classList.toggle('hidden', !active);
      artifactLoadingLabel.textContent = message;
      artifactButtons.forEach((button) => {{
        button.disabled = active;
      }});
    }}

    function populateProviderOptions(providers, selectedProvider) {{
      const cleaned = Array.from(new Map(
        (providers || []).map((item) => {{
          const id = String(item?.id || item || '').trim();
          const label = String(item?.label || id).trim();
          return [id, {{ id, label }}];
        }}).filter(([id]) => Boolean(id))
      ).values());
      providerOptions.innerHTML = '';
      for (const provider of cleaned) {{
        const option = document.createElement('option');
        option.value = provider.id;
        option.label = provider.label;
        providerOptions.appendChild(option);
      }}
      if (selectedProvider) {{
        providerInput.value = selectedProvider;
      }}
    }}

    function populateModelOptions(models, preferredModel) {{
      const cleaned = Array.from(new Set((models || []).map((item) => String(item || '').trim()).filter(Boolean)));
      modelOptions.innerHTML = '';
      for (const model of cleaned) {{
        const option = document.createElement('option');
        option.value = model;
        modelOptions.appendChild(option);
      }}
      const fallback = String(preferredModel || '').trim() || modelInput.value.trim() || (cleaned[0] || '');
      if (fallback) {{
        modelInput.value = fallback;
      }}
    }}

    function applyProviderMetadata(payload) {{
      const provider = payload.provider || providerInput.value.trim() || 'ollama';
      const apiBaseLabelText = payload.api_base_label || 'API Base';
      const apiBasePlaceholder = payload.api_base_placeholder || '';
      const apiKeyLabelText = payload.api_key_label || 'API Key';
      const requiresApiKey = Boolean(payload.requires_api_key);

      apiBaseLabel.childNodes[0].textContent = apiBaseLabelText;
      apiBaseInput.placeholder = apiBasePlaceholder;
      apiKeyRow.childNodes[0].textContent = apiKeyLabelText;
      apiKeyRow.classList.toggle('hidden', !requiresApiKey);
      modelHint.textContent = payload.model_hint || 'Suggestions come from LiteLLM. You can type any supported model manually.';

      if (provider === 'ollama' && !apiBaseInput.value.trim()) {{
        apiBaseInput.value = payload.default_api_base || '{default_api_base_html}';
      }}
      if (provider !== 'ollama' && apiBaseInput.value.trim() === '{default_api_base_html}') {{
        apiBaseInput.value = payload.default_api_base || '';
      }}
    }}

    async function refreshModels(preferredModel = '', preferredProvider = '') {{
      setButtonBusy(refreshModelsButton, refreshModelsSpinner, true);
      try {{
        const provider = (preferredProvider || providerInput.value.trim() || '{default_provider}').toLowerCase();
        const response = await fetch(`/api/models?provider=${{encodeURIComponent(provider)}}`);
        const payload = await response.json();
        populateProviderOptions(payload.providers || [], payload.provider || provider);
        populateModelOptions(payload.models || [], preferredModel || modelInput.value.trim());
        applyProviderMetadata(payload);
        if (payload.default_max_summary_model_calls) {{
          const value = String(payload.default_max_summary_model_calls);
          if (!budgetSlider.value) {{
            budgetSlider.value = value;
          }}
          budgetSliderValue.textContent = budgetSlider.value;
        }}
      }} catch (_error) {{
        populateModelOptions([], preferredModel || modelInput.value.trim());
      }} finally {{
        setButtonBusy(refreshModelsButton, refreshModelsSpinner, false);
      }}
    }}

    function scheduleProviderRefresh() {{
      if (providerRefreshTimeout) {{
        window.clearTimeout(providerRefreshTimeout);
      }}
      providerRefreshTimeout = window.setTimeout(() => {{
        providerRefreshTimeout = null;
        refreshModels('', providerInput.value.trim());
      }}, 180);
    }}

    function answerPayload() {{
      const entries = {{
        objective: document.getElementById('objective').value.trim(),
        audience: document.getElementById('audience').value.trim(),
        constraints: document.getElementById('constraints').value.trim(),
        comparison_targets: document.getElementById('comparison-targets').value.trim(),
      }};
      return Object.fromEntries(Object.entries(entries).filter(([, value]) => value));
    }}

    async function startRun(event) {{
      event.preventDefault();
      currentOutputDir = document.getElementById('output-dir').value.trim();
      currentOutput.textContent = currentOutputDir;
      setButtonBusy(startButton, startSpinner, true);
      setStatusBusy(true);
      try {{
        const response = await fetch('/api/run', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            topic: document.getElementById('topic').value.trim(),
            output_dir: currentOutputDir,
            no_clarify: document.getElementById('no-clarify').checked,
            provider: providerInput.value.trim(),
            model: modelInput.value.trim(),
            api_base: apiBaseInput.value.trim(),
            api_key: apiKeyInput.value,
            serpapi_api_key: serpApiKeyInput.value,
            semantic_scholar_api_key: semanticScholarApiKeyInput.value,
            max_summary_model_calls: Number(budgetSlider.value),
            max_queries: Number(maxQueriesInput.value),
            max_total_queries: Number(maxTotalQueriesInput.value),
            max_search_rounds: Number(maxSearchRoundsInput.value),
            max_selected_sources: Number(maxSelectedSourcesInput.value),
            max_web_results_per_query: Number(maxWebResultsInput.value),
            max_paper_results_per_query: Number(maxPaperResultsInput.value),
            max_critic_results: Number(maxCriticResultsInput.value),
            max_chunks_per_source: Number(maxChunksPerSourceInput.value),
            answers: answerPayload(),
          }}),
        }});
        const payload = await response.json();
        artifactView.textContent = JSON.stringify(payload, null, 2);
        if (!response.ok) {{
          setStatusBusy(false);
          return;
        }}
        await refreshStatus();
      }} finally {{
        setButtonBusy(startButton, startSpinner, false);
      }}
    }}

    async function stopRun() {{
      setButtonBusy(stopButton, stopSpinner, true);
      try {{
        const response = await fetch('/api/stop', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ output_dir: currentOutputDir }}),
        }});
        const payload = await response.json();
        artifactView.textContent = JSON.stringify(payload, null, 2);
        await refreshStatus();
      }} finally {{
        setButtonBusy(stopButton, stopSpinner, false);
      }}
    }}

    async function refreshStatus(options = {{}}) {{
      const manual = Boolean(options.manual);
      if (!currentOutputDir || refreshInFlight) {{
        return;
      }}
      refreshInFlight = true;
      if (manual) {{
        setButtonBusy(refreshButton, refreshSpinner, true);
      }}
      let keepStatusSpinner = false;
      setStatusBusy(true);
      try {{
        const response = await fetch(`/api/status?output_dir=${{encodeURIComponent(currentOutputDir)}}`);
        const payload = await response.json();
        statusChipLabel.textContent = payload.status || 'idle';
        currentOutput.textContent = payload.output_dir || currentOutputDir;
        document.getElementById('run-provider').textContent = payload.provider || providerInput.value.trim() || '-';
        document.getElementById('run-model').textContent = payload.model || modelInput.value.trim() || '-';
        document.getElementById('count-selected').textContent = payload.counts?.selected_sources ?? 0;
        document.getElementById('count-citations').textContent = payload.counts?.citations ?? 0;
        document.getElementById('count-notes').textContent = payload.counts?.source_notes ?? 0;
        document.getElementById('count-findings').textContent = payload.counts?.findings ?? 0;
        document.getElementById('run-stage').textContent = payload.run_stage || '-';
        document.getElementById('run-pid').textContent = payload.process?.pid ?? '-';
        document.getElementById('run-exit').textContent = payload.process?.exit_code ?? '-';
        const latex = payload.latex || {{}};
        const latexMessage = latex.message ? ` (${{latex.message}})` : '';
        const effectiveLatex = latex.status ? `${{latex.status}}${{latexMessage}}` : ((payload.status || '').includes('running') || (payload.status || '') === 'summarizing' ? 'pending until completed' : '-');
        document.getElementById('latex-summary').textContent = effectiveLatex;
        const budget = payload.budget || {{}};
        const before = budget.estimated_summary_calls_before_budget;
        const after = budget.estimated_summary_calls_after_budget;
        document.getElementById('budget-summary').textContent =
          before !== undefined && after !== undefined
            ? `${{after}} / ${{before}} calls (cap ${{budget.max_summary_model_calls ?? '-'}})`
            : (budget.max_summary_model_calls !== undefined ? `cap ${{budget.max_summary_model_calls}} calls` : '-');
        const meta = payload.constitution?.metadata || {{}};
        document.getElementById('resume-count').textContent = meta.resume_count ?? 0;
        document.getElementById('resume-from').textContent = meta.resume_from_status || '-';
        document.getElementById('checkpoint-stage').textContent = meta.last_checkpoint_stage || '-';
        const confidence = payload.constitution?.confidence_summary || {{}};
        const findingSummary = confidence.findings || {{}};
        document.getElementById('confidence-summary').textContent =
          findingSummary.count ? `findings mean ${{findingSummary.mean}}` : 'no findings yet';
        const progress = payload.progress || {{}};
        document.getElementById('progress-summary').textContent =
          Object.keys(progress).length ? JSON.stringify(progress, null, 2) : 'No structured progress yet.';
        logView.textContent = payload.log_excerpt || 'No log output yet.';
        keepStatusSpinner = Boolean(payload.process?.active) || String(payload.status || '').startsWith('running');
      }} catch (error) {{
        statusChipLabel.textContent = 'error';
        artifactView.textContent = JSON.stringify({{ error: String(error) }}, null, 2);
      }} finally {{
        if (manual) {{
          setButtonBusy(refreshButton, refreshSpinner, false);
        }}
        if (!keepStatusSpinner) {{
          setStatusBusy(false);
        }}
        refreshInFlight = false;
      }}
    }}

    async function loadArtifact(name) {{
      setArtifactBusy(true, `Loading ${{name}}...`);
      try {{
        const response = await fetch(`/api/artifact?output_dir=${{encodeURIComponent(currentOutputDir)}}&name=${{encodeURIComponent(name)}}`);
        const payload = await response.json();
        if (name === 'pdf' && payload.viewer_url) {{
          artifactView.style.display = 'none';
          artifactPdf.style.display = 'block';
          artifactPdfMeta.style.display = 'block';
          artifactPdf.data = payload.viewer_url;
          artifactPdfLink.href = payload.viewer_url;
          return;
        }}
        artifactPdf.style.display = 'none';
        artifactPdf.data = '';
        artifactPdfMeta.style.display = 'none';
        artifactPdfLink.href = '#';
        artifactView.style.display = 'block';
        artifactView.textContent = payload.content || JSON.stringify(payload, null, 2);
      }} finally {{
        setArtifactBusy(false);
      }}
    }}

    form.addEventListener('submit', startRun);
    refreshButton.addEventListener('click', () => refreshStatus({{ manual: true }}));
    refreshModelsButton.addEventListener('click', () => refreshModels(modelInput.value.trim(), providerInput.value.trim()));
    providerInput.addEventListener('input', scheduleProviderRefresh);
    providerInput.addEventListener('change', () => refreshModels('', providerInput.value.trim()));
    providerInput.addEventListener('blur', () => refreshModels('', providerInput.value.trim()));
    stopButton.addEventListener('click', stopRun);
    budgetSlider.addEventListener('input', () => {{
      budgetSliderValue.textContent = budgetSlider.value;
    }});
    document.querySelectorAll('.artifact-button').forEach((button) => {{
      button.addEventListener('click', () => loadArtifact(button.dataset.artifact));
    }});

    refreshModels('{default_provider_model}', '{default_provider}');
    refreshStatus();
    setInterval(refreshStatus, 3000);
  </script>
</body>
</html>"""


def start_gui(
    settings: Settings,
    *,
    host: str,
    port: int,
    output_root: Path,
    open_browser: bool = False,
) -> None:
    app = GuiApp(settings, output_root=output_root, launch_cwd=Path.cwd())
    app.serve(host, port, open_browser=open_browser)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[-max_chars:]


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _model_options_html(models: list[str], selected_model: str) -> str:
    options = []
    for model in models:
        if not model:
            continue
        options.append(f'<option value="{_escape_html(model)}"></option>')
    if selected_model and selected_model not in models:
        options.append(f'<option value="{_escape_html(selected_model)}"></option>')
    return "".join(options)


def _provider_options_html(selected_provider: str) -> str:
    options: list[str] = []
    for provider in provider_names():
        label = str(provider_metadata(provider)["label"])
        selected_attr = ' selected' if provider == selected_provider else ''
        options.append(
            f'<option value="{_escape_html(provider)}" label="{_escape_html(label)}"{selected_attr}></option>'
        )
    return "".join(options)
