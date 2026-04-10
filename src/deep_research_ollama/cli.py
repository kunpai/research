from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from deep_research_ollama.config import Settings
from deep_research_ollama.constitution import ConstitutionStore
from deep_research_ollama.gui import start_gui
from deep_research_ollama.pipeline import ResearchPipeline
from deep_research_ollama.program import DEFAULT_RESEARCH_PROGRAM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Deep research with LiteLLM, search tools, BibTeX, and LaTeX output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an interactive deep-research session.")
    run_parser.add_argument("topic", help="Research topic or question.")
    run_parser.add_argument(
        "--output-dir",
        default="./output/latest",
        help="Directory for constitution, BibTeX, and LaTeX artifacts.",
    )
    run_parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip interactive clarifying questions.",
    )
    run_parser.add_argument(
        "--answer",
        action="append",
        default=[],
        help="Pre-supply a clarifying answer as key=value. Can be repeated.",
    )
    run_parser.add_argument(
        "--provider",
        default="",
        help="Override the LiteLLM provider for this run, for example ollama, gemini, openai, or anthropic.",
    )
    run_parser.add_argument(
        "--model",
        default="",
        help="Override the model for this run.",
    )
    run_parser.add_argument(
        "--api-base",
        default="",
        help="Override the provider API base URL for this run.",
    )
    run_parser.add_argument(
        "--max-summary-model-calls",
        type=int,
        default=None,
        help="Override the summary-call budget for this run.",
    )
    _add_knob_arguments(run_parser)

    show_parser = subparsers.add_parser(
        "show-constitution", help="Show the current constitution snapshot."
    )
    show_parser.add_argument(
        "--output-dir",
        default="./output/latest",
        help="Directory containing constitution artifacts.",
    )

    delete_citation_parser = subparsers.add_parser(
        "delete-citation", help="Delete a citation from the constitution."
    )
    delete_citation_parser.add_argument("cite_key", help="Citation key to remove.")
    delete_citation_parser.add_argument(
        "--output-dir",
        default="./output/latest",
        help="Directory containing constitution artifacts.",
    )

    delete_finding_parser = subparsers.add_parser(
        "delete-finding", help="Delete a finding from the constitution."
    )
    delete_finding_parser.add_argument("finding_id", help="Finding id to remove.")
    delete_finding_parser.add_argument(
        "--output-dir",
        default="./output/latest",
        help="Directory containing constitution artifacts.",
    )

    init_program_parser = subparsers.add_parser(
        "init-program", help="Write the default editable research program file."
    )
    init_program_parser.add_argument(
        "--output-dir",
        default="./output/latest",
        help="Directory where the research program file should be written.",
    )

    gui_parser = subparsers.add_parser("gui", help="Launch the local research GUI.")
    gui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind the local GUI server to.",
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local GUI server.",
    )
    gui_parser.add_argument(
        "--output-root",
        default="./output",
        help="Root directory that the GUI should use for run folders.",
    )
    gui_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the GUI in the default browser after the server starts.",
    )
    gui_parser.add_argument(
        "--provider",
        default="",
        help="Default LiteLLM provider to prefill in the GUI.",
    )
    gui_parser.add_argument(
        "--model",
        default="",
        help="Default model to prefill in the GUI.",
    )
    gui_parser.add_argument(
        "--api-base",
        default="",
        help="Default provider API base URL to prefill in the GUI.",
    )
    gui_parser.add_argument(
        "--max-summary-model-calls",
        type=int,
        default=None,
        help="Default summary-call budget to prefill in the GUI.",
    )
    _add_knob_arguments(gui_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    if getattr(args, "provider", "").strip():
        settings = replace(settings, llm_provider=args.provider.strip().lower())
    if getattr(args, "model", "").strip():
        settings = replace(settings, llm_model=args.model.strip())
    if getattr(args, "api_base", "").strip():
        settings = replace(settings, llm_api_base=args.api_base.strip())
    if getattr(args, "max_summary_model_calls", None) is not None:
        settings = replace(settings, max_summary_model_calls=int(args.max_summary_model_calls))
    settings = _apply_knob_overrides(settings, args)

    if args.command == "gui":
        start_gui(
            settings,
            host=args.host,
            port=args.port,
            output_root=Path(args.output_root).expanduser().resolve(),
            open_browser=args.open_browser,
        )
        return

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.command == "run":
        pipeline = ResearchPipeline(settings, output_dir)
        result = pipeline.run(
            args.topic,
            interactive=not args.no_clarify,
            initial_answers=_parse_answers(args.answer),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "init-program":
        output_dir.mkdir(parents=True, exist_ok=True)
        program_path = output_dir / settings.program_filename
        if not program_path.exists():
            program_path.write_text(DEFAULT_RESEARCH_PROGRAM + "\n", encoding="utf-8")
        print(f"Wrote research program: {program_path}")
        return

    store = ConstitutionStore(
        output_dir / settings.constitution_filename,
        output_dir / settings.constitution_bib_filename,
    )
    data = store.load(topic="")

    if args.command == "show-constitution":
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if args.command == "delete-citation":
        store.delete_citations([args.cite_key])
        store.save()
        print(f"Deleted citation: {args.cite_key}")
        return

    if args.command == "delete-finding":
        store.delete_findings([args.finding_id])
        store.save()
        print(f"Deleted finding: {args.finding_id}")
        return


def _parse_answers(entries: list[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep:
            raise SystemExit(f"Invalid --answer value '{entry}'. Expected key=value.")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"Invalid --answer value '{entry}'. Expected key=value.")
        answers[key] = value
    return answers


def _add_knob_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum seed queries generated before expansion.",
    )
    parser.add_argument(
        "--max-total-queries",
        type=int,
        default=None,
        help="Maximum total queries across all retrieval rounds.",
    )
    parser.add_argument(
        "--max-search-rounds",
        type=int,
        default=None,
        help="Maximum retrieval expansion rounds.",
    )
    parser.add_argument(
        "--max-web-results-per-query",
        type=int,
        default=None,
        help="Maximum web results fetched per query.",
    )
    parser.add_argument(
        "--max-paper-results-per-query",
        type=int,
        default=None,
        help="Maximum paper results fetched per query.",
    )
    parser.add_argument(
        "--max-selected-sources",
        type=int,
        default=None,
        help="Maximum sources selected for reading and synthesis.",
    )
    parser.add_argument(
        "--max-critic-results",
        type=int,
        default=None,
        help="Maximum shortlist size passed to the critic. Use 0 to judge the full shortlist.",
    )
    parser.add_argument(
        "--max-chunks-per-source",
        type=int,
        default=None,
        help="Maximum chunks read from each source.",
    )


def _apply_knob_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    replacements: dict[str, int] = {}
    for name in (
        "max_queries",
        "max_total_queries",
        "max_search_rounds",
        "max_web_results_per_query",
        "max_paper_results_per_query",
        "max_selected_sources",
        "max_critic_results",
        "max_chunks_per_source",
    ):
        value = getattr(args, name, None)
        if value is not None:
            replacements[name] = int(value)
    if not replacements:
        return settings
    return replace(settings, **replacements)
