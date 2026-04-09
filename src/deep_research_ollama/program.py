from __future__ import annotations

from pathlib import Path

from deep_research_ollama.config import Settings


DEFAULT_RESEARCH_PROGRAM = """# Research Program

You are running a deep-research workflow.

Operating rules:
- Ask concise clarifying questions before research starts.
- Rewrite the request into a precise research brief before searching.
- Expand into adjacent topics only when they improve coverage of the main question.
- Prefer primary sources, strong surveys, standards, official documentation, and peer-reviewed work.
- When a claim is weakly supported, mark it as uncertain instead of presenting it as fact.
- Every report paragraph must cite valid citation keys from the current run.
- Prefer exact DOI-backed BibTeX when available.
- Keep the research constitution compact, additive, and editable.
- Track evidence gaps, contradictions, and unresolved questions in notes.
"""


def load_research_program(output_dir: Path, settings: Settings) -> str:
    path = output_dir / settings.program_filename
    if not path.exists():
        path.write_text(DEFAULT_RESEARCH_PROGRAM + "\n", encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()
