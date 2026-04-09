from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deep_research_ollama.cli import build_parser, main
from deep_research_ollama.config import Settings
from deep_research_ollama.gui import (
    GuiApp,
    _list_models_from_manifests,
    _model_options_html,
    _parse_ollama_list_output,
    build_run_command,
    collect_run_status,
)


class _RunningProcess:
    pid = 4321

    @staticmethod
    def poll() -> None:
        return None


class GuiTests(unittest.TestCase):
    def test_parse_ollama_list_output_reads_model_names(self) -> None:
        output = (
            "NAME                ID              SIZE      MODIFIED\n"
            "gemma4:e4b          abc123          5.4 GB    2 hours ago\n"
            "qwen3:14b           def456          9.1 GB    3 days ago\n"
        )

        models = _parse_ollama_list_output(output)

        self.assertEqual(models, ["gemma4:e4b", "qwen3:14b"])

    def test_list_models_from_manifests_falls_back_to_installed_models(self) -> None:
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "registry.ollama.ai" / "library" / "gemma4" / "e4b").parent.mkdir(
                parents=True, exist_ok=True
            )
            (root / "registry.ollama.ai" / "library" / "gemma4" / "e4b").write_text(
                "{}", encoding="utf-8"
            )
            (root / "registry.ollama.ai" / "openai" / "gpt-oss" / "20b").parent.mkdir(
                parents=True, exist_ok=True
            )
            (root / "registry.ollama.ai" / "openai" / "gpt-oss" / "20b").write_text(
                "{}", encoding="utf-8"
            )

            models = _list_models_from_manifests(root)

            self.assertEqual(models, ["gemma4:e4b", "openai/gpt-oss:20b"])

    def test_model_options_html_does_not_inject_missing_default(self) -> None:
        html = _model_options_html(["gemma4:e4b", "qwen3:14b"], "llama3.1")

        self.assertNotIn("llama3.1", html)
        self.assertIn("gemma4:e4b", html)
        self.assertIn("qwen3:14b", html)

    def test_build_run_command_includes_answers_and_no_clarify(self) -> None:
        command = build_run_command(
            topic="AI for mathematics",
            output_dir=Path("/tmp/gui-run"),
            answers={
                "objective": "compare theorem proving systems",
                "audience": "researchers",
            },
            model="gemma4:e4b",
            max_summary_model_calls=12,
            no_clarify=True,
        )

        self.assertEqual(command[:4], [sys.executable, "-m", "deep_research_ollama", "run"])
        self.assertIn("AI for mathematics", command)
        self.assertIn("--output-dir", command)
        self.assertIn("/tmp/gui-run", command)
        self.assertIn("--model", command)
        self.assertIn("gemma4:e4b", command)
        self.assertIn("--max-summary-model-calls", command)
        self.assertIn("12", command)
        self.assertIn("--no-clarify", command)
        self.assertIn("--answer", command)
        self.assertIn("objective=compare theorem proving systems", command)
        self.assertIn("audience=researchers", command)

    def test_collect_run_status_reads_checkpoint_and_confidence(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir)
            settings = Settings()

            run_payload = {
                "topic": "AI for math",
                "model": "qwen3:14b",
                "status": "summarizing",
                "progress": {"summarized_sources": 2, "selected_sources": 4},
                "retrieval": {
                    "budget": {
                        "estimated_summary_calls_before_budget": 9,
                        "estimated_summary_calls_after_budget": 4,
                    }
                },
                "selected_sources": [{"result_id": "paper-1"}],
                "citations": [{"cite_key": "smith2024"}],
                "source_notes": [{"source_id": "paper-1"}],
            }
            constitution_payload = {
                "topic": "AI for math",
                "metadata": {
                    "last_checkpoint_stage": "summarizing",
                    "resume_count": 1,
                    "resume_from_status": "retrieving",
                    "confidence_summary": {
                        "findings": {"count": 2, "mean": 0.58, "label": "medium"}
                    },
                },
                "findings": [{"finding_id": "finding-1"}],
            }

            (output_dir / settings.run_filename).write_text(
                json.dumps(run_payload, indent=2),
                encoding="utf-8",
            )
            (output_dir / settings.constitution_filename).write_text(
                json.dumps(constitution_payload, indent=2),
                encoding="utf-8",
            )
            (output_dir / "gui_run.log").write_text("reader: summarizing source 2/4\n", encoding="utf-8")

            status = collect_run_status(
                output_dir=output_dir,
                settings=settings,
                process=_RunningProcess(),
            )

            self.assertEqual(status["status"], "running:summarizing")
            self.assertEqual(status["topic"], "AI for math")
            self.assertEqual(status["model"], "qwen3:14b")
            self.assertEqual(status["counts"]["selected_sources"], 1)
            self.assertEqual(status["counts"]["citations"], 1)
            self.assertEqual(status["counts"]["source_notes"], 1)
            self.assertEqual(status["counts"]["findings"], 1)
            self.assertEqual(status["budget"]["estimated_summary_calls_after_budget"], 4)
            self.assertEqual(status["constitution"]["metadata"]["resume_count"], 1)
            self.assertEqual(
                status["constitution"]["confidence_summary"]["findings"]["mean"],
                0.58,
            )
            self.assertIn("summarizing source", status["log_excerpt"])

    def test_collect_run_status_uses_process_model_before_checkpoint_exists(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir)
            settings = Settings()

            status = collect_run_status(
                output_dir=output_dir,
                settings=settings,
                process=_RunningProcess(),
                process_model="gemma4:e4b",
                process_budget=11,
            )

            self.assertEqual(status["status"], "running:starting")
            self.assertEqual(status["model"], "gemma4:e4b")
            self.assertEqual(status["budget"]["max_summary_model_calls"], 11)

    def test_load_artifact_returns_pdf_viewer_url(self) -> None:
        with TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir)
            pdf_path = output_dir / "report.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\n")
            app = GuiApp(Settings(), output_root=output_dir, launch_cwd=output_dir)

            payload = app._load_artifact(output_dir, "pdf")

            self.assertTrue(payload["exists"])
            self.assertEqual(payload["content_type"], "application/pdf")
            self.assertIn("/artifact-file?output_dir=", payload["viewer_url"])
            self.assertIn("name=pdf", payload["viewer_url"])

    def test_render_index_uses_select_for_models(self) -> None:
        app = GuiApp(Settings(ollama_model="gemma4:e4b"), output_root=Path("/tmp"), launch_cwd=Path("/tmp"))

        html = app.render_index()

        self.assertIn('<select id="model"', html)
        self.assertIn("Refresh Models", html)
        self.assertIn('id="max-summary-model-calls"', html)
        self.assertIn('id="start-spinner"', html)
        self.assertIn('id="status-spinner"', html)
        self.assertIn('id="artifact-loading"', html)

    def test_cli_parser_accepts_gui_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "gui",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--output-root",
                "/tmp/deep-research-gui",
                "--model",
                "gemma4:e4b",
                "--max-summary-model-calls",
                "24",
                "--open-browser",
            ]
        )

        self.assertEqual(args.command, "gui")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.output_root, "/tmp/deep-research-gui")
        self.assertEqual(args.model, "gemma4:e4b")
        self.assertEqual(args.max_summary_model_calls, 24)
        self.assertTrue(args.open_browser)

    def test_main_dispatches_gui_command(self) -> None:
        with patch("deep_research_ollama.cli.start_gui") as mock_start_gui:
            with patch.object(
                sys,
                "argv",
                [
                    "deep-research",
                    "gui",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                    "--output-root",
                    "/tmp/gui-output",
                    "--model",
                    "gemma4:e4b",
                    "--max-summary-model-calls",
                    "24",
                ],
            ):
                main()

        mock_start_gui.assert_called_once()
        _, kwargs = mock_start_gui.call_args
        settings = mock_start_gui.call_args[0][0]
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 8765)
        self.assertEqual(kwargs["output_root"], Path("/tmp/gui-output").resolve())
        self.assertFalse(kwargs["open_browser"])
        self.assertEqual(settings.ollama_model, "gemma4:e4b")
        self.assertEqual(settings.max_summary_model_calls, 24)

    def test_main_run_applies_model_override(self) -> None:
        with patch("deep_research_ollama.cli.ResearchPipeline") as mock_pipeline_cls:
            instance = mock_pipeline_cls.return_value
            instance.run.return_value = {}
            with (
                patch("builtins.print"),
                patch.object(
                    sys,
                    "argv",
                    [
                        "deep-research",
                        "run",
                        "AI for math",
                        "--output-dir",
                        "/tmp/gui-run",
                        "--no-clarify",
                        "--model",
                        "qwen3:14b",
                        "--max-summary-model-calls",
                        "9",
                    ],
                ),
            ):
                main()

        settings = mock_pipeline_cls.call_args[0][0]
        self.assertEqual(settings.ollama_model, "qwen3:14b")
        self.assertEqual(settings.max_summary_model_calls, 9)


if __name__ == "__main__":
    unittest.main()
