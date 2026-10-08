import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compare_transcripts import load_terms
from finalize_comparison import excerpt, verify_analysis_sources, verify_review
from transcription_settings import load_settings, model_directory


class LocalConfigurationTests(unittest.TestCase):
    def test_existing_paths_are_loaded_and_cli_model_path_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"container": "local-test", "qwen_model_dir": "/cache/local-qwen"}))
            with patch.dict(os.environ, {"TRANSCRIPTION_CONFIG": str(path)}):
                self.assertEqual(load_settings()["container"], "local-test")
                self.assertEqual(model_directory("qwen"), "/cache/local-qwen")
                self.assertEqual(model_directory("qwen", "/cache/explicit"), "/cache/explicit")

    def test_missing_explicit_config_fails_instead_of_targeting_default_container(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_settings(Path(directory) / "missing.json")

    def test_relative_container_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"container_root": "relative/project"}))
            with self.assertRaises(ValueError):
                load_settings(path)


class SuppliedReviewTests(unittest.TestCase):
    def setUp(self):
        self.analysis = {"summary": {"duration_seconds": 120}, "windows": [
            {"start": 0, "end": 60, "whisper": "첫 번째 테스트입니다.", "qwen": "첫번째 테스트입니다!", "flags": []},
            {"start": 60, "end": 120, "whisper": "second", "qwen": "second", "flags": []}]}
        self.review = {"findings": [{"id": "sample", "start": 0, "end": 60,
            "whisper_phrases": ["첫 번째 테스트입니다"], "qwen_phrases": ["첫번째 테스트입니다"],
            "observation": "Spacing differs.", "interpretation": "No correctness verdict."}],
            "reviewed_window_starts": [0, 60], "review": {"completed": True}}

    def test_external_findings_retain_original_spacing_and_punctuation(self):
        findings, reviewed, notes, completed = verify_review(self.analysis, self.review, {})
        self.assertEqual(findings[0]["whisper_excerpts"], ["첫 번째 테스트입니다"])
        self.assertEqual(reviewed, {0, 60})
        self.assertTrue(completed)
        self.assertIn("Spacing differs.", notes[0][0])
        self.assertIn("whisper_phrases", self.review["findings"][0])

    def test_missing_fragment_blocks_review(self):
        self.review["findings"][0]["qwen_phrases"] = ["absent phrase"]
        with self.assertRaises(ValueError):
            verify_review(self.analysis, self.review, {})

    def test_modified_analysis_cannot_certify_source_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for backend in ("whisper", "qwen"):
                path = Path(directory) / (backend + ".json")
                path.write_text(json.dumps({"segments": [
                    {"start": 0, "end": 20, "text": self.analysis["windows"][0][backend]},
                    {"start": 60, "end": 80, "text": "second"}]}))
                sources[backend] = path
            verify_analysis_sources(self.analysis, sources)
            self.analysis["windows"][0]["qwen"] = "fabricated text"
            with self.assertRaises(ValueError):
                verify_analysis_sources(self.analysis, sources)

    def test_partial_review_is_not_reported_complete(self):
        self.review["reviewed_window_starts"] = [0]
        self.assertFalse(verify_review(self.analysis, self.review, {})[-1])

    def test_source_hash_mismatch_blocks_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("{}")
            self.review["expected_sha256"] = {"whisper": "0" * 64}
            with self.assertRaises(ValueError):
                verify_review(self.analysis, self.review, {"whisper": path})

    def test_unknown_window_and_empty_fragment_are_rejected(self):
        self.review["reviewed_window_starts"] = [999]
        with self.assertRaises(ValueError):
            verify_review(self.analysis, self.review, {})
        with self.assertRaises(ValueError):
            excerpt("text", "...")

    def test_domain_terms_are_optional_and_external(self):
        self.assertEqual(load_terms(), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.json"
            path.write_text(json.dumps(["custom", "custom", "other"]))
            self.assertEqual(load_terms(path), ["custom", "other"])
            path.write_text('{"terms": "wrong shape"}')
            with self.assertRaises(ValueError):
                load_terms(path)


if __name__ == "__main__":
    unittest.main()
