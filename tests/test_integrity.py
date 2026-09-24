"""Regressões de parsing, integridade e retomada; apenas dados temporários e mock."""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

os.environ.setdefault("TCC_SERVER_HOST", "example.invalid")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyze
import retrieval
import run_experiment as runner
from invocation import parse_response
from queries import QUERIES


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tcc-tests-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def execute(self, content):
        class StubClient:
            def chat(self, *args, **kwargs):
                return {"message": {"content": content}, "usage": {}, "latency_s": 0.0}

        query = next(q for q in QUERIES if q["type"] == "no_tool")
        condition = dict(runner.BASELINE, model="gemma-4-e4b", axis="invocacao", invocation="code_action")
        return runner.execute_one(StubClient(), None, condition, query, 1, backend="mock")

    def write_csv(self, rows, fields=runner.FIELDS):
        path = self.folder / "sample.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_invalid_code_is_a_parsing_failure(self):
        for content in ("# comentário", "pass; send_message()", "send_message()\npass", "42", "send_message("):
            with self.subTest(content=content):
                self.assertEqual(parse_response("code_action", {"content": content}), (None, True))
                row = self.execute(content)
                self.assertEqual(row["error"], "")
                self.assertEqual(row["parse_error"], 1)
                self.assertEqual(row["correct"], 1)  # definição histórica permanece explícita
                self.assertEqual(row["correct_valid_parse"], 0)
                self.assertEqual(json.loads(row["response_message"])["content"], content)

    def test_valid_abstention(self):
        row = self.execute("pass")
        self.assertEqual((row["correct"], row["correct_valid_parse"], row["parse_error"]), (1, 1, 0))

    def test_malformed_native_calls(self):
        for calls in ([{}], [{"function": {"name": ""}}], [{"function": {"name": []}}], [None], [{}, {}]):
            with self.subTest(calls=calls):
                self.assertEqual(parse_response("native", {"tool_calls": calls}), (None, True))

    def test_legacy_csv_is_readable_but_cannot_be_resumed(self):
        path = self.write_csv([self.execute("pass")], runner.LEGACY_FIELDS)
        self.assertEqual(analyze.load(path).correct_valid_parse.iloc[0], 1)
        with self.assertRaisesRegex(ValueError, "legado"):
            runner.load_done(path, "real")

    def test_historical_path_is_protected_for_both_backends(self):
        for backend in ("mock", "real"):
            with self.assertRaisesRegex(ValueError, "histórica"):
                runner.load_done(runner.RAW_CSV, backend)

    def test_cli_rejects_historical_output_before_contacting_models(self):
        argv = ["run_experiment.py", "--backend", "real", "--out", str(runner.RAW_CSV)]
        with patch("sys.argv", argv), patch.object(runner, "get_client") as client, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                runner.main()
            self.assertEqual(raised.exception.code, 2)
            client.assert_not_called()

    def test_duplicate_rows_are_rejected(self):
        row = self.execute("pass")
        path = self.write_csv([row, row])
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            analyze.load(path)
        with self.assertRaisesRegex(ValueError, "duplicada"):
            runner.load_done(path, "mock")

    def test_duplicate_conditions_with_different_ids_are_rejected(self):
        row = self.execute("pass")
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            analyze.load(self.write_csv([row, dict(row, run_id="outro-id")]))

    def test_mixed_backend_and_protocol_are_rejected(self):
        row = self.execute("pass")
        for field, value in (("backend", "real"), ("protocol_version", 1)):
            path = self.write_csv([row, dict(row, **{field: value})])
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze.load(path)
            with self.assertRaises(ValueError):
                runner.load_done(path, "mock")

    def test_invalid_measurements_are_rejected(self):
        row = self.execute("pass")
        for field, value in (("correct", 2), ("latency_s", -1), ("prompt_tokens", "inválido"),
                             ("latency_s", float("inf")), ("parse_error", ""),
                             ("repetition", 0), ("correct_valid_parse", 0)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                analyze.load(self.write_csv([dict(row, **{field: value})]))

    def test_failed_attempt_and_successful_retry(self):
        row = self.execute("pass")
        path = self.write_csv([dict(row, error="TimeoutError", correct=""), row])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(analyze.load(path)), 1)
        self.assertEqual(runner.load_done(path, "mock"), {row["run_id"]})

    def test_manifest_blocks_unattributed_or_changed_runs(self):
        path = self.folder / "new.csv"
        runner.record_metadata(path, "mock")
        runner.record_metadata(path, "mock")
        with self.assertRaisesRegex(ValueError, "mudou"):
            runner.record_metadata(path, "real")
        orphan = self.write_csv([self.execute("pass")])
        with self.assertRaisesRegex(ValueError, "sem manifesto"):
            runner.record_metadata(orphan, "mock")

    def test_mock_cli_defaults_and_resume_do_not_duplicate(self):
        argv = ["run_experiment.py", "--limit", "2", "--repetitions", "1"]
        with patch("sys.argv", argv), patch.object(runner, "RESULTS_DIR", self.folder), \
                patch.object(retrieval, "CACHE_DIR", self.folder), contextlib.redirect_stdout(io.StringIO()):
            runner.main()
            path = self.folder / "mock_results_v2.csv"
            before = path.read_bytes()
            runner.main()
            self.assertEqual(before, path.read_bytes())
        df = analyze.load(path)
        self.assertEqual(len(df), 36)
        self.assertEqual(set(df.backend), {"mock"})
        tests = analyze.ofat_tests(df)
        pd.testing.assert_frame_equal(tests, analyze.ofat_tests(df))
        incomplete = df.drop(df[(df.toolset_size == 10) & (df.model == "gemma-4-e4b")].index[:1])
        with self.assertRaisesRegex(ValueError, "não pareadas"):
            analyze.ofat_tests(incomplete)
        unequal_repetitions = df.copy()
        unequal_repetitions.loc[unequal_repetitions.toolset_size == 10, "repetition"] = 2
        with self.assertRaisesRegex(ValueError, "Repetições não pareadas"):
            analyze.ofat_tests(unequal_repetitions)

    def test_permutation_and_holm_sanity(self):
        self.assertEqual(analyze.holm([0.001, 0.04, 0.9]), [True, False, False])
        self.assertEqual(analyze._paired_permutation(np.zeros(20)), 1.0)
        self.assertLess(analyze._paired_permutation(np.ones(20), np.random.default_rng(5)), .01)

    def test_sensitivity_cli_preserves_primary_summary(self):
        baseline = self.execute("pass")
        baseline["invocation"] = "native"
        baseline["run_id"] = baseline["run_id"].replace("code_action", "native")
        path = self.write_csv([baseline, self.execute("# comentário")])
        argv = ["analyze.py", "--input", str(path)]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            analyze.main()
        primary = self.folder / "summary_by_condition_sample.csv"
        before = primary.read_bytes()
        with patch("sys.argv", argv + ["--metric", "correct_valid_parse"]), contextlib.redirect_stdout(io.StringIO()):
            analyze.main()
        self.assertEqual(before, primary.read_bytes())
        sensitivity = pd.read_csv(self.folder / "summary_by_condition_sample_correct_valid_parse.csv")
        self.assertEqual(sensitivity.loc[sensitivity.invocation == "code_action", "acuracia"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
