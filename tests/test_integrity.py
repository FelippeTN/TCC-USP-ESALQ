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

    def execute(self, content, query=None):
        class StubClient:
            def chat(self, *args, **kwargs):
                return {"message": {"content": content}, "usage": {}, "latency_s": 0.0}

        query = query or next(q for q in QUERIES if q["type"] == "no_tool")
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
            path = self.folder / f"mock_results_v{runner.PROTOCOL_VERSION}.csv"
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

    def exposed_order(self, query, model, rep, retrieval="full"):
        seen = {}

        class CaptureClient:
            def chat(self, model_key, messages, tools=None, **kwargs):
                seen["names"] = [t["function"]["name"] for t in tools]
                return {"message": {"content": "ok", "tool_calls": None}, "usage": {}, "latency_s": 0.0}

        condition = dict(runner.BASELINE, model=model, axis="recuperacao", retrieval=retrieval)
        row = runner.execute_one(CaptureClient(), None, condition, query, rep, backend="mock")
        self.assertEqual(row["error"], "")
        return seen["names"], row

    def test_exposed_order_varies_by_repetition_not_by_model(self):
        self.assertIn("expected_position", runner.FIELDS)
        query = next(q for q in QUERIES if q["type"] == "direct")
        rep1, row1 = self.exposed_order(query, "deepseek-v4-flash", 1)
        rep2, _ = self.exposed_order(query, "deepseek-v4-flash", 2)
        other, row_other = self.exposed_order(query, "gemma-4-e4b", 1)
        self.assertEqual(sorted(rep1), sorted(rep2))
        self.assertNotEqual(rep1, rep2)
        self.assertEqual(rep1, other)
        self.assertEqual(row1["expected_position"], rep1.index(query["expected"][0]))
        self.assertEqual(row_other["expected_position"], row1["expected_position"])
        positions = {self.exposed_order(query, "deepseek-v4-flash", r)[1]["expected_position"] for r in range(1, 6)}
        self.assertGreater(len(positions), 1)
        missed = next(row for q in QUERIES if q["expected"]
                      for row in [self.exposed_order(q, "gemma-4-e4b", 1, retrieval="random")[1]]
                      if not row["retrieval_hit"])
        self.assertEqual(missed["expected_position"], "")
        self.assertEqual(self.execute("pass")["expected_position"], "")

    def test_near_miss_rewards_abstention_and_flags_the_lure(self):
        near = [q for q in QUERIES if q["type"] == "near_miss"]
        self.assertEqual(len(near), 16)
        self.assertTrue(all(not q["expected"] and q["lure"] for q in near))
        query = near[0]
        row = self.execute("pass", query)
        self.assertEqual((row["correct"], row["hallucinated"], row["lure_hit"], row["parse_error"]), (1, 0, 0, 0))
        row = self.execute(f"{query['lure']}()", query)
        self.assertEqual((row["correct"], row["hallucinated"], row["lure_hit"], row["invented_tool"]), (0, 1, 1, 0))

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
