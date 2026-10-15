"""Resume and abort, offline: an interrupted pass keeps its valid rows and reruns only the rest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_search import eval as harness_eval


class ResumeRows(unittest.TestCase):
    def test_keeps_rows_without_run_error(self):
        summary = {"results": [
            {"id": "t1", "search_backend": "exa_fast", "allow_fetch": False, "passed": True, "error": None},
            {"id": "t1", "search_backend": "jevlin_fast", "allow_fetch": False, "passed": False,
             "error": "BadRequestError: credit balance is too low"},
            {"id": "t2", "search_backend": "exa_fast", "allow_fetch": False, "passed": False, "error": None},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pass-1.json"
            path.write_text(json.dumps(summary))
            rows, keys = harness_eval.load_resume_rows(path)
        self.assertEqual([r["id"] + "/" + r["search_backend"] for r in rows], ["t1/exa_fast", "t2/exa_fast"])
        self.assertEqual(keys, {("t1", "exa_fast", False), ("t2", "exa_fast", False)})

    def test_missing_file_resumes_nothing(self):
        self.assertEqual(harness_eval.load_resume_rows(None), ([], set()))
        self.assertEqual(harness_eval.load_resume_rows(Path("/nonexistent/pass.json")), ([], set()))


if __name__ == "__main__":
    unittest.main()
