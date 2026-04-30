import contextlib
import io
import json
import subprocess
import sys

import main


def test_main_prints_smoke_run_record() -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        main.main()

    record = json.loads(stdout.getvalue())

    assert record["query_id"] == "smoke-q1"
    assert record["status"] == "completed"
    assert record["result"] == [{"type": "output_text", "output": "smoke answer"}]
    assert record["tool_call_counts"] == {"search": 1, "get_document": 0}


def test_main_script_runs_from_clean_process() -> None:
    completed = subprocess.run(
        [sys.executable, "main.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    record = json.loads(completed.stdout)

    assert record["query_id"] == "smoke-q1"
    assert record["status"] == "completed"
