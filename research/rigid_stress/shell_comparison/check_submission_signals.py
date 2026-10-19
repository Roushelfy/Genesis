"""Check durable exit receipts and destination preservation without scientific work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from . import submit


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    output = submit.runtime_output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, action, expected in (
        ("success", "true", 0),
        ("failure", "exit 7", 7),
        ("term", 'kill -TERM "$$"', 143),
        ("interrupt", 'kill -INT "$$"', 130),
        ("hangup", 'kill -HUP "$$"', 129),
    ):
        run, scratch, script = output / name, output / (name + "-scratch"), output / (name + ".sh")
        script.write_text(
            "\n".join(submit.workload_preamble(run))
            + "\ntask_scratch="
            + shlex.quote(str(scratch))
            + '\nmkdir -p "$task_scratch"\n'
            + action
            + "\n"
        )
        result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=5)
        recorded = int((run / "returncode.txt").read_text())
        assert result.returncode == recorded == expected and not scratch.exists()
        rows.append({"case": name, "process_exit": result.returncode, "recorded_exit": recorded})
    run, script = output / "external-term", output / "external-term.sh"
    # Readiness comes from the actual foreground child. Signalling after a
    # parent echo can race with the child's fork and leave it sleeping.
    child = shlex.join([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"])
    script.write_text("\n".join(submit.workload_preamble(run)) + "\n" + child + "\n")
    process = subprocess.Popen(
        ["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143 and (run / "returncode.txt").read_text().strip() == "143"
        rows.append({"case": "external_process_group_TERM", "process_exit": 143, "recorded_exit": 143})
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    old = (output / "success/returncode.txt").read_bytes()
    result = subprocess.run(["bash", str(output / "success.sh")], capture_output=True, timeout=5)
    assert result.returncode == 1 and (output / "success/returncode.txt").read_bytes() == old
    rows.append({"case": "existing_destination", "process_exit": 1, "original_receipt_unchanged": True})
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cases": rows,
                "source_sha256": hashlib.sha256(Path(submit.__file__).read_bytes()).hexdigest(),
                "scope": "Stdlib/shell orchestration controls only; no scientific workload executed.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"passed": True, "cases": len(rows)}))


if __name__ == "__main__":
    main()
