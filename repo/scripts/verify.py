#!/usr/bin/env python3
"""Run meaningful scientific contract tests and save results under data/."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "data/results/verification"
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["MPLCONFIGDIR"] = str(ROOT / "data/cache/matplotlib")
commands = {
    "contracts": [sys.executable, "-m", "pytest", "-q", str(ROOT / "repo/tests"),
                  "--junitxml=" + str(out / "pytest.xml")],
    "dependencies": [sys.executable, "-m", "pip", "check"],
}
results = {}
for name, command in commands.items():
    run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    (out / (name + ".log")).write_text(run.stdout + run.stderr)
    results[name] = {"command": command, "returncode": run.returncode}
    print(run.stdout)
    if run.stderr:
        print(run.stderr, file=sys.stderr)
results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
results["source_sha256"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT / "repo").rglob("*.py") if "__pycache__" not in str(p)
}
(out / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
sys.exit(any(results[name]["returncode"] for name in commands))
