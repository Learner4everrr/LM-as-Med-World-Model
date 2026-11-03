#!/usr/bin/env python3
"""Verify the public artifacts and independently reproduce the primary contrast.

Reads only simulator records from the release ZIP. No model execution,
clinical download, or extraction of archived prompts is required.
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(archive, name):
    with archive.open(name) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    manifest = json.loads((ROOT / "data/releases/public_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    inputs = [
        json.loads(line)
        for folder in ["benchmark_v1", "hypoglycemia_stress_v1"]
        for line in (ROOT / "data/simulated" / folder / "test.inputs.jsonl").read_text().splitlines()
    ]
    allowed = {r["id"] for r in inputs}
    assert all(r["domain"] in {"simglucose", "anesthesia"} for r in inputs)
    checked = []
    with zipfile.ZipFile(ROOT / "data/releases/synthetic_outputs.zip") as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            parts = Path(name).parts
            assert not Path(name).is_absolute() and ".." not in parts, name
            assert parts[0] in {"study_v1", "stress_v1", "interface_memory_v1"}, name
            if not name.endswith("/predictions.jsonl"):
                continue
            predictions = rows(archive, name)
            assert all(r["domain"] in {"simglucose", "anesthesia"} and r["id"] in allowed
                       for r in predictions), name
            parent = name.rsplit("/", 1)[0]
            hashes = {p["prompt_hash"] for r in predictions for p in r["predictions"]
                      if "prompt_hash" in p}
            ids = {r["id"] for r in predictions}
            for filename in ["auxiliary.json", "diagnostics.json", "memory.json"]:
                extra = parent + "/" + filename
                if extra in archive.namelist():
                    for record in json.loads(archive.read(extra)):
                        assert record["id"] in ids, extra
                        hashes.add(record["prompt_hash"])
            raw_name = parent + "/raw.jsonl"
            if raw_name in archive.namelist():
                raw_hashes = set()
                with archive.open(raw_name) as handle:
                    for line in handle:
                        record = json.loads(line)
                        assert sha(record["prompt"].encode()) == record["prompt_hash"]
                        assert '"domain":"vitaldb"' not in record["prompt"]
                        raw_hashes.add(record["prompt_hash"])
                assert hashes == raw_hashes, parent
            else:
                assert not hashes and "/baselines/" in name, name
            checked.append({"path": name, "trajectories": len(predictions)})
        predictions = rows(archive, "study_v1/Qwen--Qwen3-8B/predictions.jsonl")

    oracle = {
        r["id"]: r for r in
        [json.loads(line) for line in
         (ROOT / "data/simulated/benchmark_v1/test.oracles.jsonl").read_text().splitlines()]
    }
    groups = defaultdict(dict)
    for record in predictions:
        if (record["domain"], record["kind"], record["condition"], record["mode"]) == (
            "simglucose", "dose", "state_action", "free"
        ) and record["branch"] in {"A", "B"}:
            groups[record["group"]][record["branch"]] = record
    values = defaultdict(list)
    for branches in groups.values():
        a, b = branches["A"], branches["B"]
        assert a["patient"] == b["patient"]
        assert len(a["predictions"]) == len(b["predictions"]) == 12
        assert all(p["valid"] for r in [a, b] for p in r["predictions"])
        predicted = np.array([p["values"][0] for p in a["predictions"]]) - np.array(
            [p["values"][0] for p in b["predictions"]])
        reference = np.array(oracle[a["id"]]["targets"])[:, 0] - np.array(
            oracle[b["id"]]["targets"])[:, 0]
        error = float(np.abs(predicted - reference).mean())
        zero = float(np.abs(reference).mean())
        values[a["patient"]].append([error, zero, error - zero])
    assert len(groups) == 54 and len(values) == 18
    cluster_means = np.array([np.mean(values[key], axis=0) for key in sorted(values)])
    indices = np.random.default_rng(20261102).integers(0, 18, size=(2000, 18))
    bootstrap = cluster_means[indices].mean(axis=1)
    published = json.loads((ROOT / "data/results/study_v1/primary.json").read_text())
    metrics = {}
    for index, metric in enumerate(["effect_mae", "zero_effect_mae", "delta_vs_persistence"]):
        calculated = dict(mean=float(cluster_means[:, index].mean()),
                          lo=float(np.quantile(bootstrap[:, index], .025)),
                          hi=float(np.quantile(bootstrap[:, index], .975)))
        expected = next(r for r in published if r["metric"] == metric)
        assert all(abs(calculated[key] - expected[key]) < 1e-10 for key in calculated), metric
        metrics[metric] = calculated
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  file_hashes_verified=len(manifest["files"]), zip_crc_ok=True,
                  synthetic_exports=checked, primary_pairs=54, primary_identities=18,
                  primary_metrics=metrics, matches_original_results=True,
                  clinical_records_not_included=True)
    dest = ROOT / "data/results/public_verification.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
