#!/usr/bin/env python3
"""Rehash downloaded model assets against their acquisition manifests."""
from pathlib import Path
import json,hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
def one(p):
    d=json.loads(p.read_text());base=ROOT/d["path"];total=0
    for name,info in d["files"].items():
        f=base/name;assert f.stat().st_size==info["bytes"],str(f)
        with f.open("rb") as handle:actual=hashlib.file_digest(handle,"sha256").hexdigest()
        assert actual==info["sha256"],str(f)
        total+=info["bytes"]
    return dict(model_id=d["model_id"],revision=d["revision"],files=len(d["files"]),bytes=total,all_sha256_match=True)
def main():
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,(ROOT/"data/manifests").glob("*--*.json")))
    assert len(records)==4
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),models=records)
    (ROOT/"data/results/model_integrity.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
