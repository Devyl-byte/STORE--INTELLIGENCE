from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib import request


def post_batch(url: str, batch: list[dict]) -> dict:
    payload = json.dumps(batch).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay JSONL events into the Store Intelligence API.")
    parser.add_argument("--events", type=Path, default=Path("outputs/events.jsonl"))
    parser.add_argument("--url", default="http://localhost:8000/events/ingest")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    batch: list[dict] = []
    sent = 0
    with args.events.open(encoding="utf-8") as handle:
        for line in handle:
            batch.append(json.loads(line))
            if len(batch) >= args.batch_size:
                result = post_batch(args.url, batch)
                sent += len(batch)
                print(f"sent={sent} result={result}")
                batch = []
                time.sleep(args.sleep)
    if batch:
        result = post_batch(args.url, batch)
        sent += len(batch)
        print(f"sent={sent} result={result}")


if __name__ == "__main__":
    main()
