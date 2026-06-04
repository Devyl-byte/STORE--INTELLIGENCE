from __future__ import annotations

import argparse
import json
import time
from urllib import request


def get_json(url: str) -> dict:
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal dashboard for live Store Intelligence metrics.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--store-id", default="ST1008")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run until interrupted.")
    args = parser.parse_args()
    count = 0
    while True:
        metrics = get_json(f"{args.base_url}/stores/{args.store_id}/metrics")
        anomalies = get_json(f"{args.base_url}/stores/{args.store_id}/anomalies")
        print("\033[2J\033[H", end="")
        print("Store Intelligence Live Dashboard")
        print(f"store={args.store_id} as_of={metrics.get('as_of')}")
        print(f"unique_visitors={metrics['unique_visitors']} conversion_rate={metrics['conversion_rate']:.2%}")
        print(f"queue_depth={metrics['current_queue_depth']} abandonment_rate={metrics['abandonment_rate']:.2%}")
        print("avg_dwell_ms_by_zone:")
        for zone, dwell in sorted(metrics["avg_dwell_ms_by_zone"].items())[:10]:
            print(f"  {zone:<18} {dwell:>8.0f}")
        print("anomalies:")
        for anomaly in anomalies.get("anomalies", [])[:5]:
            print(f"  {anomaly['severity']:<8} {anomaly['type']} {anomaly.get('zone_id', '')}")
        count += 1
        if args.iterations and count >= args.iterations:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
