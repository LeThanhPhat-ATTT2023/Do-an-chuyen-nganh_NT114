from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.runtime import FastPathPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay packets through the online IDS runtime.")
    parser.add_argument("--config", default="configs/pipeline.example.yaml")
    parser.add_argument("--input", required=True, help="PCAP/PCAPNG file to replay.")
    parser.add_argument("--no-worker", action="store_true", help="Do not start the slow-path worker.")
    parser.add_argument("--max-packets", type=int, default=None)
    return parser.parse_args()


def iter_packets(path: str | Path):
    from scapy.utils import PcapReader

    with PcapReader(str(path)) as reader:
        for packet in reader:
            yield packet


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    pipeline = FastPathPipeline(cfg)
    worker = None
    if not args.no_worker:
        worker = pipeline.start_slow_worker()

    count = 0
    alerts = 0
    try:
        for packet in iter_packets(args.input):
            if args.max_packets is not None and count >= args.max_packets:
                break
            result = pipeline.on_packet(packet)
            count += 1
            if result.alert_id is not None:
                alerts += 1
                print(
                    f"[ALERT] {result.alert_id} flow={result.flow_id} "
                    f"label={result.label} confidence={result.confidence:.4f}"
                )
    finally:
        if worker is not None:
            pipeline.slow_queue.put(None)
            worker.join(timeout=60)

    print(f"[OK] Processed packets={count} alerts={alerts}")
    print(f"[OK] Cold store: {cfg.cold_store_path}")


if __name__ == "__main__":
    main()
